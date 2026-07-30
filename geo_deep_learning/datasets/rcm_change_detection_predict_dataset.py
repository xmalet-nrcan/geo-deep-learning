import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Any

import numpy as np
import rasterio as rio
import torch
from rasterio.transform import Affine
from torch import Tensor

from geo_deep_learning.datasets.rcm_change_detection_dataset import (
    RCMChangeDetectionDataset,
    SatellitePass,
    NO_DATA,
)

logger = logging.getLogger("RCM-PrePost RCMChangeDetectionOnPredictDataset")
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s - %(name)s - [%(levelname)s] ] - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

class RCMChangeDetectionOnPredictDataset(RCMChangeDetectionDataset):
    """Predict-time dataset with optional spatial context from neighboring grid cells.

    When ``predict_overlap_buffer > 0``, each cell's image is expanded by
    *buffer* pixels on every side using data from the 8 adjacent cells on
    the grid.  This lets the model see context beyond the cell boundary,
    and the overlapping predictions from adjacent cells can be blended
    (cosine window) in ``on_predict_end`` to eliminate tile-boundary artefacts.

    The cell grid is inferred from the ``cell_id`` column which must follow
    the ``ROW_COL`` naming convention (e.g. ``"137_42"``).
    """

    def __init__(
        self,
        csv_root_folder: str,
        patches_root_folder: str,
        split_or_csv_file_name: str = None,
        norm_stats: dict[str, list[float]] | None = None,
        bands: Optional[List[int]] = None,
        band_names: Optional[List[str]] = None,
        satellite_pass: Optional[str | SatellitePass] = None,
        beams: Optional[List[str]] = None,
        dataset_years: Optional[list[int]] = None,
        separate_metadata: bool = True,
        tile_size: tuple[int, int] | None = None,
        tile_stride: tuple[int, int] | None = None,
        predict_overlap_buffer: int = 0,
    ) -> None:
        """
        Args:
            predict_overlap_buffer: Number of pixels to load from each
                neighboring cell on every side (top/bottom/left/right).
                Set to 0 (default) to disable spatial context expansion.
                For 50 % overlap between adjacent 200×200 cells, use 100.
        """
        self._predict_overlap_buffer = predict_overlap_buffer
        # Spatial index built lazily after _load_files is called by super().__init__
        self._cell_grid_index: dict[tuple[int, int], list[dict]] | None = None
        super().__init__(
            csv_root_folder, patches_root_folder, split_or_csv_file_name,
            norm_stats, bands, band_names, satellite_pass, beams,
            dataset_years, separate_metadata=separate_metadata,
            tile_size=tile_size, tile_stride=tile_stride,
        )

    # ------------------------------------------------------------------
    # _load_files  (predict CSV schema)
    # ------------------------------------------------------------------

    def _load_files(self) -> list[dict[str, str]]:
        df_csv = self._get_input_dataset_as_dataframe()

        files = []
        for (img_pre,
             img,
             pair_id,
             group_id_pre,
             group_id_post,
             event_id,
             cell_id,
             group_date_pre,
             group_date_post,
             beam,
             sat_pass,
             event_start_date) in df_csv[
            ['pre_input_file',
             'post_input_file',
             "pair_id",
             'group_id_pre',
             'group_id_post',
             'event_id',
             'cell_id',
             'group_date_pre',
             'group_date_post',
             'beam',
             'sat_pass',
             'event_start_date']
        ].itertuples(index=False):
            img_pre_path = img_pre.replace("$ROOT_PATH", self.patches_root_folder).strip()
            img_post_path = img.replace("$ROOT_PATH", self.patches_root_folder).strip()
            if Path(img_pre_path).exists() and Path(img_post_path).exists():
                files.append({
                    "image_pre": img_pre_path,
                    "image": img_post_path,
                    "water_mask": self._get_water_mask_path(cell_id),
                    "pair_id": pair_id,
                    "cell_id": cell_id,
                    "event_id": event_id,
                    "group_date_pre": group_date_pre,
                    "group_date_post": group_date_post,
                    "beam": beam,
                    "sat_pass": sat_pass,
                    "group_id_pre": group_id_pre,
                    "group_id_post": group_id_post,
                    "event_start_date": event_start_date,
                })

        logger.info(
            f"Loaded {len(files)} files for {len(df_csv)} rows. "
            f"Nb of event : {len(df_csv['event_id'].unique())}"
        )

        # Build spatial index for neighbor lookups
        if self._predict_overlap_buffer > 0:
            self._build_cell_grid_index(files)

        return files

    # ------------------------------------------------------------------
    # Spatial index of cells on the regular grid
    # ------------------------------------------------------------------

    def _build_cell_grid_index(self, files: list[dict]) -> None:
        """Build a lookup: (grid_row, grid_col) → list of file entries.

        Multiple entries can share the same cell_id (different pairs/dates).
        The index enables finding neighbours for spatial context expansion.
        """
        self._cell_grid_index = defaultdict(list)
        for entry in files:
            rc = self._parse_cell_id(entry["cell_id"])
            if rc is not None:
                self._cell_grid_index[rc].append(entry)

        logger.info(
            "Built cell grid index: %d unique grid positions from %d entries "
            "(buffer=%d px).",
            len(self._cell_grid_index), len(files), self._predict_overlap_buffer,
        )

    @staticmethod
    def _parse_cell_id(cell_id: str) -> tuple[int, int] | None:
        """Parse ``"ROW_COL"`` cell_id into (row, col) integers."""
        try:
            parts = str(cell_id).split("_")
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------
    # Neighbor-aware image loading
    # ------------------------------------------------------------------

    def _find_neighbor_path(
        self,
        current_entry: dict,
        neighbor_rc: tuple[int, int],
        image_key: str,
    ) -> str | None:
        """Find the path for *image_key* in a neighboring cell.

        Searches entries at grid position *neighbor_rc* that share the same
        beam, sat_pass, group_id_pre, and group_id_post as *current_entry*
        (i.e. same acquisition geometry and temporal pair).
        """
        if self._cell_grid_index is None:
            return None
        candidates = self._cell_grid_index.get(neighbor_rc, [])
        for c in candidates:
            if (c["beam"] == current_entry["beam"]
                    and c["sat_pass"] == current_entry["sat_pass"]
                    and c["group_id_pre"] == current_entry["group_id_pre"]
                    and c["group_id_post"] == current_entry["group_id_post"]):
                path = c.get(image_key)
                if path is not None and Path(str(path)).exists():
                    return str(path)
        return None

    def _load_image_with_buffer(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor, Tensor, str, str, int, int]:
        """Load pre/post images expanded by *buffer* pixels using neighbors.

        Returns:
            image_pre: [C, H+2*buf, W+2*buf]
            image_post: [C, H+2*buf, W+2*buf]
            common_mask: [1, H+2*buf, W+2*buf]
            pre_name, post_name: str
            orig_h, orig_w: original cell dimensions (before buffer)
        """
        data = self.files[index]
        buf = self._predict_overlap_buffer

        # Load the center cell normally
        center_pre_arr, center_pre_mask = self._read_image_and_get_no_data(data["image_pre"])
        center_post_arr, center_post_mask = self._read_image_and_get_no_data(data["image"])
        C, orig_h, orig_w = center_pre_arr.shape

        # Expanded canvas size
        exp_h = orig_h + 2 * buf
        exp_w = orig_w + 2 * buf

        # Initialize with NO_DATA (will be masked out by common_mask)
        expanded_pre = np.full((C, exp_h, exp_w), NO_DATA, dtype=center_pre_arr.dtype)
        expanded_post = np.full((C, exp_h, exp_w), NO_DATA, dtype=center_post_arr.dtype)

        # Place center cell
        expanded_pre[:, buf:buf + orig_h, buf:buf + orig_w] = center_pre_arr
        expanded_post[:, buf:buf + orig_h, buf:buf + orig_w] = center_post_arr

        # Grid position of this cell
        rc = self._parse_cell_id(data["cell_id"])
        if rc is not None:
            r, c_idx = rc
            # 8-connected neighbors: (delta_row, delta_col) → (dst_row_slice, dst_col_slice, src_row_slice, src_col_slice)
            neighbors = self._get_neighbor_slices(buf, orig_h, orig_w)
            for (dr, dc), (dst_r, dst_c, src_r, src_c) in neighbors.items():
                neighbor_rc = (r + dr, c_idx + dc)
                pre_path = self._find_neighbor_path(data, neighbor_rc, "image_pre")
                post_path = self._find_neighbor_path(data, neighbor_rc, "image")
                if pre_path is None or post_path is None:
                    continue  # No neighbor → stays NO_DATA (will be masked)
                try:
                    n_pre, _ = self._read_image_and_get_no_data(pre_path)
                    n_post, _ = self._read_image_and_get_no_data(post_path)
                    expanded_pre[:, dst_r, dst_c] = n_pre[:, src_r, src_c]
                    expanded_post[:, dst_r, dst_c] = n_post[:, src_r, src_c]
                except Exception as e:
                    logger.debug("Could not load neighbor %s: %s", neighbor_rc, e)

        # Build common mask from bitmask (band 0 == 1 → valid)
        pre_mask = expanded_pre[0, :, :] == 1
        post_mask = expanded_post[0, :, :] == 1
        common_mask = pre_mask & post_mask
        common_mask_tensor = torch.from_numpy(common_mask).unsqueeze(0)  # [1, H, W]

        image_pre_tensor = torch.from_numpy(expanded_pre).float()
        image_post_tensor = torch.from_numpy(expanded_post).float()

        pre_name = Path(data["image_pre"]).name
        post_name = Path(data["image"]).name

        return image_pre_tensor, image_post_tensor, common_mask_tensor, pre_name, post_name, orig_h, orig_w

    @staticmethod
    def _get_neighbor_slices(
        buf: int, h: int, w: int,
    ) -> dict[tuple[int, int], tuple[slice, slice, slice, slice]]:
        """Compute source/dest slices for each of the 8 neighbors.

        Cell ID convention: ``"X_Y"`` where X increases **rightward**
        (image column direction) and Y increases **upward** (opposite to
        image row direction in a north-up GeoTIFF).

        So the deltas ``(dr, dc)`` applied to ``(X, Y)`` mean:
        - ``dr = +1`` → RIGHT neighbor (higher column indices)
        - ``dr = -1`` → LEFT  neighbor (lower column indices)
        - ``dc = +1`` → UP    neighbor (lower row indices in image)
        - ``dc = -1`` → DOWN  neighbor (higher row indices in image)

        For a grid cell of size (h, w) with buffer *buf*:
        - RIGHT neighbor (dr=+1): take its left *buf* columns
        - LEFT  neighbor (dr=-1): take its right *buf* columns
        - UP    neighbor (dc=+1): take its bottom *buf* rows
        - DOWN  neighbor (dc=-1): take its top *buf* rows
        - Corners: take the corresponding buf×buf rectangle

        Returns:
            Dict of ``(dr, dc)`` → ``(dst_row_slice, dst_col_slice,
            src_row_slice, src_col_slice)``
        """
        # Full column range for vertically-aligned neighbours (UP/DOWN)
        center_cols_dst = slice(buf, buf + w)
        center_cols_src = slice(0, w)
        # Full row range for horizontally-aligned neighbours (LEFT/RIGHT)
        center_rows_dst = slice(buf, buf + h)
        center_rows_src = slice(0, h)

        return {
            # --- Cardinal neighbours ---
            (+1, 0): (  # RIGHT  (X+1) → place at right side of canvas
                center_rows_dst, slice(buf + w, buf + w + buf),
                center_rows_src, slice(0, buf),
            ),
            (-1, 0): (  # LEFT   (X-1) → place at left side of canvas
                center_rows_dst, slice(0, buf),
                center_rows_src, slice(w - buf, w),
            ),
            (0, +1): (  # UP     (Y+1) → place at top of canvas (lower rows)
                slice(0, buf), center_cols_dst,
                slice(h - buf, h), center_cols_src,
            ),
            (0, -1): (  # DOWN   (Y-1) → place at bottom of canvas (higher rows)
                slice(buf + h, buf + h + buf), center_cols_dst,
                slice(0, buf), center_cols_src,
            ),
            # --- Diagonal neighbours ---
            (+1, +1): (  # RIGHT+UP → top-right corner
                slice(0, buf), slice(buf + w, buf + w + buf),
                slice(h - buf, h), slice(0, buf),
            ),
            (+1, -1): (  # RIGHT+DOWN → bottom-right corner
                slice(buf + h, buf + h + buf), slice(buf + w, buf + w + buf),
                slice(0, buf), slice(0, buf),
            ),
            (-1, +1): (  # LEFT+UP → top-left corner
                slice(0, buf), slice(0, buf),
                slice(h - buf, h), slice(w - buf, w),
            ),
            (-1, -1): (  # LEFT+DOWN → bottom-left corner
                slice(buf + h, buf + h + buf), slice(0, buf),
                slice(0, buf), slice(w - buf, w),
            ),
        }

    def _load_image(self, index: int) -> tuple[Tensor, Tensor, Tensor, str, str]:
        """Override to use neighbor-aware loading when buffer > 0."""
        if self._predict_overlap_buffer > 0 and self._cell_grid_index is not None:
            pre, post, mask, pre_name, post_name, orig_h, orig_w = (
                self._load_image_with_buffer(index)
            )
            # Store original dimensions on the file entry for profile adjustment
            self.files[index]["_buffer_orig_h"] = orig_h
            self.files[index]["_buffer_orig_w"] = orig_w
            return pre, post, mask, pre_name, post_name
        return super()._load_image(index)

    def __getitem__(self, index: int) -> dict:
        """Override to adjust GeoTIFF profile for the expanded area."""
        sample = super().__getitem__(index)

        data = self.files[index]
        buf = self._predict_overlap_buffer
        if buf > 0 and "_buffer_orig_h" in data:
            orig_h = data["_buffer_orig_h"]
            orig_w = data["_buffer_orig_w"]

            # Adjust the GeoTIFF transform to account for the buffer offset.
            # The origin shifts by -buf pixels in both row and col directions.
            transform_list = sample["profile"]["transform"]
            orig_transform = Affine(*transform_list[:6])
            # Shift origin: buffer pixels to the left and up
            expanded_transform = orig_transform * Affine.translation(-buf, -buf)
            sample["profile"]["transform"] = list(expanded_transform)

            # Update dimensions to reflect the expanded image
            sample["original_height"] = sample["image"].shape[1]
            sample["original_width"] = sample["image"].shape[2]

            # Store buffer metadata for on_predict_end to know the core area
            sample["_overlap_buffer"] = buf
            sample["_core_height"] = orig_h
            sample["_core_width"] = orig_w

        return sample

    # ------------------------------------------------------------------
    # Metadata helpers (predict-specific)
    # ------------------------------------------------------------------

    def _get_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """Metadata specific to the predict CSV schema."""
        return {
            "pair_id": int(data["pair_id"]) if data["pair_id"] is not None else -1,
            "event_id": int(data["event_id"]) if data["event_id"] is not None else -1,
            "event_start_date": str(data["event_start_date"]) if data["event_start_date"] is not None else "",
            "group_id_pre": int(data["group_id_pre"]) if data["group_id_pre"] is not None else -1,
            "group_id_post": int(data["group_id_post"]) if data["group_id_post"] is not None else -1,
            "group_date_pre": str(data["group_date_pre"]) if data["group_date_pre"] is not None else "",
            "group_date_post": str(data["group_date_post"]) if data["group_date_post"] is not None else "",
        }

    def _get_pre_post_name(self, data: dict[str, str]) -> str:
        """Build identifier adapted to predict CSV columns."""
        return (
            f"{data['cell_id']}_event_{data['event_id']}_pair_{data['pair_id']}|"
            f"{'ASC' if data['sat_pass'] == SatellitePass.ASCENDING else 'DESC'}-{data['beam'].name}|"
            f"({data['group_id_pre']}){data['group_date_pre']}_"
            f"({data['group_id_post']}){data['group_date_post']}|"
            f"event_({data['event_id']})_{data['event_start_date']}"
        )
