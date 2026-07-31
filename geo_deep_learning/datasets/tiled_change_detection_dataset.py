"""Intermediate dataset adding tiling and spatial-context buffer to any CSV dataset.

Sits between :class:`ChangeDetectionDataset` and domain-specific datasets such
as :class:`RCMChangeDetectionDataset`.  Callers only need to implement
:meth:`_build_raw_file_list` to supply their per-sample file dicts.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
import torch
from rasterio.transform import Affine
from torch import Tensor

from geo_deep_learning.datasets.change_detection_dataset import ChangeDetectionDataset

logger = logging.getLogger(__name__)


def _read_image_and_get_no_data_default(
    path: str,
    in_dtype: np.dtype = np.int16,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback image reader used when a subclass does not override *_read_image_and_get_no_data*."""
    with rio.open(path) as src:
        arr = src.read().astype(in_dtype)
    return arr, (arr[0, :, :] == 1)


class TiledChangeDetectionDataset(ChangeDetectionDataset):
    """``ChangeDetectionDataset`` extended with tiling and spatial-context buffer.

    Responsibilities
    ----------------
    * **Tiling** – splits large images into a grid of fixed-size tiles at load
      time (:meth:`_expand_files_with_tiles`).
    * **Spatial-context buffer** – expands each cell by loading pixels from its
      8 neighbouring cells (:meth:`_load_image_with_buffer`).
    * **Profile management** – adjusts the GeoTIFF affine transform to reflect
      the tile / buffer position in :meth:`_apply_tile_crop`.
    * **Template method** – :meth:`_load_files` orchestrates
      ``_build_raw_file_list → grid-index → tiling``.  Subclasses only need to
      implement :meth:`_build_raw_file_list`.

    Non-responsibilities (delegated to subclasses)
    -----------------------------------------------
    * CSV column schema parsing
    * Band / satellite-pass / beam filtering
    * Ground-truth mask loading
    * Normalisation / standardisation
    * FiLM conditioning metadata
    """

    # Default nodata value; subclasses may override.
    NO_DATA: int = 32767

    @staticmethod
    def _read_image_and_get_no_data(
        path: str,
        in_dtype: np.dtype = np.int16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read a GeoTIFF; return ``(array [C,H,W], bitmask [H,W])``.

        Subclasses may override to change the dtype default or mask logic.
        """
        return _read_image_and_get_no_data_default(path, in_dtype)

    def __init__(
        self,
        csv_root_folder: str,
        patches_root_folder: str,
        split_or_csv_file_name: str = None,
        norm_stats: dict[str, list[float]] | None = None,
        *,
        tile_size: tuple[int, int] | None = None,
        tile_stride: tuple[int, int] | None = None,
        predict_overlap_buffer: int = 0,
    ) -> None:
        """
        Args:
            tile_size: (height, width) of each tile.  ``None`` disables tiling.
            tile_stride: Step between tile origins.  Defaults to *tile_size*
                (no overlap).  Set smaller for overlapping tiles.
            predict_overlap_buffer: Pixels loaded from each neighbouring cell
                on every side.  0 disables the feature.
        """
        # Must be set BEFORE super().__init__() because it triggers _load_files().
        self.tile_size = tile_size
        self.tile_stride = tile_stride if tile_stride is not None else tile_size
        self._predict_overlap_buffer: int = predict_overlap_buffer
        self._cell_grid_index: dict[tuple[int, int], list[dict[str, Any]]] = {}

        super().__init__(
            csv_root_folder, patches_root_folder,
            split_or_csv_file_name, norm_stats,
        )

    # ------------------------------------------------------------------
    # Template method: _load_files orchestrates the full pipeline
    # ------------------------------------------------------------------

    def _load_files(self) -> list[dict[str, Any]]:
        """Load file entries, build the spatial index, and apply tiling.

        Subclasses provide the per-sample file dicts via
        :meth:`_build_raw_file_list`; this method handles the rest.
        """
        files = self._build_raw_file_list()

        if self._predict_overlap_buffer > 0:
            self._build_cell_grid_index(files)

        if self.tile_size is not None:
            if self._predict_overlap_buffer > 0:
                files = self._expand_files_with_tiles_buffered(files)
            else:
                files = self._expand_files_with_tiles(files)

        return files

    def _build_raw_file_list(self) -> list[dict[str, Any]]:
        """Return the raw per-sample file dicts (no tiling, no index).

        **Must be overridden** by every concrete subclass to supply the
        dataset-specific CSV schema.

        Returns
        -------
        list[dict]
            Each dict must contain at least ``"image_pre"`` and ``"image"``
            keys with valid file paths, plus whatever additional keys are
            needed by the subclass (``"cell_id"``, ``"group_id_pre"``, …).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_raw_file_list()."
        )

    # ------------------------------------------------------------------
    # Tiling helpers
    # ------------------------------------------------------------------

    def _expand_files_with_tiles(self, files: list[dict]) -> list[dict]:
        """Split images larger than *tile_size* into a grid of tiles.

        Images that already fit within *tile_size* are kept as-is.
        Edge tiles are shifted inward so every tile is exactly *tile_size*.
        """
        tile_h, tile_w = self.tile_size
        stride_h, stride_w = self.tile_stride
        n_files = len(files)

        expanded: list[dict] = []
        for entry in files:
            with rio.open(entry["image"]) as src:
                img_h, img_w = src.height, src.width

            if img_h <= tile_h and img_w <= tile_w:
                expanded.append(entry)
                continue

            rows = sorted(set(
                list(range(0, max(img_h - tile_h, 0) + 1, stride_h))
                + ([max(0, img_h - tile_h)] if img_h > tile_h else [0])
            ))
            cols = sorted(set(
                list(range(0, max(img_w - tile_w, 0) + 1, stride_w))
                + ([max(0, img_w - tile_w)] if img_w > tile_w else [0])
            ))

            for r in rows:
                for c in cols:
                    tile_entry = entry.copy()
                    tile_entry["_tile_row"] = r
                    tile_entry["_tile_col"] = c
                    tile_entry["_source_h"] = img_h
                    tile_entry["_source_w"] = img_w
                    expanded.append(tile_entry)

        if len(expanded) != n_files:
            logger.info(
                "Tile expansion: %d files → %d tiles (tile_size=%s, stride=%s)",
                n_files, len(expanded), self.tile_size, self.tile_stride,
            )
        return expanded

    def _expand_files_with_tiles_buffered(self, files: list[dict]) -> list[dict]:
        """Like :meth:`_expand_files_with_tiles` but uses the effective buffered size."""
        buf = self._predict_overlap_buffer
        tile_h, tile_w = self.tile_size
        stride_h, stride_w = self.tile_stride
        n_files = len(files)

        expanded: list[dict] = []
        for entry in files:
            with rio.open(entry["image"]) as src:
                file_h, file_w = src.height, src.width
            img_h = file_h + 2 * buf
            img_w = file_w + 2 * buf

            if img_h <= tile_h and img_w <= tile_w:
                expanded.append(entry)
                continue

            rows = sorted(set(
                list(range(0, max(img_h - tile_h, 0) + 1, stride_h))
                + ([max(0, img_h - tile_h)] if img_h > tile_h else [0])
            ))
            cols = sorted(set(
                list(range(0, max(img_w - tile_w, 0) + 1, stride_w))
                + ([max(0, img_w - tile_w)] if img_w > tile_w else [0])
            ))

            for r in rows:
                for c in cols:
                    tile_entry = entry.copy()
                    tile_entry["_tile_row"] = r
                    tile_entry["_tile_col"] = c
                    tile_entry["_source_h"] = img_h
                    tile_entry["_source_w"] = img_w
                    expanded.append(tile_entry)

        if len(expanded) != n_files:
            logger.info(
                "Tile expansion (buffered): %d files → %d tiles "
                "(tile_size=%s, stride=%s, buffer=%d)",
                n_files, len(expanded), self.tile_size, self.tile_stride, buf,
            )
        return expanded

    def _apply_tile_crop(self, sample: dict, data: dict) -> dict:
        """Crop spatial tensors and adjust the GeoTIFF profile for a tile.

        Also applies the buffer transform offset when a spatial-context
        buffer is active.  No-op when tiling is inactive.

        Transform adjustment order:
        1. Buffer offset  (expands the origin by ``-buf`` pixels).
        2. Tile-crop offset (shifts the origin by ``(tile_col, tile_row)``).
        """
        # 1. Buffer: shift the GeoTIFF origin by -buf pixels
        buf = self._predict_overlap_buffer
        if buf > 0 and "_buffer_orig_h" in data and sample.get("profile") is not None:
            tl = sample["profile"]["transform"]
            buffered = Affine(*tl[:6]) * Affine.translation(-buf, -buf)
            sample["profile"]["transform"] = list(buffered)

        if "_tile_row" not in data:
            return sample

        r, c = data["_tile_row"], data["_tile_col"]
        th, tw = self.tile_size

        # Crop every spatial tensor [C, H, W] → [C, th, tw]
        for key in ("image", "image_post", "image_pre", "mask", "mask-common", "water_mask"):
            t = sample.get(key)
            if t is not None and isinstance(t, Tensor) and t.dim() >= 3:
                sample[key] = t[:, r:r + th, c:c + tw]

        # 2. Tile-crop: shift the GeoTIFF origin by (c, r) pixels
        if sample.get("profile") is not None:
            tl = sample["profile"]["transform"]
            shifted = Affine(*tl[:6]) * Affine.translation(c, r)
            sample["profile"]["transform"] = list(shifted)

        # Update sample dimensions to reflect the tile
        sample["original_height"] = sample["image"].shape[1]
        sample["original_width"] = sample["image"].shape[2]
        sample["tile_row_start"] = r
        sample["tile_col_start"] = c
        sample["source_height"] = data["_source_h"]
        sample["source_width"] = data["_source_w"]

        return sample

    # ------------------------------------------------------------------
    # Spatial-context buffer helpers
    # ------------------------------------------------------------------

    def _build_cell_grid_index(self, files: list[dict[str, Any]]) -> None:
        """Build a ``(grid_row, grid_col) → [file_entry, …]`` lookup."""
        index: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for entry in files:
            rc = self._parse_cell_id(entry["cell_id"])
            if rc is not None:
                index[rc].append(entry)
        self._cell_grid_index = dict(index)
        logger.info(
            "Built cell grid index: %d positions from %d entries (buffer=%d px).",
            len(self._cell_grid_index), len(files), self._predict_overlap_buffer,
        )

    @staticmethod
    def _parse_cell_id(cell_id: str) -> tuple[int, int] | None:
        """Parse a ``"ROW_COL"`` cell_id into ``(row, col)`` integers."""
        try:
            parts = str(cell_id).split("_")
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _get_neighbor_slices(
        buf: int, h: int, w: int,
    ) -> dict[tuple[int, int], tuple[slice, slice, slice, slice]]:
        """Return source/destination slice pairs for each of the 8 neighbours."""
        cr_dst = slice(buf, buf + w)
        cr_src = slice(0, w)
        rr_dst = slice(buf, buf + h)
        rr_src = slice(0, h)
        return {
            (+1,  0): (rr_dst, slice(buf + w, buf + w + buf), rr_src, slice(0, buf)),
            (-1,  0): (rr_dst, slice(0, buf), rr_src, slice(w - buf, w)),
            ( 0, +1): (slice(0, buf), cr_dst, slice(h - buf, h), cr_src),
            ( 0, -1): (slice(buf + h, buf + h + buf), cr_dst, slice(0, buf), cr_src),
            (+1, +1): (slice(0, buf), slice(buf + w, buf + w + buf), slice(h - buf, h), slice(0, buf)),
            (+1, -1): (slice(buf + h, buf + h + buf), slice(buf + w, buf + w + buf), slice(0, buf), slice(0, buf)),
            (-1, +1): (slice(0, buf), slice(0, buf), slice(h - buf, h), slice(w - buf, w)),
            (-1, -1): (slice(buf + h, buf + h + buf), slice(0, buf), slice(0, buf), slice(w - buf, w)),
        }

    def _find_neighbor_path(
        self,
        current_entry: dict[str, Any],
        neighbor_rc: tuple[int, int],
        image_key: str,
    ) -> str | None:
        """Find the *image_key* path in a neighbouring cell with matching geometry."""
        if not self._cell_grid_index:
            return None
        neighbors: list[dict[str, Any]] = self._cell_grid_index.get(neighbor_rc, [])
        for c in neighbors:
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
        """Load pre/post images expanded by *buffer* pixels using neighbours.

        Returns
        -------
        image_pre  : ``[C, H+2b, W+2b]``
        image_post : ``[C, H+2b, W+2b]``
        common_mask: ``[1, H+2b, W+2b]``
        pre_name, post_name : str
        orig_h, orig_w : int  — original (un-expanded) cell dimensions
        """
        data = self.files[index]
        buf = self._predict_overlap_buffer
        no_data = self.__class__.NO_DATA

        center_pre, _ = self._read_image_and_get_no_data(data["image_pre"])
        center_post, _ = self._read_image_and_get_no_data(data["image"])
        C, orig_h, orig_w = center_pre.shape

        exp_h, exp_w = orig_h + 2 * buf, orig_w + 2 * buf
        exp_pre  = np.full((C, exp_h, exp_w), no_data, dtype=center_pre.dtype)
        exp_post = np.full((C, exp_h, exp_w), no_data, dtype=center_post.dtype)
        exp_pre [:, buf:buf + orig_h, buf:buf + orig_w] = center_pre
        exp_post[:, buf:buf + orig_h, buf:buf + orig_w] = center_post

        rc = self._parse_cell_id(data["cell_id"])
        if rc is not None:
            r, c_idx = rc
            for (dr, dc), (dst_r, dst_c, src_r, src_c) in (
                self._get_neighbor_slices(buf, orig_h, orig_w).items()
            ):
                nb_rc = (r + dr, c_idx + dc)
                pre_path  = self._find_neighbor_path(data, nb_rc, "image_pre")
                post_path = self._find_neighbor_path(data, nb_rc, "image")
                if pre_path is None or post_path is None:
                    continue
                try:
                    n_pre,  _ = self._read_image_and_get_no_data(pre_path)
                    n_post, _ = self._read_image_and_get_no_data(post_path)
                    exp_pre [:, dst_r, dst_c] = n_pre [:, src_r, src_c]
                    exp_post[:, dst_r, dst_c] = n_post[:, src_r, src_c]
                except Exception as e:
                    logger.debug("Could not load neighbour %s: %s", nb_rc, e)

        common = torch.from_numpy((exp_pre[0] == 1) & (exp_post[0] == 1)).unsqueeze(0)
        return (
            torch.from_numpy(exp_pre).float(),
            torch.from_numpy(exp_post).float(),
            common,
            Path(data["image_pre"]).name,
            Path(data["image"]).name,
            orig_h, orig_w,
        )

    # Override _load_image to activate buffer loading when requested
    def _load_image(self, index: int) -> tuple[Tensor, Tensor, Tensor, str, str]:
        if self._predict_overlap_buffer > 0 and self._cell_grid_index:
            pre, post, mask, pre_name, post_name, orig_h, orig_w = (
                self._load_image_with_buffer(index)
            )
            # Cache original dimensions so _apply_tile_crop and _finalize_sample
            # can compute the buffer zone.
            self.files[index]["_buffer_orig_h"] = orig_h
            self.files[index]["_buffer_orig_w"] = orig_w
            return pre, post, mask, pre_name, post_name
        return super()._load_image(index)

    def _apply_buffer_padding(
        self,
        mask: Tensor,
        data: dict,
    ) -> Tensor:
        """Pad a single-cell mask/raster to the buffered image size.

        Call this from :meth:`_load_water_mask` (and similar loaders) after
        loading the raw file-level tensor.

        The padding value is 0 (= invalid / no-water); the common_mask from
        the bitmask band handles real invalidity in the buffer zone.
        """
        buf = self._predict_overlap_buffer
        if buf <= 0 or "_buffer_orig_h" not in data:
            return mask
        _, orig_h, orig_w = mask.shape
        exp_h = orig_h + 2 * buf
        exp_w = orig_w + 2 * buf
        padded = torch.zeros((1, exp_h, exp_w), dtype=mask.dtype)
        padded[:, buf:buf + orig_h, buf:buf + orig_w] = mask
        return padded

    # ------------------------------------------------------------------
    # _finalize_sample  — called at the END of every subclass __getitem__
    # ------------------------------------------------------------------

    def _finalize_sample(self, sample: dict, data: dict) -> dict:
        """Apply tile crop and attach buffer metadata to the sample.

        Subclasses should call this as the **last step** of their
        ``__getitem__`` implementation, replacing the manual call to
        ``_apply_tile_crop`` plus the buffer-metadata block.

        Example::

            def __getitem__(self, index):
                # ... build sample dict ...
                return self._finalize_sample(sample, self.files[index])
        """
        sample = self._apply_tile_crop(sample, data)

        # Expose buffer metadata so the model can restrict the loss to the
        # central-cell pixels (see _forward_and_get_loss in the model).
        buf = self._predict_overlap_buffer
        if buf > 0 and "_buffer_orig_h" in data:
            if "_tile_row" not in data:
                # Non-tiled path: dimensions have not been updated yet
                sample["original_height"] = sample["image"].shape[1]
                sample["original_width"]  = sample["image"].shape[2]
            sample["buffer_size"]       = buf
            sample["cell_orig_height"]  = int(data["_buffer_orig_h"])
            sample["cell_orig_width"]   = int(data["_buffer_orig_w"])

        return sample
