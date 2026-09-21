"""RCM Change Detection Dataset (training schema).

Hierarchy
---------
``CSVDataset``
  └── ``ChangeDetectionDataset``
        └── ``TiledChangeDetectionDataset``   ← tiling + spatial-context buffer
              └── ``RCMChangeDetectionDataset``  ← RCM CSV schema + bands + FiLM metadata
                    └── ``RCMChangeDetectionOnPredictDataset``  ← predict CSV schema
"""
from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from pandas import DataFrame
from torch import Tensor

from geo_deep_learning.datasets.tiled_change_detection_dataset import TiledChangeDetectionDataset
from geo_deep_learning.utils.tensors import manage_bands

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain enumerations & constants
# ---------------------------------------------------------------------------

class SatellitePass(Enum):
    ASCENDING = 0
    DESCENDING = 1

    @classmethod
    def from_str(cls, s: str) -> "SatellitePass":
        translate = {
            "A": "Ascending", "D": "Descending",
            "ASC": "Ascending", "DESC": "Descending",
            "ASCENDING": "Ascending", "DESCENDING": "Descending",
        }
        try:
            return cls[translate[s.upper()].upper()]
        except KeyError:
            raise ValueError(f"Satellite pass {s!r} not recognized.") from None


class BandName(Enum):
    BITMASK_CROPPED = 1
    LOCALINCANGLE   = 2
    M               = 3
    NDSV            = 4
    PDN             = 5
    PSN             = 6
    PVN             = 7
    RFDI            = 8
    RL              = 9
    RR              = 10
    S0              = 11
    SP1             = 12
    SP2             = 13
    SP3             = 14


class Beams(Enum):
    A = 0
    B = 1
    C = 2
    D = 3


BEAM_BAND_NAME           = "BEAM"
SATELLITE_PASS_BAND_NAME = "SATELLITE_PASS"

# Stats updated 2026-06-03 with 201 286 thumbnails
bands_stats = {
    'mean': [1.0671012440715284, 26.133220506930545, 4282.53025830744, -1762.5102994669658,
             238.6081335327169, 4043.3150312687353, 5716.365492395923, 3587.3789677071168,
             1366.3266999839548, 606.6409300828175, 1972.7496275401488, 662.1524939064722,
             585.9621173767042, 8360.947639324142],
    'std':  [0.5140239082593165, 7.776109811847935, 2096.273627364101, 4390.274997471279,
             337.8120308539174, 2197.621407866461, 2096.399549952241, 2099.0776151168197,
             965.7990528898996, 395.7478935818052, 1149.0959360818918, 2805.261797215982,
             3114.6516936980624, 2463.9644096193365],
    'min':  [1.0, 0.0, 0.0, -9998.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0, -9998.0, -9995.0],
    'max':  [17.0, 122.0, 9999.0, 9998.0, 9967.0, 9998.0, 9964.0, 9999.0, 32766.0, 32766.0,
             32766.0, 9999.0, 9999.0, 10000.0],
}

# Time-delta bin boundaries in days (upper bound exclusive).
TIME_DELTA_BINS = [4, 12, 24, 48]

NO_DATA      = 32767
IGNORE_INDEX = 255


def band_names_to_indices(band_names: Optional[List[Any]]) -> Optional[List[int]]:
    """Convert a list of band names (str or BandName) to integer indices."""
    if band_names is None:
        return None
    logger.info("Treating bands: %s", band_names)
    indices = []
    for name in band_names:
        if isinstance(name, BandName):
            indices.append(name.value)
        elif isinstance(name, str):
            try:
                indices.append(BandName[name].value)
            except KeyError:
                raise ValueError(f"Unknown band name: {name!r}") from None
        else:
            raise TypeError(f"Unsupported type for band_names: {type(name)}")
    return indices


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class RCMChangeDetectionDataset(TiledChangeDetectionDataset):
    """RCM SAR change-detection dataset for **training**.

    Extends :class:`TiledChangeDetectionDataset` with:

    * Training CSV schema (``pre_path``, ``post_path``, ``mask``, …)
    * Band / satellite-pass / beam / year filtering
    * RCM-specific image loading (int16, bitmask band)
    * Ground-truth fire mask loading (NBaC)
    * FiLM conditioning metadata (``sat_pass_value``, ``beam_value``,
      ``pre_season``, ``post_season``, ``time_delta_bin``)
    * GeoTIFF profile management for output predictions

    Spatial concerns (tiling, neighbour-buffer) are fully handled by
    :class:`TiledChangeDetectionDataset`.
    """

    NO_DATA = NO_DATA  # re-expose module constant as class attribute

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
        self.separate_metadata = separate_metadata

        # Band setup — must precede super().__init__() (which triggers _load_files)
        if band_names is not None:
            self.bands = band_names_to_indices(band_names)
            self.band_names = [
                bn.name if isinstance(bn, BandName) else str(bn) for bn in band_names
            ]
        elif bands is not None:
            self.bands = bands
            self.band_names = []
            for idx in bands:
                try:
                    self.band_names.append(BandName(idx).name)
                except ValueError:
                    self.band_names.append(str(idx))
        else:
            self.bands = [i.value for i in BandName]
            self.band_names = [i.name for i in BandName]

        match satellite_pass:
            case None:
                self.satellite_pass = None
            case str():
                self.satellite_pass = SatellitePass.from_str(satellite_pass)
            case SatellitePass():
                self.satellite_pass = satellite_pass
            case _:
                raise TypeError("satellite_pass must be a string or SatellitePass enum")

        self.beams = [] if beams is None else [b.upper() for b in beams]
        if norm_stats is None:
            norm_stats = bands_stats
        logger.debug("dataset_years ==> %s", dataset_years)
        self._dataset_years = [] if dataset_years is None else [int(y) for y in dataset_years]

        # Tiling / buffer params forwarded to TiledChangeDetectionDataset
        super().__init__(
            csv_root_folder=csv_root_folder,
            patches_root_folder=patches_root_folder,
            split_or_csv_file_name=split_or_csv_file_name,
            norm_stats=norm_stats,
            tile_size=tile_size,
            tile_stride=tile_stride,
            predict_overlap_buffer=predict_overlap_buffer,
        )

    # ------------------------------------------------------------------
    # _build_raw_file_list — training CSV schema
    # ------------------------------------------------------------------

    def _build_raw_file_list(self) -> list[dict[str, Any]]:
        """Load per-sample dicts from the training CSV."""
        df_csv = self._get_input_dataset_as_dataframe()

        files = []
        for (img_pre, img, group_id_pre, group_id_post, db_nbac_fire_id,
             cell_id, group_date_pre, group_date_post, beam, sat_pass,
             fire_start_date, fire_end_date) in df_csv[
            ['pre_path', 'post_path', 'group_id_pre', 'group_id_post',
             'db_nbac_fire_id', 'cell_id', 'group_date_pre', 'group_date_post',
             'beam', 'sat_pass', 'fire_start_date', 'fire_end_date']
        ].itertuples(index=False):
            resolved = self._resolve_pre_post_paths(img_pre, img)
            if resolved is None:
                continue
            img_pre_path, img_post_path = resolved
            files.append({
                "image_pre":       img_pre_path,
                "image":           img_post_path,
                "mask":            self._get_mask_path(cell_id, group_date_post),
                "water_mask":      self._get_water_mask_path(cell_id),
                "cell_id":         cell_id,
                "db_nbac_fire_id": db_nbac_fire_id,
                "group_date_pre":  group_date_pre,
                "group_date_post": group_date_post,
                "beam":            beam,
                "sat_pass":        sat_pass,
                "group_id_pre":    group_id_pre,
                "group_id_post":   group_id_post,
                "fire_start_date": fire_start_date,
                "fire_end_date":   fire_end_date,
            })

        logger.info(
            "Loaded %d entries (%d with mask, %d without mask)",
            len(files),
            sum(1 for f in files if f["mask"] is not None),
            sum(1 for f in files if f["mask"] is None),
        )
        return files

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve_pre_post_paths(self, raw_pre: str, raw_post: str) -> tuple[str, str] | None:
        """Resolve ``$ROOT_PATH`` placeholders and verify both files exist.

        Returns ``None`` when either file is missing so the caller can skip
        the CSV row. Shared by the training and predict-time CSV loaders.
        """
        pre_path = raw_pre.replace("$ROOT_PATH", self.patches_root_folder).strip()
        post_path = raw_post.replace("$ROOT_PATH", self.patches_root_folder).strip()
        if Path(pre_path).exists() and Path(post_path).exists():
            return pre_path, post_path
        return None

    def _get_water_mask_path(self, cell_id) -> Path | None:
        p = (
            Path(self.patches_root_folder) / cell_id / "static_data"
            / f"{cell_id}_WATER_mask_100m.tif"
        )
        if not p.exists():
            logger.debug("Water mask not found: %s", p)
            return None
        return p

    def _get_mask_path(self, cell_id, group_date_post) -> Path | None:
        p = (
            Path(self.patches_root_folder) / cell_id / "static_data"
            / f"{cell_id}_nbac_{int(group_date_post[:4])}_mask_unburn_burn_reject_100m.tif"
        )
        if not p.exists():
            logger.debug("Mask not found: %s", p)
            return None
        return p

    # ------------------------------------------------------------------
    # Image / mask loaders
    # ------------------------------------------------------------------

    def _load_mask(self, index: int) -> tuple[Tensor, str]:
        mask, name = self._load_image_by_name(index, "mask")
        mask = self._apply_buffer_padding(mask, self.files[index])
        return mask, name

    def _load_water_mask(self, index: int) -> tuple[Tensor, str]:
        # When a spatial-context buffer is active the pre/post images are
        # expanded with *real* neighbour imagery, so the model predicts across
        # the whole expanded tile. The water mask must therefore also carry the
        # neighbours' water in the buffer ring; zero-padding it would leave
        # water pixels near the cell borders (and in cross-cell merge overlaps)
        # unmasked.
        if self._predict_overlap_buffer > 0 and getattr(self, "_cell_grid_index", None):
            return self._load_static_raster_with_buffer(index, "water_mask", fill_value=0.0)
        water_mask, name = self._load_image_by_name(index, "water_mask")
        water_mask = self._apply_buffer_padding(water_mask, self.files[index])
        return water_mask, name

    def _load_image_by_name(
        self, index: int, key: str, as_type=np.int32,
    ) -> tuple[Tensor, str]:
        data = self.files[index]
        path = data.get(key)
        if path is not None and Path(str(path)).exists():
            arr, _ = self._read_image_and_get_no_data(str(path), as_type)
            return torch.from_numpy(arr).float(), Path(str(path)).name
        height, width = self._get_raster_dimensions(data["image"])
        return torch.zeros((1, height, width), dtype=torch.float32), f"no_{key}"

    # ------------------------------------------------------------------
    # CSV filtering
    # ------------------------------------------------------------------

    def _get_input_dataset_as_dataframe(self) -> DataFrame:
        csv_path = self._get_csv_path()
        df_csv = pd.read_csv(csv_path)

        logger.info("BEAM FILTER: %s", self.beams)
        logger.info("SATELLITE PASS FILTER: %s", self.satellite_pass)
        logger.info("DATASET YEARS FILTER: %s", self._dataset_years)

        df_csv['sat_pass'] = df_csv['sat_pass'].map(SatellitePass.from_str)
        df_csv['beam']     = df_csv['beam'].map(lambda x: Beams[x.upper()])

        if self.satellite_pass is not None:
            df_csv = df_csv[df_csv['sat_pass'] == self.satellite_pass]
        if self.beams:
            df_csv = df_csv[df_csv['beam'].apply(lambda x: x.name in self.beams)]
        if self._dataset_years:
            df_csv = df_csv[df_csv['group_date_pre'].apply(
                lambda x: int(str(x)[:4]) in self._dataset_years
            )]

        logger.info("Loaded %d rows from CSV after filtering.", len(df_csv))
        return df_csv

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        data = self.files[index]
        image_pre, image_post, common_mask_tensor, image_pre_name, image_post_name = (
            self._load_image(index)
        )

        mask, mask_name, has_mask = self._load_ground_truth_mask(index, common_mask_tensor, image_pre)

        image_pre = self._apply_common_mask_to_tensor(common_mask_tensor, image_pre, IGNORE_INDEX)
        image_post = self._apply_common_mask_to_tensor(common_mask_tensor, image_post, IGNORE_INDEX)

        bands_index = self._get_bands_to_load()
        image_pre, image_post = self._select_bands_with_bitmask(image_pre, image_post, bands_index)

        # Normalise BEFORE adding categorical bands so the stats always
        # match the selected SAR channels only.
        water_mask, _ = self._load_water_mask(index)
        image_post, image_pre, mean, std, mins, maxs = self._normalize_and_standardize(image_post, image_pre)

        image_pre, image_post, band_names, film_values = self._build_conditioning_bands(
            image_pre, image_post, common_mask_tensor, data, bands_index,
        )

        image_profile = self._build_output_profile(data["image"], len(band_names))

        sample = {
            "image":           image_post,
            "image_post":      image_post,
            "image_pre":       image_pre,
            "image_pre_name":  image_pre_name,
            "image_name_post": image_post_name,
            "image_name":      image_post_name,
            "mask":            mask,
            "has_mask":        has_mask,
            "mask_name":       mask_name,
            "bands":           band_names,
            "cell_id":         data["cell_id"],
            "profile":         image_profile,
            "mask-common":     common_mask_tensor,
            "mean":            mean,
            "std":             std,
            "min":             mins,
            "max":             maxs,
            "water_mask":      water_mask,
            "pre_post_name":   self._get_pre_post_name(data),
            "original_height": image_post.shape[1],
            "original_width":  image_post.shape[2],
        }

        if film_values:
            sample.update(film_values)

        sample.update(self._get_metadata(data))

        # Tile crop + buffer metadata — handled by TiledChangeDetectionDataset
        return self._finalize_sample(sample, data)

    # ------------------------------------------------------------------
    # __getitem__ helpers
    # ------------------------------------------------------------------

    def _load_ground_truth_mask(
        self, index: int, common_mask_tensor: Tensor, reference_image: Tensor,
    ) -> tuple[Tensor, str, bool]:
        """Load the ground-truth fire mask, or an all-zero placeholder if absent."""
        data = self.files[index]
        mask_path = data.get("mask")
        has_mask = mask_path is not None and Path(str(mask_path)).exists()
        if has_mask:
            mask, mask_name = self._load_mask(index)
            mask = self._apply_common_mask_to_tensor(common_mask_tensor, mask, IGNORE_INDEX)
        else:
            height, width = reference_image.shape[1], reference_image.shape[2]
            mask, mask_name = torch.zeros((1, height, width), dtype=torch.float32), "no_mask"
        return mask, mask_name, has_mask

    @staticmethod
    def _select_bands_with_bitmask(
        image_pre: Tensor, image_post: Tensor, bands_index: list[int] | None,
    ) -> tuple[Tensor, Tensor]:
        """Select the requested bands, always keeping BITMASK_CROPPED as channel 0.

        BITMASK_CROPPED (channel 0) is saved before band selection and
        prepended afterwards, since the model always expects it as channel 0.
        """
        if bands_index is None:
            return image_pre, image_post
        bitmask_pre = image_pre[:1, :, :]
        bitmask_post = image_post[:1, :, :]
        image_pre = torch.cat([bitmask_pre, manage_bands(image_pre, bands_index)], dim=0)
        image_post = torch.cat([bitmask_post, manage_bands(image_post, bands_index)], dim=0)
        return image_pre, image_post

    def _build_conditioning_bands(
        self,
        image_pre: Tensor,
        image_post: Tensor,
        common_mask_tensor: Tensor,
        data: dict,
        bands_index: list[int] | None,
    ) -> tuple[Tensor, Tensor, list[str], dict[str, int]]:
        """Attach FiLM conditioning metadata, or fold pass/beam in as extra bands.

        Returns the (possibly modified) pre/post tensors, the resulting band
        name list, and a dict of FiLM scalar values (empty when
        ``separate_metadata`` is disabled, since pass/beam are instead
        concatenated as image bands).
        """
        selected_band_names = (
            ['BITMASK_CROPPED'] + self.band_names
            if bands_index is not None else [i.name for i in BandName]
        )

        if self.separate_metadata:
            # Channel 0 is always BITMASK_CROPPED; remaining channels are the selected bands.
            film_values = {
                "sat_pass_value": data["sat_pass"].value,
                "beam_value": data["beam"].value,
                "pre_season": self._extract_month(data.get("group_date_pre")),
                "post_season": self._extract_month(data.get("group_date_post")),
                "time_delta_bin": self._extract_time_delta_bin(
                    data.get("group_date_pre"), data.get("group_date_post"),
                ),
                "processing_year": self._decode_year_processing(data.get("group_date_pre")),
            }
            return image_pre, image_post, selected_band_names, film_values

        image_pre = torch.cat([common_mask_tensor, image_pre], dim=0)
        image_post = torch.cat([common_mask_tensor, image_post], dim=0)
        image_pre, image_post = self.add_pass_and_beam_in_out_bands(image_pre, image_post, data)
        band_names = ['COMMON_MASK'] + selected_band_names + [SATELLITE_PASS_BAND_NAME, BEAM_BAND_NAME]
        return image_pre, image_post, band_names, {}

    @staticmethod
    def _build_output_profile(reference_path: str, band_count: int) -> dict:
        """Build the GeoTIFF profile for output predictions from a reference raster."""
        with rio.open(reference_path) as src:
            profile = src.profile
        profile['count'] = band_count
        raw_crs = profile.get('crs')
        if raw_crs is not None:
            epsg = raw_crs.to_epsg()
            profile['crs'] = f"EPSG:{epsg}" if epsg else "EPSG:3979"
        else:
            profile['crs'] = "EPSG:3979"
        profile['transform'] = list(profile['transform'])
        if profile.get('nodata') is None:
            profile['nodata'] = float(NO_DATA)
        return profile

    # ------------------------------------------------------------------
    # Metadata hooks
    # ------------------------------------------------------------------

    def _get_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"db_nbac_fire_id": data["db_nbac_fire_id"]}

    def _get_pre_post_name(self, data: dict[str, Any]) -> str:
        name = (
            f"{data['cell_id']}|"
            f"{'ASC' if data['sat_pass'] == SatellitePass.ASCENDING else 'DESC'}"
            f"-{data['beam'].name}|"
            f"({data['group_id_pre']}){data['group_date_pre']}_"
            f"({data['group_id_post']}){data['group_date_post']}|"
            f"fire_({data['db_nbac_fire_id']})_{data['fire_start_date']}_{data['fire_end_date']}"
        )
        if "_tile_row" in data:
            name += f"|tile_r{data['_tile_row']}_c{data['_tile_col']}"
        return name

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _normalize_and_standardize(
        self, image_post: Tensor, image_pre: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Z-score standardise pre/post images using the (optionally band-subset) stats."""
        bands = getattr(self, "bands", None)

        def _stat(key: str) -> Tensor:
            vals = self.norm_stats[key]
            if bands is not None:
                # Channel 0 is always BITMASK_CROPPED (index 0); then the selected bands.
                vals = [vals[0]] + [vals[b] for b in bands]
            return torch.tensor(vals, dtype=torch.float32, device=image_pre.device).view(-1, 1, 1)

        mean, std, mins, maxs = _stat("mean"), _stat("std"), _stat("min"), _stat("max")
        image_pre = (image_pre - mean) / (std + 1e-8)
        image_post = (image_post - mean) / (std + 1e-8)
        return image_post, image_pre, mean, std, mins, maxs

    def _get_bands_to_load(self) -> list[int] | None:
        return getattr(self, "bands", None)

    def add_pass_and_beam_in_out_bands(
        self, image_pre: Tensor, image_post: Tensor, data: dict,
    ) -> tuple[Tensor, Tensor]:
        """Append the satellite-pass and beam values as two constant-value bands."""
        height, width = image_pre.shape[1], image_pre.shape[2]
        sat = torch.full((1, height, width), data["sat_pass"].value, dtype=image_pre.dtype)
        beam = torch.full((1, height, width), data["beam"].value, dtype=image_pre.dtype)
        image_pre = torch.cat([image_pre, sat, beam], dim=0)
        image_post = torch.cat([image_post, sat, beam], dim=0)
        return image_pre, image_post

    # ------------------------------------------------------------------
    # Date / season helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_month(date_str: str | None) -> int:
        if date_str is None:
            return 2
        try:
            return int(str(date_str).replace("-", "")[4:6])
        except (ValueError, IndexError):
            return 0

    @staticmethod
    def _extract_time_delta_bin(pre: str | None, post: str | None) -> int:
        if pre is None or post is None:
            return len(TIME_DELTA_BINS)
        try:
            def _parse(d: str) -> datetime:
                return datetime.strptime(str(d).replace("-", ""), "%Y%m%d")

            delta = abs((_parse(post) - _parse(pre)).days)
            for i, boundary in enumerate(TIME_DELTA_BINS):
                if delta < boundary:
                    return i
            return len(TIME_DELTA_BINS)
        except (ValueError, TypeError):
            return len(TIME_DELTA_BINS)

    @staticmethod
    def _decode_year_processing(date_str: str | None) -> int:
        """Encode the acquisition year as a FiLM category index.

        Must match ``film_metadata_fields["processing_year"] = 3`` in
        ``ChangeDetectionChangeFormer.configure_model``:
            0 = undefined, 1 = year 2023, 2 = any other year.
        Returning the raw calendar year (e.g. 2023, 2025) here would index
        out of bounds in ``nn.Embedding(3, ...)`` once this value is fed to
        the FiLM conditioner.
        """
        if date_str is None:
            return 0
        try:
            year = datetime.strptime(str(date_str).replace("-", ""), "%Y%m%d").year
            return 1 if year == 2023 else 2
        except (ValueError, IndexError):
            return 0
