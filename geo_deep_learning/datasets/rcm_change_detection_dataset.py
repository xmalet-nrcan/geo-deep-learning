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
from pathlib import Path
from typing import Optional, List, Any

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from enum import Enum
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor

from geo_deep_learning.datasets.tiled_change_detection_dataset import TiledChangeDetectionDataset
from geo_deep_learning.utils.tensors import manage_bands

logger = logging.getLogger("RCM-PrePost ChangeDetectionDataset")
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s - %(name)s - [%(levelname)s] ] - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


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
            raise ValueError(f"Satellite pass {s!r} not recognized.")


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
    logger.info(f"TREATING BANDS : {band_names}")
    indices = []
    for name in band_names:
        if isinstance(name, BandName):
            indices.append(name.value)
        elif isinstance(name, str):
            try:
                indices.append(BandName[name].value)
            except KeyError:
                raise ValueError(f"Unknown band name: {name!r}")
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
                except Exception:
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
            img_pre_path  = img_pre.replace("$ROOT_PATH", self.patches_root_folder).strip()
            img_post_path = img.replace("$ROOT_PATH", self.patches_root_folder).strip()
            if Path(img_pre_path).exists() and Path(img_post_path).exists():
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

    @staticmethod
    def _read_image_and_get_no_data(
        path: str,
        in_dtype: np.dtype = np.int16,
    ) -> tuple[ndarray, ndarray]:
        """Read a GeoTIFF; return ``(array [C,H,W], bitmask [H,W])``."""
        with rio.open(path) as src:
            arr = src.read().astype(in_dtype)
        return arr, (arr[0, :, :] == 1)

    def _load_mask(self, index: int) -> tuple[Tensor, str]:
        mask, name = self._load_image_by_name(index, "mask")
        mask = self._apply_buffer_padding(mask, self.files[index])
        return mask, name

    def _load_water_mask(self, index: int) -> tuple[Tensor, str]:
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
        with rio.open(data["image"]) as src:
            H, W = src.height, src.width
        return torch.zeros((1, H, W), dtype=torch.float32), f"no_{key}"

    @staticmethod
    def _apply_common_mask_to_tensor(
        common_mask_tensor: Tensor,
        in_image_tensor: Tensor,
        fill_value=np.nan,
    ) -> Tensor:
        return in_image_tensor.masked_fill_(~common_mask_tensor, fill_value)

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

    def __getitem__(self, index: int) -> dict:  # noqa: C901
        data = self.files[index]
        image_pre, image_post, common_mask_tensor, image_pre_name, image_post_name = (
            self._load_image(index)
        )

        # Ground-truth fire mask
        mask_path = data.get("mask")
        has_mask = mask_path is not None and Path(str(mask_path)).exists()
        if has_mask:
            mask, mask_name = self._load_mask(index)
            mask = self._apply_common_mask_to_tensor(common_mask_tensor, mask, IGNORE_INDEX)
        else:
            H, W = image_pre.shape[1], image_pre.shape[2]
            mask, mask_name = torch.zeros((1, H, W), dtype=torch.float32), "no_mask"

        image_pre  = self._apply_common_mask_to_tensor(common_mask_tensor, image_pre,  IGNORE_INDEX)
        image_post = self._apply_common_mask_to_tensor(common_mask_tensor, image_post, IGNORE_INDEX)

        bands_index = self._get_bands_to_load()
        if bands_index is not None:
            image_pre  = manage_bands(image_pre,  bands_index)
            image_post = manage_bands(image_post, bands_index)

        # Normalise BEFORE adding categorical bands so the stats always
        # match the selected SAR channels only.
        water_mask, _ = self._load_water_mask(index)
        image_pre, image_post, mean, std, mins, maxs = self._normalize_and_standardize(
            image_post, image_pre
        )

        # Initialise FiLM metadata to safe defaults; only populated in separate_metadata mode
        sat_pass_value: int | None = None
        beam_value:     int | None = None
        pre_month_value:   int = 0
        post_month_value:  int = 0
        time_delta_bin:    int = len(TIME_DELTA_BINS)
        decoded_year:      int = 0

        if self.separate_metadata:
            band_names = (
                [BandName(i + 1).name for i in bands_index]
                if bands_index is not None else [i.name for i in BandName]
            )
            sat_pass_value   = data["sat_pass"].value
            beam_value       = data["beam"].value
            pre_month_value  = self._extract_month(data.get("group_date_pre"))
            post_month_value = self._extract_month(data.get("group_date_post"))
            time_delta_bin   = self._extract_time_delta_bin(
                data.get("group_date_pre"), data.get("group_date_post"),
            )
            decoded_year = self._decode_year_processing(data.get("group_date_pre"))
        else:
            image_pre  = torch.cat([common_mask_tensor, image_pre],  dim=0)
            image_post = torch.cat([common_mask_tensor, image_post], dim=0)
            image_pre, image_post = self.add_pass_and_beam_in_out_bands(image_pre, image_post, data)
            band_names = (
                ['COMMON_MASK']
                + ([BandName(i + 1).name for i in bands_index] if bands_index else [i.name for i in BandName])
                + [SATELLITE_PASS_BAND_NAME, BEAM_BAND_NAME]
            )
            sat_pass_value = beam_value = None

        # GeoTIFF profile for output TIFs
        with rio.open(data['image']) as src:
            image_profile = src.profile
        image_profile['count'] = len(band_names)
        raw_crs = image_profile.get('crs')
        if raw_crs is not None:
            epsg = raw_crs.to_epsg()
            image_profile['crs'] = f"EPSG:{epsg}" if epsg else "EPSG:3979"
        else:
            image_profile['crs'] = "EPSG:3979"
        image_profile['transform'] = list(image_profile['transform'])
        if image_profile.get('nodata') is None:
            image_profile['nodata'] = float(NO_DATA)

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

        if sat_pass_value is not None:
            sample.update({
                "sat_pass_value":  sat_pass_value,
                "beam_value":      beam_value,
                "pre_season":      pre_month_value,
                "post_season":     post_month_value,
                "time_delta_bin":  time_delta_bin,
                "processing_year": decoded_year,
            })

        sample.update(self._get_metadata(data))

        # Tile crop + buffer metadata — handled by TiledChangeDetectionDataset
        return self._finalize_sample(sample, data)

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
        bands = getattr(self, "bands", None)

        def _t(key):
            vals = self.norm_stats[key]
            if bands is not None:
                vals = [vals[b] for b in bands]
            return torch.tensor(
                vals, dtype=torch.float32, device=image_pre.device,
            ).view(-1, 1, 1)

        mean, std, mins, maxs = _t("mean"), _t("std"), _t("min"), _t("max")
        image_pre  = (image_pre  - mean) / (std + 1e-8)
        image_post = (image_post - mean) / (std + 1e-8)
        return image_pre, image_post, mean, std, mins, maxs

    def _get_bands_to_load(self) -> list[int] | None:
        return getattr(self, "bands", None)

    def add_pass_and_beam_in_out_bands(
        self, image_pre: Tensor, image_post: Tensor, data: dict,
    ) -> tuple[Tensor, Tensor]:
        H, W = image_pre.shape[1], image_pre.shape[2]
        sat = torch.full((1, H, W), data["sat_pass"].value, dtype=image_pre.dtype)
        bm  = torch.full((1, H, W), data["beam"].value,     dtype=image_pre.dtype)
        image_pre  = torch.cat([image_pre,  sat, bm], dim=0)
        image_post = torch.cat([image_post, sat, bm], dim=0)
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
            from datetime import datetime
            def _p(d):
                return datetime.strptime(str(d).replace("-", ""), "%Y%m%d")
            delta = abs((_p(post) - _p(pre)).days)
            for i, boundary in enumerate(TIME_DELTA_BINS):
                if delta < boundary:
                    return i
            return len(TIME_DELTA_BINS)
        except Exception:
            return len(TIME_DELTA_BINS)

    @staticmethod
    def _decode_year_processing(date_str: str | None) -> int:
        if date_str is None:
            return 0
        try:
            from datetime import datetime
            return datetime.strptime(str(date_str).replace("-", ""), "%Y%m%d").year
        except (ValueError, IndexError):
            return 0
