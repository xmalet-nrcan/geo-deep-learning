import logging
import re
from enum import Enum
from pathlib import Path
from typing import Optional, List, Any

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from numpy import ndarray, dtype
from pandas import DataFrame
from torch import Tensor

from geo_deep_learning.datasets.change_detection_dataset import ChangeDetectionDataset
from geo_deep_learning.utils.tensors import normalization, standardization, manage_bands

logger = logging.getLogger("RCM-PrePost ChangeDetectionDataset")
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s - %(name)s - [%(levelname)s] ] - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


class SatellitePass(Enum):
    ASCENDING = 0
    DESCENDING = 1

    @classmethod
    def from_str(cls, s: str) -> "SatellitePass":
        """Convert a string to a SatellitePass enum."""
        translate_dict = {
            "A": "Ascending",
            "D": "Descending",
            "ASC": "Ascending",
            "DESC": "Descending",
            "ASCENDING": "Ascending",
            "DESCENDING": "Descending",
        }
        try:
            return cls[translate_dict[s.upper()].upper()]
        except KeyError:
            raise ValueError(f"Satellite pass {s} not recognized.")


class BandName(Enum):
    BITMASK_CROPPED = 1
    LOCALINCANGLE = 2
    M = 3
    NDSV = 4
    PDN = 5
    PSN = 6
    PVN = 7
    RFDI = 8
    RL = 9
    RR = 10
    S0 = 11
    SP1 = 12
    SP2 = 13
    SP3 = 14


class Beams(Enum):
    A = 0
    B = 1
    C = 2
    D = 3


BEAM_BAND_NAME = "BEAM"
SATELLITE_PASS_BAND_NAME = "SATELLITE_PASS"

bands_stats = {'mean': [1.0088686544882763,
                        22.678325648034726,
                        4820.030168929148,
                        -578.1138439754548,
                        174.35119966169816,
                      4645.179761547494,
                        5178.970253993203,
                        4074.12440505587,
                        1427.3155618129722,
                        517.5479435073069,
                      1945.2480656061873,
                        514.8092047489475,
                        425.98675130681056,
                        8939.542957169055],
               'std': [0.17514777918322952,
                       4.602293040200134,
                       np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan],
               'min': [1.0, 0.0, 446.0, -9810.0, 0.0, 81.0, 14.0, 93.0, 2.0, 2.0, 5.0, -9340.0, -9584.0, -8947.0],
               'max': [9.0, 112.0, 9985.0, 9969.0, 6358.0, 9971.0, 9553.0, 9979.0, 32766.0, 32766.0, 32766.0, 9484.0,
                     9901.0, 9999.0]
               }

def band_names_to_indices(band_names: Optional[List[Any]]) -> Optional[List[int]]:
    """
    Convert a list of band names (str or BandName) into indices (int) according to BandName.


    """
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
                raise ValueError(f"Unknown band name: {name}")
        else:
            raise TypeError(f"Unsupported type for band_names: {type(name)}")
    return indices


NO_DATA = 32767
IGNORE_INDEX = 255


class RCMChangeDetectionDataset(ChangeDetectionDataset):
    def __init__(self, csv_root_folder: str,
                 patches_root_folder: str,
                 split_or_csv_file_name: str = None,
                 norm_stats: dict[str, list[float]] | None = None,
                 bands: Optional[List[int]] = None,
                 band_names: Optional[List[str]] = None,
                 satellite_pass: Optional[str | SatellitePass] = None,
                 beams: Optional[List[str]] = None
                 ) -> None:
        # Set bands index and band names
        if band_names is not None:
            self.bands = band_names_to_indices(band_names)
            self.band_names = [
                bn.name if isinstance(bn, BandName) else str(bn) for bn in band_names
            ]
        elif bands is not None:
            self.bands = bands
            # Try to retrieve band names from indices
            self.band_names = []
            for idx in bands:
                try:
                    self.band_names.append(BandName(idx).name)
                except Exception:
                    self.band_names.append(str(idx))
        else:
            self.bands = [i.value for i in BandName]
            self.band_names = [i.name for i in BandName]

        if satellite_pass:
            match satellite_pass:
                case str():
                    self.satellite_pass = SatellitePass.from_str(satellite_pass)
                case SatellitePass():
                    self.satellite_pass = satellite_pass
                case _:
                    raise TypeError("satellite_pass must be string or SatellitePass enum")
        else:
            self.satellite_pass = None
        self.beams = [] if beams is None else [i.upper() for i in beams]
        if norm_stats is None:
            norm_stats = bands_stats
        super().__init__(csv_root_folder=csv_root_folder, patches_root_folder=patches_root_folder,
                         split_or_csv_file_name=split_or_csv_file_name, norm_stats=norm_stats)

    def _load_files(self) -> list[dict[str, str]]:
        df_csv = self._get_input_dataset_as_dataframe()

        files = []
        for (img_pre, img, group_id_pre, group_id_post, db_nbac_fire_id,
             cell_id, group_date_pre, group_date_post, beam, sat_pass,
             fire_start_date, fire_end_date) in df_csv[
            ['pre_path', 'post_path', 'group_id_pre', 'group_id_post',
             'db_nbac_fire_id', 'cell_id', 'group_date_pre', 'group_date_post',
             'beam', 'sat_pass', 'fire_start_date', 'fire_end_date']
        ].itertuples(index=False):
            img_pre_path = img_pre.replace("$ROOT_PATH", self.patches_root_folder).strip()
            img_post_path = img.replace("$ROOT_PATH", self.patches_root_folder).strip()
            if Path(img_pre_path).exists() and Path(img_post_path).exists():
                files.append({
                    "image_pre": img_pre.replace("$ROOT_PATH", self.patches_root_folder).strip(),
                    "image": img.replace("$ROOT_PATH", self.patches_root_folder).strip(),
                    "mask": self._get_mask_path(cell_id, group_date_post),
                    "water_mask": self._get_water_mask_path(cell_id),
                    "cell_id": cell_id,
                    "db_nbac_fire_id": db_nbac_fire_id,
                    "group_date_pre": group_date_pre,
                    "group_date_post": group_date_post,
                    "beam": beam,
                    "sat_pass": sat_pass,
                    "group_id_pre": group_id_pre,
                    "group_id_post": group_id_post,
                    "fire_start_date": fire_start_date,
                    "fire_end_date": fire_end_date,
                })

        logger.info(
            "Loaded %d entries (%d with mask, %d without mask)",
            len(files),
            sum(1 for f in files if f["mask"] is not None),
            sum(1 for f in files if f["mask"] is None),
        )

        return files

    def _get_water_mask_path(self, cell_id) -> Path | Any:
        # --- Masque d'eau (optionnel) ---
        water_mask_path = (
                Path(self.patches_root_folder) / cell_id / "static_data"
                / f"{cell_id}_WATER_mask_100m.tif"
        )
        if not water_mask_path.exists():
            logger.debug("Water mask not found, setting to None: %s", water_mask_path)
            water_mask_path = None
        return water_mask_path

    def _get_mask_path(self, cell_id, group_date_post) -> Path | Any:
        mask_path = (
                Path(self.patches_root_folder) / cell_id / "static_data"
                / f"{cell_id}_nbac_{int(group_date_post[:4])}_mask_unburn_burn_reject_100m.tif"
        )
        if not mask_path.exists():
            logger.debug("Mask not found, setting to None: %s", mask_path)
            mask_path = None
        return mask_path

    def _get_input_dataset_as_dataframe(self) -> DataFrame:
        csv_path = self._get_csv_path()
        df_csv = pd.read_csv(csv_path)

        logger.info("BEAM FILTER: {}".format(self.beams))
        logger.info("SATELLITE PASS FILTER: {}".format(self.satellite_pass))
        df_csv['sat_pass'] = df_csv['sat_pass'].apply(lambda x: SatellitePass.from_str(x))
        df_csv['beam'] = df_csv['beam'].apply(lambda x: Beams[str(x).upper()])

        if self.satellite_pass is not None:
            df_csv = df_csv[df_csv['sat_pass'] == self.satellite_pass]
            if df_csv.empty:
                logger.warning(f"No entries found for satellite pass {self.satellite_pass}")

        if len(self.beams) > 0:
            beams_str = [Beams[str(b).upper()] for b in self.beams]
            df_csv = df_csv[df_csv['beam'].isin(beams_str)]
            if df_csv.empty:
                logger.warning(f"No entries found for beams {beams_str}")
        return df_csv

    def __len__(self) -> int:
        return super().__len__()

    def _get_bands_to_load(self) -> list | None:
        """
        Get the list of bands to load, ensuring BITMASK_CROPPED (band 1) is always included.
        The returned list is 0-based indices for rasterio based on BandName enum numbering.
        """
        if self.bands is None:
            return None

        bands = list(self.bands)  # copie
        if 1 not in bands:
            bands.append(1)  # toujours inclure BITMASK_CROPPED

        # Dédoublonner et trier
        bands = sorted(set(bands))

        # Convertir en 0-based pour rasterio
        return [i - 1 for i in bands]

    @staticmethod
    def add_pass_and_beam_in_out_bands(pre_img, post_img, current_sample):
        H, W = pre_img.shape[1], pre_img.shape[2]

        # encode sat_pass : Ascending=0, Descending=1
        sat_pass_val = current_sample['sat_pass'].value
        sat_pass_band = torch.full((1, H, W), sat_pass_val, dtype=pre_img.dtype)

        beam_map = current_sample["beam"].value
        beam_val = beam_map
        beam_band = torch.full((1, H, W), beam_val, dtype=pre_img.dtype)

        # concat à pre_img et post_img
        pre_img = torch.cat([pre_img, sat_pass_band, beam_band], dim=0)
        post_img = torch.cat([post_img, sat_pass_band, beam_band], dim=0)
        return pre_img, post_img

    @staticmethod
    def _read_image_and_get_no_data(path: str, in_dtype: np.dtype = np.int16):
        with rio.open(path, nodata=NO_DATA, dtype='int16') as src:
            arr = src.read().astype(in_dtype)  # shape (C,H,W)
            mask = arr[0, :, :] == 1  # Read the bitmask cropped band to get data mask

        return arr, mask

    def convert_tif_to_tensor(self, in_image: str, in_dtype=np.int16) -> tuple[
        Tensor, bool | ndarray[tuple[Any, ...], dtype[Any]] | Any]:
        return super().convert_tif_to_tensor(in_image, in_dtype)

    def __getitem__(self, index: int) -> dict:
        data = self.files[index]
        image_pre, image_post, common_mask_tensor, image_pre_name, image_post_name = self._load_image(index)

        # --- Water mask (optionnel) ---
        water_mask_path = data.get("water_mask")
        if water_mask_path is not None:
            water_mask, _ = self._load_water_mask(index)
            no_water_mask = (water_mask == 0)  # True = pas d'eau = garder
        else:
            H, W = image_pre.shape[1], image_pre.shape[2]
            water_mask = torch.ones((1, H, W), dtype=torch.float32)
            no_water_mask = torch.ones((1, H, W), dtype=torch.bool)

        image_post, image_pre, mean, std, mins, maxs = self._normalize_and_standardize(image_post, image_pre)
        common_mask_tensor = common_mask_tensor & no_water_mask

        # --- Mask from NBAC (optionnel) ---
        mask_path = data.get("mask")
        has_mask = mask_path is not None and Path(str(mask_path)).exists()
        if has_mask:
            mask, mask_name = self._load_mask(index)
            mask = self._apply_common_mask_to_tensor(common_mask_tensor, mask, IGNORE_INDEX)
        else:
            H, W = image_pre.shape[1], image_pre.shape[2]
            mask = torch.zeros((1, H, W), dtype=torch.float32)
            mask_name = "no_mask"

        # Apply common mask to images
        image_pre = self._apply_common_mask_to_tensor(common_mask_tensor, image_pre, IGNORE_INDEX)
        image_post = self._apply_common_mask_to_tensor(common_mask_tensor, image_post, IGNORE_INDEX)

        # Band selection
        bands_index = self._get_bands_to_load()
        if bands_index is not None:
            image_pre = manage_bands(image_pre, bands_index)
            image_post = manage_bands(image_post, bands_index)

        # Add common mask as first band + sat_pass/beam bands
        image_pre = torch.cat([common_mask_tensor, image_pre], dim=0)
        image_post = torch.cat([common_mask_tensor, image_post], dim=0)
        image_pre, image_post = self.add_pass_and_beam_in_out_bands(image_pre, image_post, data)

        band_names = [BandName(i + 1).name for i in bands_index] if bands_index is not None else [i.name for i in
                                                                                                  BandName]
        band_names = ['COMMON_MASK'] + band_names + [SATELLITE_PASS_BAND_NAME, BEAM_BAND_NAME]

        image_profile = None
        with rio.open(data['image']) as src:
            image_profile = src.profile
        image_profile['count'] = len(band_names)
        image_profile['crs'] = str(image_profile['crs'])
        image_profile['transform'] = list(image_profile['transform'])
        pre_post_name = self._get_pre_post_name(data)

        sample = {
            "image": image_post,
            "image_post": image_post,
            "image_pre": image_pre,
            "image_pre_name": image_pre_name,
            "image_name_post": image_post_name,
            "image_name": image_post_name,
            "mask": mask,
            "has_mask": has_mask,
            "mask_name": mask_name,
            "bands": band_names,
            "cell_id": data["cell_id"],
            "profile": image_profile,
            "mask-common": common_mask_tensor,
            "mean": mean,
            "std": std,
            "min": mins,
            "max": maxs,
            "water_mask": water_mask,
            "pre_post_name": pre_post_name,
            "original_height": image_post.shape[1],
            "original_width": image_post.shape[2],
        }

        sample.update(self._get_metadata(data))

        return sample

    def _get_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return dataset-specific metadata to include in the sample dict.
        Override in subclasses for different CSV schemas."""
        return {
            "db_nbac_fire_id": data["db_nbac_fire_id"],
        }

    def _get_pre_post_name(self, data: dict[str, str]) -> str:
        pre_post_name = (
            f"{data['cell_id']}|"
            f"{'ASC' if data['sat_pass'] == SatellitePass.ASCENDING else 'DESC'}-{data['beam'].name}|"
            f"({data['group_id_pre']}){data['group_date_pre']}_"
            f"({data['group_id_post']}){data['group_date_post']}|"
            f"fire_({data['db_nbac_fire_id']})_{data['fire_start_date']}_{data['fire_end_date']}"
        )
        return pre_post_name

    def _normalize_and_standardize(self, image_post: Tensor, image_pre: Tensor) -> tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]:
        # Per-patch min-max normalization per band, without torch.nanmin / nanmax
        eps = 1e-6

        self._norm_image(image_pre, eps )
        self._norm_image(image_post, eps )

        image_pre = torch.clamp(torch.nan_to_num(image_pre, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        image_post = torch.clamp(torch.nan_to_num(image_post, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

        dummy = torch.zeros((image_pre.shape[0], 1, 1), dtype=torch.float32, device=image_pre.device)
        return image_post, image_pre, dummy, dummy, dummy, dummy

    @staticmethod
    def _norm_image( input_image: Tensor, eps: float,):
        for i in range(input_image.shape[0]):
            curr_band = input_image[i]

            # Mask non-finite values
            infinte_mask = torch.isfinite(curr_band)
            # Handle case where all values are non-finite
            if infinte_mask.any():
                band_min_post = curr_band[infinte_mask].min()
                band_max_post = curr_band[infinte_mask].max()
            else:
                band_min_post = torch.tensor(0.0, device=input_image.device, dtype=input_image.dtype)
                band_max_post = torch.tensor(1.0, device=input_image.device, dtype=input_image.dtype)

            clamped_images = torch.clamp(band_max_post - band_min_post, min=eps)

            input_image[i] = (curr_band - band_min_post) / clamped_images

    def _load_water_mask(self, index: int) -> tuple[Tensor, str]:
        """Load water mask."""
        return self._load_image_by_name(index, "water_mask")


if __name__ == '__main__':
    dataset = RCMChangeDetectionDataset(
        csv_root_folder=r"C:\Users\xmalet\PycharmProjects\geo-deep-learning\data",
        patches_root_folder=r"C:\Users\xmalet\PycharmProjects\geo-deep-learning\data\raw",
        split_or_csv_file_name=r"pre_post_datasets.csv",
        band_names=["RR", "RL", "M", 'PSN'],
        satellite_pass="Descending",
        beams=['A']
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Image pre shape: {sample['image_pre'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"BANDS: {sample['bands']}")
    print(f"Cell ID: {sample['cell_id']}")
    print(f"DB NBAC Fire ID: {sample['db_nbac_fire_id']}")
    print(sample['bands'].index(SATELLITE_PASS_BAND_NAME), sample['bands'].index(BEAM_BAND_NAME))
    print(sample['image'][sample['bands'].index(SATELLITE_PASS_BAND_NAME), :5, :5])
    print(sample['image'][sample['bands'].index(BEAM_BAND_NAME), :5, :5])

    print(sample['profile'])
    print(sample['image_name'])
    data : Tensor = sample['image']

    with rio.open(r"C:\Users\xmalet\PycharmProjects\geo-deep-learning\data\image_post.tiff", 'w',
                  **sample['profile']) as src:
        src.write(data.numpy())

    # print(f"Mean: {sample['mean']}")
    # print(f"Std: {sample['std']}")
