import logging
from enum import Enum
from pathlib import Path
from typing import Optional, List, Any

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from numpy import ndarray, dtype
from torch import Tensor

from geo_deep_learning.datasets.change_detection_dataset import ChangeDetectionDataset
from geo_deep_learning.utils.tensors import normalization, standardization

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
SATTELITE_PASS_BAND_NAME = "SATTELITE_PASS"


def band_names_to_indices(band_names: Optional[List[Any]]) -> Optional[List[int]]:
    """
    Convert a list of band names (str or BandName) into indices (int) according to BandName.


    """
    if band_names is None:
        return None

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

        super().__init__(csv_root_folder=csv_root_folder, patches_root_folder=patches_root_folder,
                         split_or_csv_file_name=split_or_csv_file_name, norm_stats=norm_stats)

    def _load_files(self) -> list[dict[str, str]]:
        csv_path = self._get_csv_path()
        df_csv = pd.read_csv(csv_path)

        logger.info("BEAM FILTER: {}".format(self.beams))
        logger.info("SATELLITE PASS FILTER: {}".format(self.satellite_pass))
        df_csv['sat_pass'] = df_csv['sat_pass'].apply(lambda x: SatellitePass.from_str(x))
        df_csv['beam'] = df_csv['beam'].apply(lambda x: Beams[str(x).upper()])

        if self.satellite_pass is not None:
            df_csv = df_csv[df_csv['sat_pass'] == self.satellite_pass]
            if df_csv.empty:
                raise ValueError(f"No entries found for satellite pass {self.satellite_pass}")

        if len(self.beams) > 0:
            beams_str = [Beams[str(b).upper()] for b in self.beams]
            df_csv = df_csv[df_csv['beam'].isin(beams_str)]
            if df_csv.empty:
                raise ValueError(f"No entries found for beams {beams_str}")

        return [
            {
                "image_pre": (img_pre.replace("$ROOT_PATH", self.patches_root_folder).strip()),
                "image": (img.replace("$ROOT_PATH", self.patches_root_folder).strip()),
                "mask": Path(
                    self.patches_root_folder) / cell_id / "static_data" / f"{cell_id}_nbac_{int(group_date_post[:4]) + 1}_prefire_100m.tif",
                "water_mask": Path(
                    self.patches_root_folder) / cell_id / "static_data" / f"{cell_id}_WATER_mask_100m.tif",
                "cell_id": cell_id,
                "db_nbac_fire_id": db_nbac_fire_id,
                "group_date_pre": group_date_pre,
                "group_date_post": group_date_post,
                "beam": beam,
                "sat_pass": sat_pass
            }
            for img_pre,
            img,
            db_nbac_fire_id,
            cell_id,
            group_date_pre,
            group_date_post,
            beam,
            sat_pass in
            df_csv[
                ['pre_path',
                 'post_path',
                 'db_nbac_fire_id',
                 'cell_id',
                 'group_date_pre',
                 'group_date_post',
                 'beam',
                 'sat_pass']].itertuples(index=False)
        ]

    def __len__(self) -> int:
        return super().__len__()

    def _get_bands_to_load(self) -> list | None:
        # Always include BITMASK_CROPPED (band 1)
        bands_to_read = self.bands if self.bands is not None else None
        if bands_to_read is None:
            return None
        elif bands_to_read is not None and 1 not in bands_to_read:
            bands_to_read_with_mask = [1] + bands_to_read
        else:
            bands_to_read_with_mask = bands_to_read
        bands_to_read_with_mask.sort()
        bands_to_read_with_mask = set(bands_to_read_with_mask)  # to remove duplicates
        # Convert to 0-based indices for rasterio
        return [i - 1 for i in bands_to_read_with_mask]

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
            mask = arr[0, :, :] == 1    # Read the bitmask cropped band to get data mask

        return arr, mask

    def convert_tif_to_tensor(self, in_image: str, in_dtype=np.int16) -> tuple[
        Tensor, bool | ndarray[tuple[Any, ...], dtype[Any]] | Any]:
        return super().convert_tif_to_tensor(in_image, in_dtype)

    def __getitem__(self, index: int) -> dict:
        """
                Return the image and mask tensors for the given index.

                Args:
                    index (int): index of the sample to return

                Returns:
                    Tuple[Tensor, Tensor]: image and mask tensors

                """
        data = self.files[index]
        image_pre, image_post, common_mask_tensor, image_pre_name, image_post_name = self._load_image(index)
        water_mask, water_mask_name = self._load_water_mask(index)
        no_water_mask = (water_mask == 0)  # True where water

        common_mask_tensor = common_mask_tensor & no_water_mask
        print("COMMUN MASK AFTER WATER", common_mask_tensor.shape, common_mask_tensor.sum(), common_mask_tensor.all())
        mask, mask_name = self._load_mask(index)

        # Apply common mask to all and set NO_DATA where mask is False
        image_pre = self._apply_common_mask_to_tensor(common_mask_tensor, image_pre, NO_DATA)
        image_post = self._apply_common_mask_to_tensor(common_mask_tensor, image_post, NO_DATA)
        mask = self._apply_common_mask_to_tensor(common_mask_tensor, mask, NO_DATA)



        bands_index = self._get_bands_to_load()
        if bands_index is not None:
            image_pre = image_pre[bands_index, :, :]
            image_post = image_post[bands_index, :, :]

        # Add common mask as first band
        image_pre = torch.cat([common_mask_tensor, image_pre], dim=0)
        image_post = torch.cat([common_mask_tensor, image_post], dim=0)

        image_pre, image_post = self.add_pass_and_beam_in_out_bands(image_pre, image_post, data)

        # image_post, image_pre, mean, std = self._normalize_and_standardize(image_post, image_pre)


        band_names = [BandName(i + 1).name for i in bands_index] if bands_index is not None else [i.name for i in
                                                                                                  BandName]
        band_names = ['COMMON_MASK'] + band_names + [SATTELITE_PASS_BAND_NAME, BEAM_BAND_NAME]

        image_profile = None
        with rio.open(data['image']) as src:
            image_profile = src.profile
        image_profile['count'] = len(band_names)

        sample = {"image": image_post,
                  "image_pre": image_pre,
                  "mask": mask,
                  "image_pre_name": image_pre_name,
                  "image_name": image_post_name,
                  "mask_name": mask_name,
                  "bands": band_names,
                  "cell_id": data["cell_id"],
                  "db_nbac_fire_id": data["db_nbac_fire_id"],
                  "profile": image_profile,
                  "common_data_mask": common_mask_tensor,
                  # "mean": mean,
                  # "std": std
                  }
        return sample

    def _normalize_and_standardize(self, image_post: Tensor, image_pre: Tensor) -> tuple[
        Tensor, Tensor, Tensor, Tensor]:
        image_pre, image_post = normalization(image_pre), normalization(image_post)
        mean = torch.tensor(self.norm_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.norm_stats["std"], dtype=torch.float32).view(-1, 1, 1)
        #
        image_pre = standardization(image_pre, mean, std)
        image_post = standardization(image_post, mean, std)
        return image_post, image_pre, mean, std

    def _load_water_mask(self, index: int) -> tuple[Tensor, str]:
        """Load water mask."""
        return self._load_image_by_name(index, "water_mask")


if __name__ == '__main__':
    dataset = RCMChangeDetectionDataset(
        csv_root_folder=r"C:\Users\xmalet\PycharmProjects\geo-deep-learning\data",
        patches_root_folder=r"C:\Users\xmalet\PycharmProjects\geo-deep-learning\data\raw",
        split_or_csv_file_name=r"pre_post_datasets.csv",
        norm_stats={"mean": [0.0] * 14, "std": [1.0] * 14},
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
    print(sample['bands'].index(SATTELITE_PASS_BAND_NAME), sample['bands'].index(BEAM_BAND_NAME))
    print(sample['image'][sample['bands'].index(SATTELITE_PASS_BAND_NAME), :5, :5])
    print(sample['image'][sample['bands'].index(BEAM_BAND_NAME), :5, :5])

    print(sample['image_pre'].min(), sample['image_pre'].max())
    print(sample['image'].min(), sample['image'].max())

    print(sample['image_name'])
    with rio.open(r"C:\Users\xmalet\PycharmProjects\geo-deep-learning\data\image_post.tiff", 'w',
                  **sample['profile']) as src:
        src.write(sample['image'])

    # print(f"Mean: {sample['mean']}")
    # print(f"Std: {sample['std']}")
