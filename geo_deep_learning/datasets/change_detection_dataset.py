import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from numpy import ndarray, dtype
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor

from geo_deep_learning.datasets.csv_dataset import CSVDataset
from geo_deep_learning.tools.utils import normalization, standardization

logger = logging.getLogger(__name__)


@rank_zero_only
def log_dataset(split: str, patch_count: int) -> None:
    """Log dataset."""
    logger.info("Created dataset for %s split with %s patches", split, patch_count)


class ChangeDetectionDataset(CSVDataset):
    """

    Change Detection Dataset base class.
    Loads geospatial image and mask patches from csv files.

    Dataset format:

    * images are composed of arbitrary number of bands
    ** By default, the post-image is used as input image and the pre-image as additional input (image_pre)
    * masks are single band images with pixel values representing binary change (0=no change, 1=change)
    * images and masks are stored in separate folders as

    * csv files contain the path part for each association as PRE / POST / LABEL
        e.g. 'image/pre_0.tif;image/post_0.tif;label/0_lbl.tif'

    * csv files may contain additional columns for other metadata
                e.g. 'image/pre_0.tif;image/post_0.tif;label/0_lbl.tif;aoi_id'

    * no need for the CSV files to be named after the data split e.g. 'trn.csv', 'val.csv', 'tst.csv'

    root_directory
    ├───data
    │   ├───image
    │   │       pre_0.tif
    │   │       post_0.tif
    │   │       pre_1.tif
    │   │       post_1.tif
    │   ├───label
    │           0_lbl.tif
    │           1_lbl.tif
    ├───pre_post_association.csv
    """

    def __init__(self, csv_root_folder: str,
                 patches_root_folder: str,
                 split_or_csv_file_name: str = None,
                 norm_stats: dict[str, list[float]] | None = None) -> None:

        super().__init__(csv_root_folder
                         , patches_root_folder
                         , split_or_csv_file_name
                         , norm_stats)

    def _get_csv_path(self):
        if self.split.endswith(".csv"):
            csv_path = Path(self.csv_root_folder) / self.split
        else:
            csv_path = Path(self.csv_root_folder) / f"{self.split}.csv"
        if not csv_path.exists():
            msg = f"CSV file {csv_path} not found."
            raise FileNotFoundError(msg)
        return csv_path

    def _load_files(self) -> list[dict[str, str]]:
        """Load image (pre - post) and mask paths from csv files.
        Returns a list of dictionaries with keys: 'image_pre', 'image' (image-post), 'mask'
        and values being the corresponding file paths.
        """

        csv_path = self._get_csv_path()
        df_csv = pd.read_csv(csv_path, header=None, sep=";")
        if len(df_csv.columns) == 1:
            msg = "CSV file must contain at least three columns: image_pre;image_post;mask_path"
            raise ValueError(msg)

        return [
            {
                "image_pre": Path(self.patches_root_folder) / img_pre,
                "image": Path(self.patches_root_folder) / img,
                "mask": Path(self.patches_root_folder) / lbl,
            }
            for img_pre, img, lbl in df_csv[[0, 1, 2]].itertuples(index=False)
        ]

    # ----------------------------------------------------------------------
    # Dataset protocol
    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        Return the image and mask tensors for the given index.

        Args:
            index (int): index of the sample to return

        Returns:
            Tuple[Tensor, Tensor]: image and mask tensors

        """
        image_pre, image_post, common_mask_tensor, image_pre_name, image_post_name = self._load_image(index)
        image_pre = self._apply_common_mask_to_tensor(common_mask_tensor, image_pre)
        image_post = self._apply_common_mask_to_tensor(common_mask_tensor, image_post)
        mask, mask_name = self._load_mask(index)
        mask = self._apply_common_mask_to_tensor(common_mask_tensor, mask)

        image_post, image_pre, mean, std = self._normalize_and_standardize(image_post, image_pre)

        sample = {"image": image_post,
                  "image_pre": image_pre,
                  "mask": mask,
                  "image_pre_name": image_pre_name,
                  "image_name": image_post_name,
                  "mask_name": mask_name,
                  "mean": mean,
                  "std": std}

        return sample

    def _normalize_and_standardize(self, image_post: Tensor, image_pre: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        image_pre, image_post = normalization(image_pre), normalization(image_post)
        mean = torch.tensor(self.norm_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.norm_stats["std"], dtype=torch.float32).view(-1, 1, 1)

        image_pre = standardization(image_pre, mean, std)
        image_post = standardization(image_post, mean, std)
        return image_post, image_pre, mean, std

    # ----------------------------------------------------------------------
    # Image / label loaders
    # ----------------------------------------------------------------------

    def _load_image(self, index: int) -> tuple[Tensor, Tensor, Tensor, str, str]:
        """Load pre-post images.
        return : Tensor pre, Tensor post, str pre_name, str post_name
        """
        data_at_index = self.files[index]
        image_pre = data_at_index["image_pre"]
        image_post = data_at_index["image"]
        image_pre_name = Path(image_pre).name
        image_post_name = Path(image_pre).name

        image_pre_tensor, pre_data_mask = self.convert_tif_to_tensor(image_pre)
        image_post_tensor, post_data_mask = self.convert_tif_to_tensor(image_post)
        # Common mask = areas where both images are valid (not nodata)
        # Extract BITMASK_CROPPED (always first channel)
        pre_mask = image_pre_tensor[0] if image_pre_tensor is not None else None
        post_mask = image_post_tensor[0] if image_post_tensor is not None else None

        # Common mask = areas where both masks == 1
        if pre_mask is not None and post_mask is not None:
            common_mask_tensor = (pre_mask == 1) & (post_mask == 1)
        else:
            common_mask_tensor = None


        return image_pre_tensor, image_post_tensor, common_mask_tensor, image_pre_name, image_post_name

    @staticmethod
    def _apply_common_mask_to_tensor(common_mask_tensor: Tensor, in_image_tensor: Tensor) -> Tensor:
        print(common_mask_tensor)
        in_image_tensor = in_image_tensor.masked_fill_(~common_mask_tensor, np.nan)
        return in_image_tensor

    @staticmethod
    def _read_image_and_get_no_data(path: str):
        with rio.open(path) as src:
            arr = src.read().astype(np.int32)  # shape (C,H,W)
            nodata = src.nodata
            if nodata is not None:
                mask = (arr != nodata)
            else:
                mask = ~np.isnan(arr)

        return arr, mask

    def convert_tif_to_tensor(self, in_image: str) -> tuple[Tensor, bool | ndarray[tuple[Any, ...], dtype[Any]] | Any]:

        img_array, no_data_mask = self._read_image_and_get_no_data(in_image)
        img_as_tensor = torch.from_numpy(img_array).float()
        return img_as_tensor, no_data_mask
