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


NO_DATA = 32767


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
                 split: str = None,
                 csv_file_name: str = None,
                 norm_stats: dict[str, list[float]] | None = None) -> None:
        if split is None and csv_file_name is None:
            raise ValueError("Either split or csv_file_name must be specified")

        super().__init__(csv_root_folder, patches_root_folder, split if split is not None else csv_file_name,
                         norm_stats)

    def _load_files(self) -> list[dict[str, str]]:
        """Load image (pre - post) and mask paths from csv files."""

        if self.split.endswith(".csv"):
            csv_path = Path(self.csv_root_folder) / self.split
        else:
            csv_path = Path(self.csv_root_folder) / f"{self.split}.csv"
        if not csv_path.exists():
            msg = f"CSV file {csv_path} not found."
            raise FileNotFoundError(msg)
        df_csv = pd.read_csv(csv_path, header=None, sep=";")
        if len(df_csv.columns) == 1:
            msg = "CSV file must contain at least two columns: image_path;mask_path"
            raise ValueError(msg)

        return [
            {
                "image_pre": Path(self.patches_root_folder) / img_pre,
                "image": Path(self.patches_root_folder) / img,
                "mask": Path(self.patches_root_folder) / lbl,
            }
            for img_pre, img, lbl in df_csv[[0, 1, 2]].itertuples(index=False)
        ]

    def _load_mask(self, index: int) -> Tensor:
        return super()._load_mask(index)

    def __post_init__(
            self,
            samples: Optional[List[Dict[str, Any]]] = None,
            transform=None,
            bands: Optional[List[int]] = None,
            band_names: Optional[List[str]] = None,
            patch_size: tuple[int, int] | None = None,
    ):
        """
        Dataset for pre/post remote sensing change detection.

        Args:
            samples: List of dicts with keys:
                    'pre_path': Image pre-path,
                    'post_path': Image post-path,
                    'cell_id': cell_id of the sample,
                    'label_path': burned area label path for the pre-post association,
                    'water_mask_path': water mask path for the current cell.
                    'db_nbac_fire_id': fire id in the database,
                    'sat_pass': RCM satellite pass (Ascending or Descending),
                    'beam': RCM beam (A,B,C,D),
                    'group_date_pre': date of the pre image,
                    'group_date_post': date of the post image,
                    'group_id_pre': database group id of the pre image,
                    'group_id_post': database group id of the post image,
                    'bounds': bounds of the pre or post image,
                    'tif_transform': GeoTiff transform of the pre or post image,
                    'crs': CRS of the pre or post image.,
            transform: Optional transform to apply to images. Transformer must be callable.
            bands: List of band indices to read (1-based, e.g. [1,2,3]). If None, all bands are read.
            band_names: List of band names to read (e.g. ["RR", "S0"]). If provided, overrides bands.
            patch_size: Optional tuple of patch size (height, width).
                If provided, the dataset will return patches of the specified size.
        """
        super().__init__()
        self.samples = samples or []
        self.transform = transform
        self.patch_size = patch_size if patch_size is not None else (256, 256)

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
            self.bands = None
            self.band_names = None

        if self.patch_size and self.transform is None:
            self.transform = A.Compose(
                [
                    # Padding and cropping
                    A.PadIfNeeded(
                        min_height=self.patch_size[1],
                        min_width=self.patch_size[0],
                        position="random",
                        fill_mask=NO_DATA,
                        fill=NO_DATA,
                        p=1,
                    ),
                    A.RandomCrop(
                        width=self.patch_size[1],
                        height=self.patch_size[0],
                        pad_if_needed=True,
                        fill_mask=NO_DATA,
                        fill=NO_DATA,
                        p=0.5,
                    ),
                    # Rotations
                    A.Transpose(p=0.5),
                    A.OneOf(
                        [
                            A.RandomRotate90(p=0.8),
                            A.Rotate(
                                interpolation=cv2.INTER_NEAREST,
                                mask_interpolation=cv2.INTER_NEAREST,
                                fill_mask=NO_DATA,
                                fill=NO_DATA,
                                p=1,
                            ),
                        ],
                        p=0.7,
                    ),
                    A.SquareSymmetry(p=0.5),
                    # Local Geometric Distortions (no scale change)
                    A.OneOf(
                        [
                            A.ElasticTransform(
                                alpha=40,
                                sigma=5,
                                border_mode=0,
                                interpolation=cv2.INTER_NEAREST,
                                mask_interpolation=cv2.INTER_NEAREST,
                                fill=NO_DATA,
                                fill_mask=NO_DATA,
                                p=1,
                            ),
                            A.GridDistortion(
                                num_steps=5,
                                distort_limit=0.2,
                                border_mode=0,
                                interpolation=cv2.INTER_NEAREST,
                                mask_interpolation=cv2.INTER_NEAREST,
                                fill=NO_DATA,
                                fill_mask=NO_DATA,
                                p=1,
                            ),
                            A.Perspective(
                                scale=(0.02, 0.05),
                                keep_size=True,
                                interpolation=cv2.INTER_NEAREST,
                                mask_interpolation=cv2.INTER_NEAREST,
                                fill=NO_DATA,
                                fill_mask=NO_DATA,
                                p=1,
                            ),
                        ],
                        p=0.3,
                    ),
                    # Dropout
                    A.OneOf(
                        [
                            A.CoarseDropout(
                                num_holes_range=(1, 5),
                                hole_height_range=(0.1, 0.5),
                                hole_width_range=(0.1, 0.5),
                                fill=NO_DATA,
                                fill_mask=NO_DATA,
                                p=1,
                            ),
                            A.GridDropout(
                                ratio=0.2,
                                random_offset=True,
                                fill=NO_DATA,
                                fill_mask=NO_DATA,
                                p=1,
                            ),
                        ],
                        p=0.6,
                    ),
                    A.ToTensorV2(transpose_mask=True),
                ],
                additional_targets={
                    "image_pre": "image",
                },
            )

    # ----------------------------------------------------------------------
    # Class methods for building samples
    # ----------------------------------------------------------------------
    @classmethod
    def build_dataset_from_csv(
            cls,
            csv_path: str | pathlib.Path,
            data_directory: str = None,
            transform=None,
            bands: Optional[List[int]] = None,
            band_names: Optional[List[str | BandName]] = None,
            sat_pass_filter: SatellitePass | str = None,
            beams_filter: Optional[str] = None,
    ) -> "PrePostChangeDataset":
        """
        Builds samples from a CSV file and returns a class instance containing these samples. The
        CSV file must be extracted from the database from the table `deeplearning_dataset.mat_vw_input_pre_post_data`

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing sample metadata.
        data_directory : Optional[str]
            Root directory path to replace placeholders in file paths in the CSV. Defaults to 'data/raw'.
        transform : Any
            A transformation or preprocessing function to be applied to the samples. This can be `None`.
        label_dir : Optional[str]
            Directory containing label data. If unspecified, labels are handled via default paths or logic.
        mask_dir : Optional[str]
            Directory containing mask data. If unspecified, masks are handled via default paths or logic.
        bands : Optional[List[int]]
            List of band indices to extract from raster data. Can be `None` to include all bands.
        band_names : Optional[List[Union[str, BandName]]]
            List of band names or enumeration values to use for naming bands in raster data. Can be `None`.
        sat_pass_filter : Union[SatellitePass, str, None]
            Satellite pass filter. Only samples matching this satellite pass are included. Can be `None`.
        beams_filter : Optional[str]
            Beam filter. Only samples matching this beam string are included. Can be `None`.

        Returns
        -------
        instance : cls
            An instance of the class containing the created samples dataset.
        """
        samples = []
        logger.debug(f'Reading CSV dataset from {csv_path}')

        if isinstance(sat_pass_filter, str):
            sat_pass_filter = SatellitePass.from_str(sat_pass_filter)
        logger.info("BEAM FILTER: {}".format(beams_filter))
        logger.info("SATELLITE PASS FILTER: {}".format(sat_pass_filter))
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                sat_pass = SatellitePass.from_str(row["sat_pass"])
                beam = row["beam"]

                if (sat_pass_filter is None or sat_pass == sat_pass_filter) and (
                        beams_filter is None or beam.upper() == beams_filter.upper()
                ):
                    cell_id = row["cell_id"]
                    group_date_post = row["group_date_post"]
                    year = int(group_date_post[:4]) + 1

                    pre_path = (
                        row["pre_path"]
                        .replace("$ROOT_PATH", data_directory or "data/raw")
                        .strip()
                    )
                    post_path = (
                        row["post_path"]
                        .replace("$ROOT_PATH", data_directory or "data/raw")
                        .strip()
                    )
                    label_path = os.path.join(
                        data_directory or "data/raw",
                        cell_id,
                        "static_data",
                        f"{cell_id}_nbac_{year}_prefire_100m.tif",
                    )
                    water_mask_path = os.path.join(
                        data_directory or "data/raw",
                        cell_id,
                        "static_data",
                        f"{cell_id}_WATER_mask_100m.tif",
                    )

                    with rasterio.open(pre_path) as src:
                        bounds = src.bounds
                        tif_transform = src.transform
                        crs = src.crs

                    samples.append(
                        {
                            "pre_path": pre_path,
                            "post_path": post_path,
                            "cell_id": cell_id,
                            "group_date_post": group_date_post,
                            "label_path": label_path,
                            "water_mask_path": water_mask_path,
                            "db_nbac_fire_id": row["db_nbac_fire_id"],
                            "sat_pass": row["sat_pass"],
                            "beam": row["beam"],
                            "group_date_pre": row["group_date_pre"],
                            "group_id_pre": row["group_id_pre"],
                            "group_id_post": row["group_id_post"],
                            "bounds": bounds,
                            "tif_transform": tif_transform,
                            "crs": str(crs),
                        }
                    )

        return cls(samples, transform=transform, bands=bands, band_names=band_names)

    # ----------------------------------------------------------------------
    # Dataset protocol
    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def train_val_test_split(self, ratios=(0.7, 0.15, 0.15), seed: int = 42):
        """
        Split dataset into train/val/test.

        Args:
            ratios (tuple): (train, val, test) ratios. Must sum to 1.0.
            seed (int): Seed for random number generator. Default is 42, the answer to life, the universe, and everything...

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        assert abs(sum(ratios) - 1.0) < 1e-6, "Les ratios doivent sommer à 1.0"
        return torch.utils.data.random_split(
            self, ratios, generator=torch.Generator().manual_seed(seed)
        )

    def get_train_val_test_dataloaders(
            self,
            batch_size=8,
            ratios=(0.7, 0.15, 0.15),
            seed: int = 42,
            num_workers: int = 4,
            pin_memory: bool = True,
            **kwargs,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get DataLoader for train/val/test. The method splits the dataset into train/val/test and returns a DataLoader for each.

        Args:
            ratios (tuple): (train, val, test) ratio for separability
            seed (int): reproductibilité
            batch_size (int): batch size (for DataLoader)
            num_workers (int): number of workers (for DataLoader)
            pin_memory (bool): pin memory (for DataLoader)
            kwargs: additional arguments for DataLoader (e.g. shuffle, sampler, collate_fn, drop_last, etc.)


        Returns:
            (train_loader, val_loader, test_loader)
        """
        train_ds, val_ds, test_ds = self.train_val_test_split(ratios, seed)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )

        return train_loader, val_loader, test_loader

    @staticmethod
    def _to_hwc(img: np.ndarray) -> np.ndarray:
        """Convert [C,H,W] / [H,W] to [H,W,C]."""
        if img.ndim == 2:  # [H,W]
            img = np.expand_dims(img, axis=0)  # -> [1,H,W]
        return np.transpose(img, (1, 2, 0))  # -> [H,W,C]

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

        image_pre, image_post = normalization(image_pre), normalization(image_post)
        mean = torch.tensor(self.norm_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.norm_stats["std"], dtype=torch.float32).view(-1, 1, 1)

        image_pre = standardization(image_pre, mean, std)
        image_post = standardization(image_post, mean, std)

        mask, mask_name = self._load_mask(index)
        mask = self._apply_common_mask_to_tensor(common_mask_tensor, mask)

        sample = {"image": image_post,
                  "image_pre": image_pre,
                  "mask": mask,
                  "image_pre_name": image_pre_name,
                  "image_name": image_post_name,
                  "mask_name": mask_name,
                  "mean": mean,
                  "std": std}

        return sample

    def __getitem__(self, idx) -> dict[str, Any]:
        """
        Retrieves and processes a dataset sample at the specified index. This method handles the loading of pre-event,
        post-event images, labels, and additional metadata for a specific entry in the dataset. The images are loaded
        including a mandatory bitmask, and the final data is optionally transformed if a transform function is defined.

        The method also applies a mask to the images and labels to ensure that only the common area between
        pre- and post-event images is used. Then, the water mask is applied to the images to ensure that only
        non-water areas are considered. The processed images and labels are returned as tensors along with the
        sample metadata.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve from the dataset.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the following keys:
            - "pre_img": The processed pre-event image data as a tensor.
            - "post_img": The processed post-event image data as a tensor.
            - "label": The processed label data corresponding to the input images as a tensor.
            - "meta": A dictionary containing metadata about the sample. This includes the original sample metadata with additional keys added for geospatial postprocessing.
        """
        sample = self.samples[idx]

        # Always include BITMASK_CROPPED (band 1)
        bands_to_read = self.bands if self.bands is not None else None
        if bands_to_read is not None and 1 not in bands_to_read:
            bands_to_read_with_mask = [1] + bands_to_read
        else:
            bands_to_read_with_mask = bands_to_read

        # Load pre and post images
        pre_img_full = self._load_image_with_dtype(
            sample["pre_path"], bands=bands_to_read_with_mask
        )
        post_img_full = self._load_image_with_dtype(
            sample["post_path"], bands=bands_to_read_with_mask
        )

        common_mask = self._get_images_common_area(post_img_full, pre_img_full)

        # Water mask (raw)
        raw_water_mask = self._load_water_mask(sample)
        water_mask_bool = (raw_water_mask == 0) if raw_water_mask is not None else None

        # Final mask = common mask AND water mask
        if common_mask is not None and water_mask_bool is not None:
            final_mask = common_mask & water_mask_bool
        else:
            final_mask = common_mask

        # Apply mask on images (excluding BITMASK_CROPPED)
        def mask_img(img_full):
            if img_full is None:
                return None
            img = img_full[1:] if img_full.shape[0] > 1 else img_full
            if final_mask is not None:
                img = img.astype("int16")
                for i in range(img.shape[0]):
                    img[i][~final_mask] = NO_DATA
            return img

        pre_img = mask_img(pre_img_full)
        post_img = mask_img(post_img_full)

        # Apply mask on label
        label = self._load_label(sample)

        def mask_single(img):
            if img is None or final_mask is None:
                return img
            img = img.astype("int16")
            img_masked = img.copy()
            img_masked[~final_mask] = NO_DATA
            return img_masked

        label = mask_single(label)

        # Apply transforms
        if self.transform:
            logger.debug("Applying transform")
            pre_img, post_img, label = self._transform_data(pre_img, post_img, label)
        else:
            pre_img, post_img, label = (
                torch.from_numpy(pre_img).float(),
                torch.from_numpy(post_img).float(),
                torch.from_numpy(label).float(),
            )

        pre_img, post_img = self.add_pass_and_beam_in_out_bands(pre_img, post_img, sample)

        return {
            "image": post_img,
            "image_pre": pre_img,
            "mask": label,
            "meta": sample,
        }

    def _get_images_common_area(self, post_img_full, pre_img_full) -> bool | Any:
        # Extract BITMASK_CROPPED (always first channel)
        pre_mask = pre_img_full[0] if pre_img_full is not None else None
        post_mask = post_img_full[0] if post_img_full is not None else None

        # Common mask = areas where both masks == 1
        if pre_mask is not None and post_mask is not None:
            common_mask = (pre_mask == 1) & (post_mask == 1)
        else:
            common_mask = None
        return common_mask

    @staticmethod
    def add_pass_and_beam_in_out_bands(pre_img, post_img, sample):
        H, W = pre_img.shape[1], pre_img.shape[2]

        # encode sat_pass : Ascending=0, Descending=1
        sat_pass_val = 0 if sample["sat_pass"].upper().startswith("A") else 1
        sat_pass_band = torch.full((1, H, W), sat_pass_val, dtype=pre_img.dtype)

        # encode beam (A,B,C,D → 0,1,2,3)
        beam_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        beam_val = beam_map.get(sample["beam"].upper(), -1)
        beam_band = torch.full((1, H, W), beam_val, dtype=pre_img.dtype)

        # concat à pre_img et post_img
        pre_img = torch.cat([pre_img, sat_pass_band, beam_band], dim=0)
        post_img = torch.cat([post_img, sat_pass_band, beam_band], dim=0)
        return pre_img, post_img

    def _transform_data(
            self, pre_img, post_img, label
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply transformation to data. This method applies the transformation function to the input data and returns
        pre- and post-event images and labels as tensors.
        """
        # Albumentations attend (H,W,C) -> donc transpose si nécessaire
        pre_img = self._to_hwc(pre_img)  # [C,H,W] -> [H,W,C]
        post_img = self._to_hwc(post_img)

        augmented = self.transform(image=post_img, image_pre=pre_img, mask=label)
        post_img = augmented["image"].float()
        pre_img = augmented["image_pre"].float()
        label = augmented["mask"].float()

        return pre_img, post_img, label

    # ----------------------------------------------------------------------
    # Image / label loaders
    # ----------------------------------------------------------------------
    @staticmethod
    def _load_single_band_image(path: str, common_mask=None) -> Optional[np.ndarray]:
        """Load single band image."""
        if not os.path.exists(path):
            return None
        with rasterio.open(path, dtype="int16") as src:
            try:
                img = src.read(1).astype("int16")
                if common_mask is not None:
                    img[~common_mask] = NO_DATA
                return img
            except Exception as e:
                logger.error(f"Error reading {path}: {e}")
                return None

    def _load_image(self, index: int) -> tuple[Tensor, Tensor, Tensor, str, str]:
        """Load pre-post images.
        return : Tensor pre, Tensor post, str pre_name, str post_name
        """
        image_pre = self.files[index]["image_pre"]
        image_post = self.files[index]["image"]
        image_pre_name = Path(image_pre).name
        image_post_name = Path(image_pre).name

        image_pre_tensor, pre_data_mask = self.convert_tif_to_tensor(image_pre)
        image_post_tensor, post_data_mask = self.convert_tif_to_tensor(image_post)
        common_mask = pre_data_mask.all(0) & post_data_mask.all(0)
        common_mask_tensor = torch.from_numpy(common_mask)

        return image_pre_tensor, image_post_tensor, common_mask_tensor, image_pre_name, image_post_name

    @staticmethod
    def _apply_common_mask_to_tensor(common_mask_tensor: Tensor, in_image_tensor: Tensor) -> Tensor:
        in_image_tensor[:, ~common_mask_tensor] = float("nan")
        return in_image_tensor

    @staticmethod
    def convert_tif_to_tensor(in_image: str) -> tuple[Tensor, bool | ndarray[tuple[Any, ...], dtype[Any]] | Any]:
        def load_with_mask(path: str):
            with rio.open(path) as src:
                arr = src.read().astype(np.int32)  # shape (C,H,W)
                nodata = src.nodata
                if nodata is not None:
                    mask = (arr != nodata)
                else:
                    mask = ~np.isnan(arr)

            return arr, mask

        img_array, no_data_mask = load_with_mask(in_image)
        img_as_tensor = torch.from_numpy(img_array).float()
        return img_as_tensor, no_data_mask

    def _load_mask(self, index: int) -> tuple[Tensor, str]:
        """Load mask."""
        mask_path = self.files[index]["mask"]
        mask_name = Path(mask_path).name
        with rio.open(mask_path) as mask:
            mask_array = mask.read().astype(np.int32)
            mask_tensor = torch.from_numpy(mask_array).float()

        return mask_tensor, mask_name

    # def _load_image(self, path: str) -> Optional[np.ndarray]:
    #     """Load image with multiple bands."""
    #     if not os.path.exists(path):
    #         return None
    #     with rasterio.open(path) as src:
    #         try:
    #             if self.bands is not None:
    #                 return src.read(indexes=self.bands)
    #             return src.read()
    #         except Exception as e:
    #             logger.error(f"Error reading {path}: {e}")
    #             return None

    # @staticmethod
    # def _load_image_with_dtype(path: str, bands=None) -> Optional[np.ndarray]:
    #     if not os.path.exists(path):
    #         return None
    #     with rasterio.open(path, dtype="int16") as src:
    #         try:
    #             img = src.read(indexes=bands) if bands is not None else src.read()
    #             return img.astype("int16")
    #         except Exception as e:
    #             logger.error(f"Error reading {path}: {e}")
    #             return None

    # def _load_label(
    #         self, sample: Dict[str, Any], common_mask=None
    # ) -> Optional[np.ndarray]:
    #     """Load label image."""
    #     return self._load_single_band_image(sample["label_path"], common_mask)
    #
    # def _load_water_mask(
    #         self, sample: Dict[str, Any], common_mask=None
    # ) -> Optional[np.ndarray]:
    #     """Load water mask image."""
    #     return self._load_single_band_image(sample["water_mask_path"], common_mask)

    # ----------------------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------------------
    def visualize_sample(self, idx: int, show_label: bool = True, n_runs: int = 1):
        """Visualize a dataset sample with matplotlib."""
        import matplotlib.pyplot as plt

        sample = self.samples[idx]

        # Metadata
        cell_id = sample.get("cell_id", "N/A")
        db_nbac_fire_id = sample.get("db_nbac_fire_id", "N/A")
        sat_pass = sample.get("sat_pass", "N/A")
        beam = sample.get("beam", "N/A")
        pre_name = (
            f"GID{sample.get('group_id_pre', '')} - {sample.get('group_date_pre', '')}"
        )
        post_name = f"GID{sample.get('group_id_post', '')} - {sample.get('group_date_post', '')}"

        # Prepare display images (use first band)
        def prepare_img(img):
            if img is None:
                return None
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            img_to_show = img[0].copy() if img.ndim == 3 else img.copy()
            img_to_show[img_to_show == NO_DATA] = 0
            return img_to_show

        # Plot
        fig, axes = plt.subplots(n_runs, 4, figsize=(18, 5 * n_runs))
        if n_runs == 1:  # si une seule ligne, axes est 1D → rendre en 2D
            axes = np.expand_dims(axes, 0)

        for run in range(n_runs):
            row_axes = axes[run] if n_runs > 1 else axes
            data = self.__getitem__(idx)
            pre_img, post_img, label = data["image_pre"], data["image"], data["mask"]
            pre_to_show = prepare_img(pre_img)
            post_to_show = prepare_img(post_img)
            label_to_show = prepare_img(label) if label is not None else None
            if label_to_show is not None:
                label_to_show[label_to_show == NO_DATA] = 0

            if pre_to_show is not None:
                row_axes[0].imshow(pre_to_show, cmap="gray")

            row_axes[0].axis("off")

            if post_to_show is not None:
                row_axes[1].imshow(post_to_show, cmap="gray")
            row_axes[1].axis("off")

            if label_to_show is not None:
                row_axes[2].imshow(label_to_show, cmap="jet")
            row_axes[2].axis("off")

            if post_to_show is not None:
                row_axes[3].imshow(post_to_show, cmap="gray")
                if show_label and label_to_show is not None:
                    row_axes[3].imshow(label_to_show, cmap="jet", alpha=0.4)
            row_axes[3].axis("off")
            if run == 0:
                row_axes[0].set_title(
                    f"Pre-{pre_name}",
                )
                row_axes[1].set_title(f"Post-{post_name}")
                row_axes[2].set_title("Label")
                row_axes[3].set_title("Post + Label overlay")

        band_names_str = ", ".join(self.band_names) if self.band_names else "N/A"
        plt.suptitle(
            f"{sat_pass}/{beam} - cell_id: {cell_id} | db_nbac_fire_id: {db_nbac_fire_id}\n"
            f"Bands used: {band_names_str}",
            fontsize=29,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()


if __name__ == "__main__":
    csv_dataset = "C:\\Users\\xmalet\\PycharmProjects\\scanfire_segmentation\\data\\pre_post_datasets.csv"
    root_path = r"C:\Users\xmalet\PycharmProjects\scanfire_segmentation\data\raw"

    dataset = PrePostChangeDataset.build_dataset_from_csv(
        csv_dataset,
        band_names=[BandName.M, BandName.RL, BandName.RR],
        data_directory=root_path,
        beams_filter="A",
    )
    print(f"Dataset size - ALL: {len(dataset)}")
    dataset.visualize_sample(25, show_label=True, n_runs=10)
