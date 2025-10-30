"""RcmChangeDetectionDataModule."""

from typing import Any, Optional, List, Iterable

import numpy as np
import torch
import torch.utils.data as data
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from geo_deep_learning.datasets.rcm_change_detection_dataset import SatellitePass, RCMChangeDetectionDataset

bands_stats = {
    'mean': [1.0088686544882763, 22.678325648034726, 4820.030168929148, -578.1138439754548, 174.35119966169816,
             4645.179761547494, 5178.970253993203, 4074.12440505587, 1427.3155618129722, 517.5479435073069,
             1945.2480656061873, 514.8092047489475, 425.98675130681056, 8939.542957169055],
    'std': [0.17514777918322952, 4.602293040200134, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan,
            np.nan],
    'min': [1.0, 0.0, 446.0, -9810.0, 0.0, 81.0, 14.0, 93.0, 2.0, 2.0, 5.0, -9340.0, -9584.0, -8947.0],
    'max': [9.0, 112.0, 9985.0, 9969.0, 6358.0, 9971.0, 9553.0, 9979.0, 32766.0, 32766.0, 32766.0, 9484.0,
            9901.0, 9999.0]
    }


class RcmChangeDetectionDataModule(LightningDataModule):
    """RCM Change Detection DataModule."""

    def __init__(  # noqa: PLR0913
            self,
            csv_root_folder: str,
            csv_file_name: str,
            patches_root_folder: str,
            batch_size: int = 16,
            num_workers: int = 8,
            patch_size: tuple[int, int] = (256, 256),
            mean: list[float] | None = None,
            std: list[float] | None = None,
            bands: Optional[List[int]] = None,
            band_names: Optional[List[str]] = None,
            satellite_pass: Optional[str | SatellitePass] = None,
            beams: Optional[List[str]] = None,
            split_ratios=(0.70, 0.15, 0.15),
            split_on_columns: Optional[str] = None,
            data_type_max: Optional[int] = None,

    ) -> None:
        """Initialize CSVDataModule."""
        super().__init__()

        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.csv_root_folder: str = csv_root_folder[0] if isinstance(csv_root_folder,
                                                                     (list, tuple)) else csv_root_folder
        self.csv_file_name = csv_file_name
        self.patches_root_folder = patches_root_folder
        self.norm_stats = bands_stats
        self.bands = bands
        self.band_names = band_names
        self.satellite_pass = satellite_pass
        self.beams = beams
        self.split_ratios = split_ratios
        self.dataset: RCMChangeDetectionDataset = None
        if split_on_columns is None:
            self._split_on_columns = None
        elif isinstance(split_on_columns, str):
            self._split_on_columns = split_on_columns
        elif isinstance(split_on_columns, Iterable):
            self._split_on_columns = list(split_on_columns)[0]

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Create dataset."""
        self.dataset = RCMChangeDetectionDataset(
            split_or_csv_file_name=self.csv_file_name,
            norm_stats=self.norm_stats,
            csv_root_folder=self.csv_root_folder,
            patches_root_folder=self.patches_root_folder,
            bands=self.bands,
            band_names=self.band_names,
            satellite_pass=self.satellite_pass,
            beams=self.beams
        )

        self._set_train_test_val_datasets()

    def _split_by_column(self, column_name, split_ratios=(0.7, 0.15, 0.15), seed=42):
        """Split dataset by unique values in a specified column."""

        # Récupère la valeur de la colonne pour chaque sample
        values = [sample[column_name] for sample in self.dataset.files]
        unique_values = list(set(values))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_values)

        n = len(unique_values)
        n_train = int(split_ratios[0] * n)
        n_val = int(split_ratios[1] * n)

        train_values = set(unique_values[:n_train])
        val_values = set(unique_values[n_train:n_train + n_val])
        test_values = set(unique_values[n_train + n_val:])

        train_indices = [i for i, v in enumerate(values) if v in train_values]
        val_indices = [i for i, v in enumerate(values) if v in val_values]
        test_indices = [i for i, v in enumerate(values) if v in test_values]

        return Subset(self.dataset, train_indices), Subset(self.dataset, val_indices), Subset(self.dataset,
                                                                                              test_indices)

    def _set_train_test_val_datasets(self):
        if self._split_on_columns is not None:
            self.train_dataset, self.val_dataset, self.test_dataset = self._split_by_column(
                column_name=self._split_on_columns,
                split_ratios=self.split_ratios,
                seed=42
            )
        else:
            self.train_dataset, self.val_dataset, self.test_dataset = data.random_split(
                self.dataset, self.split_ratios,
                generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Dataloader for training."""

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Dataloader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Dataloader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=False,
        )


if __name__ == "__main__":
    in_csv_root_folder = r"C:\Users\xmalet\PycharmProjects\geo-deep-learning\data"
    in_patches_root_folder = r"C:\Users\xmalet\PycharmProjects\geo-deep-learning\data\raw"
    dataset = RcmChangeDetectionDataModule(
        csv_root_folder=in_csv_root_folder,
        patches_root_folder=in_patches_root_folder,
        csv_file_name=r"pre_post_datasets_all.csv",
        patch_size=(256, 256),
        band_names=['M', 'RL', 'RR', 'S0'],
        beams=['A'],
        split_on_columns='cell_id')
    dataset.setup()

    tdl = dataset.train_dataloader()
    val = dataset.val_dataloader()
    test = dataset.test_dataloader()

    print(len(tdl.dataset))
    print(len(val.dataset))
    print(len(test.dataset))

    # print(f"mean:{dataset.mean}, std:{dataset.std}")
