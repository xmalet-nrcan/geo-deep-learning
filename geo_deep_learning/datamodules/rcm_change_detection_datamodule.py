"""RcmChangeDetectionDataModule."""

from typing import Any, Optional, List

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import torch.utils.data as data

from geo_deep_learning.datasets.rcm_change_detection_dataset import SatellitePass, BandName, Beams, RCMChangeDetectionDataset


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
            split_ratios = (0.70, 0.15, 0.15),

    ) -> None:
        """Initialize CSVDataModule."""
        super().__init__()

        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.csv_root_folder:str = csv_root_folder[0] if isinstance(csv_root_folder , (list, tuple)) else csv_root_folder
        self.csv_file_name =csv_file_name
        self.patches_root_folder = patches_root_folder
        self.norm_stats = {
            "mean": mean or [0.0, 0.0, 0.0],
            "std": std or [1.0, 1.0, 1.0],
        }
        self.bands = bands
        self.band_names = band_names
        self.satellite_pass = satellite_pass
        self.beams = beams
        self.split_ratios = split_ratios
        self.dataset = None


    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Create dataset."""
        self.dataset =  RCMChangeDetectionDataset(
            split_or_csv_file_name=self.csv_file_name,
            norm_stats=self.norm_stats,
            csv_root_folder=self.csv_root_folder,
            patches_root_folder=self.patches_root_folder,
            bands=self.bands,
            band_names=self.band_names,
            satellite_pass=self.satellite_pass,
            beams=self.beams
        )

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
        csv_file_name=r"pre_post_datasets.csv",)
    dataset.setup()


    # print(f"mean:{dataset.mean}, std:{dataset.std}")
