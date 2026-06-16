"""RcmChangeDetectionDataModule."""
import logging
from collections import defaultdict
from typing import Any, Optional, List, Iterable, Type

import numpy as np
import torch
import torch.utils.data as data
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset

from geo_deep_learning.datasets.rcm_change_detection_dataset import RCMChangeDetectionDataset


from geo_deep_learning.datasets.rcm_change_detection_dataset import bands_stats
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s - %(name)s - [%(levelname)s] ] - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


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
            bands: Optional[List[int]] = None,
            band_names: Optional[List[str]] = None,
            satellite_pass: Optional[str] = None,
            beams: Optional[List[str]] = None,
            dataset_years : Optional[list[int]] = None,
            split_ratios=(0.70, 0.15, 0.15),
            split_on_columns: Optional[str | list] = None,
            dataset_class: Type[RCMChangeDetectionDataset] = RCMChangeDetectionDataset,
            separate_metadata: bool = True,

    ) -> None:
        """Initialize RcmChangeDetectionDataModule.

        Args:
            separate_metadata: When True (default), the dataset will NOT
                concatenate COMMON_MASK / SAT_PASS / BEAM to the image tensors.
                Instead they are returned as separate dict entries for FiLM
                conditioning.  Set to False for legacy 13-channel behaviour.
        """
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
        self.dataset_class = dataset_class
        self._dataset_years = dataset_years
        self.separate_metadata = separate_metadata

        self.dataset: RCMChangeDetectionDataset = None
        if split_on_columns is None:
            self._split_on_columns = None
        elif isinstance(split_on_columns, str):
            self._split_on_columns = split_on_columns
        elif isinstance(split_on_columns, Iterable):
            self._split_on_columns = list(split_on_columns)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Create dataset."""
        self.dataset = self.dataset_class(
            split_or_csv_file_name=self.csv_file_name,
            norm_stats=self.norm_stats,
            csv_root_folder=self.csv_root_folder,
            patches_root_folder=self.patches_root_folder,
            bands=self.bands,
            band_names=self.band_names,
            satellite_pass=self.satellite_pass,
            beams=self.beams,
            dataset_years=self._dataset_years,
            separate_metadata=self.separate_metadata,
        )

        if stage != "predict":
            self._set_train_test_val_datasets()

    def _split_by_column(self, column_name, split_ratios=(0.7, 0.15, 0.15), seed=42):
        """
        Split the dataset by unique values in `column_name`, ensuring that all samples
        sharing the same value are placed in the same subset.
        Uses a round-robin assignment strategy while approximately respecting ratio targets.
        """
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "Ratios must sum to 1."
        is_multi = isinstance(column_name, (list, tuple))
        logger.debug(f"Is split on column multi ? : {is_multi}")

        # Count samples per unique group and gather indices per group ---
        value_counts = defaultdict(int)
        value_indices = defaultdict(list)
        for i, sample in enumerate(self.dataset.files):
            key = tuple(sample[col] for col in column_name) if is_multi else sample[column_name]
            value_counts[key] += 1
            value_indices[key].append(i)

        # # Shuffle groups randomly for fair distribution ---
        rng = np.random.default_rng(seed)
        unique_values = list(value_counts.keys())
        rng.shuffle(unique_values)

        # Initialize split containers and targets ---
        total = len(self.dataset.files)
        ratios = dict(zip(["train", "test", "val"], split_ratios))
        splits = {
            name: {"target": ratios[name] * total,
                   "count": 0,
                   "indices": []}
            for name in ratios
        }
        cycle = list(ratios.keys())
        idx = 0

        # Round-robin assignment with target constraints ---
        for value in unique_values:
            count = value_counts[value]
            assigned = False
            attempts = 0
            group_indices = value_indices[value]

            while not assigned and attempts < len(cycle):
                name = cycle[idx % len(cycle)]
                if splits[name]["count"] + count <= splits[name]["target"]:
                    splits[name]["count"] += count
                    splits[name]["indices"].extend(group_indices)
                    assigned = True
                else:
                    idx += 1
                    attempts += 1

            if not assigned:
                # Toutes pleines : on ajoute au plus petit split actuel
                name = min(splits, key=lambda k: splits[k]["count"])
                splits[name]["indices"].extend(group_indices)
                splits[name]["count"] += count

            idx += 1  # passe au subset suivant

        # Log final subset statistics ---
        for name in cycle:
            pct = splits[name]["count"] / total
            logger.debug(f"{name.capitalize():<5}: {splits[name]['count']} ({pct:.2%})")

        # Return the dataset subsets ---
        return (
            Subset(self.dataset, splits["train"]["indices"]),
            Subset(self.dataset, splits["val"]["indices"]),
            Subset(self.dataset, splits["test"]["indices"]),
        )

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

    def predict_dataloader(self) -> DataLoader[Any]:
        """Dataloader for prediction (full dataset)."""
        return DataLoader(
            self.dataset,  # tout le dataset sans split
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=False,
        )