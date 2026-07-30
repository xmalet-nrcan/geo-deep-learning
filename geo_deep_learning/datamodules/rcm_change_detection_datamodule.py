"""RcmChangeDetectionDataModule."""
import csv
import logging
from collections import defaultdict
from pathlib import Path
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
            tile_size: tuple[int, int] | None = None,
            tile_stride: tuple[int, int] | None = None,
            predict_overlap_buffer: int = 0,

    ) -> None:
        """Initialize RcmChangeDetectionDataModule.

        Args:
            separate_metadata: When True (default), the dataset will NOT
                concatenate COMMON_MASK / SAT_PASS / BEAM to the image tensors.
                Instead they are returned as separate dict entries for FiLM
                conditioning.  Set to False for legacy 13-channel behaviour.
            tile_size: Crop size for tiling large images (e.g. ``(512, 512)``).
                Images smaller than this are returned as-is.
                ``None`` (default) disables tiling entirely.
            tile_stride: Step between tile origins.  Defaults to *tile_size*
                (no overlap).  Use a smaller value for overlapping tiles.
            predict_overlap_buffer: Number of pixels to load from each
                neighboring cell on every side.  **Only used at predict time**
                — ignored for train/val/test.  Passed to the dataset class
                only when ``stage == "predict"``.
                Set to 0 (default) to disable.
                For 50 % overlap on 200×200 cells, use 100.
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
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self._predict_overlap_buffer = predict_overlap_buffer

        self.dataset: RCMChangeDetectionDataset = None
        if split_on_columns is None:
            self._split_on_columns = None
        elif isinstance(split_on_columns, str):
            self._split_on_columns = split_on_columns
        elif isinstance(split_on_columns, Iterable):
            self._split_on_columns = list(split_on_columns)

    def setup(self, stage: str | None = None) -> None:
        """Create dataset."""
        ds_kwargs = dict(
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
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )
        # Only enable overlap buffer at predict time — training uses
        # independent tiles without spatial context from neighbours.
        if stage == "predict" and self._predict_overlap_buffer > 0:
            ds_kwargs["predict_overlap_buffer"] = self._predict_overlap_buffer
        try:
            self.dataset = self.dataset_class(**ds_kwargs)
        except TypeError:
            # Fallback: dataset class doesn't accept predict_overlap_buffer
            ds_kwargs.pop("predict_overlap_buffer", None)
            self.dataset = self.dataset_class(**ds_kwargs)

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
        self._log_split_contents()

    def _log_split_contents(self) -> None:
        """Log and save to CSV the unique fire IDs and group IDs in each split.

        Produces one CSV per split (train/val/test) with columns:
            split, db_nbac_fire_id, group_id_pre, group_id_post, cell_id, beam, sat_pass

        Files are saved to the current working directory as:
            split_contents_train.csv, split_contents_val.csv, split_contents_test.csv
        """
        splits = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset,
        }

        for split_name, subset in splits.items():
            if subset is None:
                continue

            # Extract indices from Subset or RandomSplit
            if hasattr(subset, "indices"):
                indices = subset.indices
            else:
                indices = range(len(subset))

            # Collect unique identifiers
            fire_ids = set()
            group_ids = set()
            rows = []
            for idx in indices:
                sample = self.dataset.files[idx]
                fire_id = sample.get("db_nbac_fire_id", sample.get("event_id", "N/A"))
                gid_pre = sample.get("group_id_pre", "N/A")
                gid_post = sample.get("group_id_post", "N/A")
                cell_id = sample.get("cell_id", "N/A")
                beam = sample.get("beam", "N/A")
                sat_pass = sample.get("sat_pass", "N/A")

                fire_ids.add(str(fire_id))
                group_ids.add((str(gid_pre), str(gid_post)))
                rows.append({
                    "split": split_name,
                    "db_nbac_fire_id": fire_id,
                    "group_id_pre": gid_pre,
                    "group_id_post": gid_post,
                    "cell_id": cell_id,
                    "beam": beam.name if hasattr(beam, "name") else beam,
                    "sat_pass": sat_pass.name if hasattr(sat_pass, "name") else sat_pass,
                })

            # Log summary
            logger.info(
                "Split %-5s: %d samples, %d unique fires, %d unique group pairs",
                split_name, len(indices), len(fire_ids), len(group_ids),
            )
            logger.info(
                "  Fire IDs (%s): %s",
                split_name, sorted(fire_ids),
            )

            # Save CSV
            csv_path = Path(self.csv_root_folder) / f"split_contents_{split_name}.csv"
            fieldnames = ["split", "db_nbac_fire_id", "group_id_pre", "group_id_post",
                          "cell_id", "beam", "sat_pass"]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            logger.info("  Saved split details to %s", csv_path.resolve())

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