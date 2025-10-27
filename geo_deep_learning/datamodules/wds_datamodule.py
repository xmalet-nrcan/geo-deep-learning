"""Multi-sensor data module for webdataset."""

import logging
import math

import torch
import webdataset as wds
from lightning.pytorch import LightningDataModule
from webdataset import WebLoader

from geo_deep_learning.datasets.wds_dataset import create_sensor_datasets

logger = logging.getLogger(__name__)


class MultiSensorDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for multi-sensor Earth observation data."""

    def __init__(  # noqa: PLR0913
        self,
        sensor_configs_path: str,
        model_type: str = "clay",
        patch_size: tuple[int, int] = (512, 512),
        epoch_size: int | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        shuffle_buffer: int = 0,
        shardshuffle: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize MultiSensorDataModule.

        Args:
            sensor_configs_path: Path to YAML config with sensor configurations
            model_type: Output format - "clay", "dofa", or "unified"
            patch_size: Target patch size for augmentations
            epoch_size: Number of iterations (batches) per epoch
            batch_size: Batch size for all dataloaders
            num_workers: Number of worker processes
            prefetch_factor: Number of batches to prefetch
            shuffle_buffer: Number of batches to prefetch for shuffling
            shardshuffle: Number of shards to shuffle
            seed: Random seed for shuffling

        """
        super().__init__()

        self.sensor_configs_path = sensor_configs_path
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle_buffer = shuffle_buffer
        self.shardshuffle = shardshuffle
        self.seed = seed
        self.patch_size = patch_size
        self.epoch_size = epoch_size
        self.datasets = {}
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.trn_patch_count = 0
        self.val_patch_count = 0
        self.tst_patch_count = 0

    def prepare_data(self) -> None:
        """Prepare data."""

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Create datasets for each stage."""
        self.datasets = create_sensor_datasets(
            sensor_configs_path=self.sensor_configs_path,
            model_type=self.model_type,
            batch_size=self.batch_size,
            epoch_size=self.epoch_size,
            shuffle_buffer=self.shuffle_buffer,
            shardshuffle=self.shardshuffle,
            seed=self.seed,
        )
        self._compute_patch_counts()
        self._setup_train_loader()
        self._setup_val_loader()
        self._setup_test_loader()

    def _compute_patch_counts(self) -> None:
        """Collect total patch counts from all sensors."""
        for splits in self.datasets.values():
            if "trn" in splits:
                self.trn_patch_count += splits["trn"].patch_count
            if "val" in splits:
                self.val_patch_count += splits["val"].patch_count
            if "tst" in splits:
                self.tst_patch_count += splits["tst"].patch_count
        logger.info(
            "Total patch counts - Train: %d, Val: %d, Test: %d",
            self.trn_patch_count,
            self.val_patch_count,
            self.tst_patch_count,
        )

    def _calculate_epoch_size(self, total_patches: int, split: str = "trn") -> int:
        """
        Calculate epoch_size.

        Args:
            total_patches: Total number of patches in the dataset
            split: Dataset split (trn, val, tst)

        Returns:
            Number of iterations (batches) per epoch

        """
        world_size = 1
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        patches_per_gpu = math.ceil(total_patches / world_size)
        iterations_per_gpu = math.ceil(patches_per_gpu / self.batch_size)
        logger.info(
            "Auto-calculated epoch_size for %s split: %d iterations "
            "(total_patches=%d, world_size=%d, batch_size=%d, patches_per_gpu=%d)",
            split,
            iterations_per_gpu,
            total_patches,
            world_size,
            self.batch_size,
            patches_per_gpu,
        )
        return iterations_per_gpu

    def _setup_train_loader(self) -> None:
        """Set up training loader with sensor mixing."""
        train_datasets = {}
        for sensor_name, splits in self.datasets.items():
            if "trn" in splits:
                wds_dataset = splits["trn"].build_web_dataset()
                if wds_dataset is not None:
                    train_datasets[sensor_name] = wds_dataset

        if not train_datasets:
            logger.warning("No training datasets found!")
            return

        if len(train_datasets) == 1:
            # Single sensor
            sensor_name = next(iter(train_datasets.keys()))
            sensor_dataset = train_datasets[sensor_name]
        else:
            # Multiple sensors
            sensor_dataset = self._create_mixed_dataset(train_datasets)

        self.train_loader = WebLoader(
            sensor_dataset,
            num_workers=self.num_workers,
            batch_size=None,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=(self.num_workers > 0),
        )
        epoch_size = self.epoch_size
        if epoch_size is None:
            epoch_size = self._calculate_epoch_size(self.trn_patch_count, "trn")
        self.train_loader = self.train_loader.with_epoch(epoch_size)

    def _setup_val_loader(self) -> None:
        """Set up validation loader."""
        val_datasets = {}
        for sensor_name, splits in self.datasets.items():
            if "val" in splits:
                wds_dataset = splits["val"].build_web_dataset()
                if wds_dataset is not None:
                    val_datasets[sensor_name] = wds_dataset

        if not val_datasets:
            logger.warning("No validation datasets found!")
            return

        if len(val_datasets) == 1:
            # Single sensor
            sensor_name = next(iter(val_datasets.keys()))
            sensor_dataset = val_datasets[sensor_name]
        else:
            # Multiple sensors
            sensor_dataset = self._create_mixed_dataset(val_datasets)

        self.val_loader = WebLoader(
            sensor_dataset,
            num_workers=self.num_workers,
            batch_size=None,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=(self.num_workers > 0),
        )
        epoch_size = self._calculate_epoch_size(self.val_patch_count, "val")
        self.val_loader = self.val_loader.with_epoch(epoch_size)

    def _setup_test_loader(self) -> None:
        """Set up test loader."""
        test_datasets = {}
        for sensor_name, splits in self.datasets.items():
            if "tst" in splits:
                wds_dataset = splits["tst"].build_web_dataset()
                if wds_dataset is not None:
                    test_datasets[sensor_name] = wds_dataset

        if not test_datasets:
            logger.info("No test datasets found - this is optional")
            return

        if len(test_datasets) == 1:
            # Single sensor
            sensor_name = next(iter(test_datasets.keys()))
            sensor_dataset = test_datasets[sensor_name]
        else:
            # Multiple sensors
            sensor_dataset = self._create_mixed_dataset(test_datasets)

        self.test_loader = WebLoader(
            sensor_dataset,
            num_workers=self.num_workers,
            batch_size=None,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=(self.num_workers > 0),
        )

    def _create_mixed_dataset(
        self,
        sensor_datasets: dict[str, wds.WebDataset],
    ) -> wds.WebDataset:
        """Create a mixed dataset from multiple sensor datasets."""
        datasets_list = list(sensor_datasets.values())
        # Round robin through sensors
        return wds.RandomMix(
            datasets=datasets_list,
            probs=None,  # Equal probability
            longest=True,
        )

    def train_dataloader(self) -> WebLoader:
        """Create training DataLoader."""
        return self.train_loader

    def val_dataloader(self) -> WebLoader:
        """Create validation DataLoader."""
        return self.val_loader

    def test_dataloader(self) -> WebLoader:
        """Create test DataLoader."""
        return self.test_loader

    def teardown(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Clean up after training/testing."""
        if hasattr(self, "datasets"):
            self.datasets.clear()
