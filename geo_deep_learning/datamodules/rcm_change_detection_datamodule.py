"""RcmChangeDetectionDataModule."""
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, List, Iterable, Type

import numpy as np
import rasterio as rio
import torch
import torch.utils.data as data
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

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
            dataset_years: Optional[list[int]] = None,
            split_ratios=(0.70, 0.15, 0.15),
            split_on_columns: Optional[str | list] = None,
            dataset_class: Type[RCMChangeDetectionDataset] = RCMChangeDetectionDataset,
            separate_metadata: bool = True,
            tile_size: tuple[int, int] | None = None,
            tile_stride: tuple[int, int] | None = None,
            predict_overlap_buffer: int = 0,
            train_overlap_buffer: int = 0,
            burned_oversample_factor: float = 1.0,
            burned_tile_min_ratio: float = 0.0,
    ) -> None:
        """Initialize RcmChangeDetectionDataModule.

        Args:
            dataset_class: Dataset class used for **all** stages (train / val /
                test / predict).  Override in the YAML config to switch between
                ``RCMChangeDetectionDataset`` (training) and
                ``RCMChangeDetectionOnPredictDataset`` (inference).
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
                neighboring cell on every side at **predict time** only.
                Set to 0 (default) to disable.
                For 50 % overlap on 200×200 cells, use 100.
            train_overlap_buffer: Number of pixels to load from each
                neighboring cell on every side during **train / val / test**.
                Expands each cell with real SAR context from its 8 neighbours.
                The loss is computed only on the central cell pixels so that
                the buffer zone (which has no ground-truth label) is excluded.
                Set to 0 (default) to disable.
            burned_oversample_factor: Relative sampling weight given to
                training tiles that contain burned pixels, versus background-only
                tiles.  ``1.0`` (default) disables oversampling.  Values > 1
                enable a :class:`WeightedRandomSampler` on the **train split
                only** (val/test untouched, so metrics stay comparable).  This
                is a *data-level* rebalancing, complementary to the *loss-level*
                ``burned_class_weight`` — it fixes the tile-level imbalance that
                tiling aggravates (many pure-background tiles), which
                ``burned_class_weight`` cannot address (it has no positive pixel
                to up-weight on an empty tile).
            burned_tile_min_ratio: Minimum burned-pixel ratio (over the tile's
                valid central-cell zone) for a tile to be flagged "burned" and
                receive the higher sampling weight.  ``0.0`` (default) means any
                single burned pixel qualifies.
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
        self._train_overlap_buffer = train_overlap_buffer
        self.burned_oversample_factor = float(burned_oversample_factor)
        self.burned_tile_min_ratio = float(burned_tile_min_ratio)
        self._train_sample_weights: list[float] | None = None

        self.dataset: RCMChangeDetectionDataset = None
        if split_on_columns is None:
            self._split_on_columns = None
        elif isinstance(split_on_columns, str):
            self._split_on_columns = split_on_columns
        elif isinstance(split_on_columns, Iterable):
            self._split_on_columns = list(split_on_columns)

    def setup(self, stage: str | None = None) -> None:
        """Create dataset."""
        is_predict = (stage == "predict")

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

        # Inject the overlap buffer that matches the current stage
        buf = self._predict_overlap_buffer if is_predict else self._train_overlap_buffer
        if buf > 0:
            ds_kwargs["predict_overlap_buffer"] = buf

        try:
            self.dataset = self.dataset_class(**ds_kwargs)
        except TypeError:
            # Fallback: dataset class doesn't accept predict_overlap_buffer
            ds_kwargs.pop("predict_overlap_buffer", None)
            self.dataset = self.dataset_class(**ds_kwargs)

        if not is_predict:
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

    # ------------------------------------------------------------------
    # Burned-tile oversampling (data-level class rebalancing, TRAIN only)
    # ------------------------------------------------------------------

    def _get_train_sample_weights(self) -> list[float] | None:
        """Return (and cache) per-sample sampling weights for the train split.

        Weights are aligned with the ordering of ``self.train_dataset`` (a
        :class:`~torch.utils.data.Subset`), because :class:`WeightedRandomSampler`
        draws positions ``0 … len(subset)-1`` of that subset — not indices into
        the underlying dataset.
        """
        if self._train_sample_weights is not None:
            return self._train_sample_weights
        if self.train_dataset is None:
            return None

        indices = getattr(self.train_dataset, "indices", None)
        if indices is None:
            indices = list(range(len(self.train_dataset)))

        weights: list[float] = []
        mask_cache: dict[str, np.ndarray | None] = {}
        n_burned = 0
        for idx in indices:
            entry = self.dataset.files[idx]
            ratio = self._tile_burned_ratio(entry, mask_cache)
            is_burned = ratio > self.burned_tile_min_ratio
            if is_burned:
                n_burned += 1
            weights.append(self.burned_oversample_factor if is_burned else 1.0)

        logger.info(
            "Burned-tile oversampling: %d/%d train tiles flagged burned "
            "(factor=%.2f, min_ratio=%.3f)",
            n_burned, len(weights), self.burned_oversample_factor,
            self.burned_tile_min_ratio,
        )
        if n_burned == 0:
            logger.warning(
                "No burned tiles detected — disabling oversampling (check "
                "mask paths / burned_tile_min_ratio)."
            )
            self._train_sample_weights = None
            return None

        self._train_sample_weights = weights
        return weights

    def _tile_burned_ratio(
            self,
            entry: dict[str, Any],
            mask_cache: dict[str, np.ndarray | None],
    ) -> float:
        """Fraction of burned pixels (value == 1) in a tile's valid central-cell zone.

        The on-disk mask covers the *cell* (no buffer).  When a spatial-context
        buffer is active, tile coordinates live in the expanded
        ``cell + 2*buffer`` frame, so we intersect the tile footprint with the
        central-cell region ``[buf, buf + cell_size)`` before counting.
        """
        mask_path = entry.get("mask")
        if mask_path is None:
            return 0.0

        key = str(mask_path)
        arr = mask_cache.get(key, "MISS")
        if isinstance(arr, str):  # not yet cached
            try:
                with rio.open(key) as src:
                    arr = src.read(1)
            except Exception:  # noqa: BLE001
                logger.debug("Could not read mask for weighting: %s", key)
                arr = None
            mask_cache[key] = arr
        if arr is None:
            return 0.0

        cell_h, cell_w = arr.shape
        buf = self._train_overlap_buffer

        if "_tile_row" in entry:
            r, c = entry["_tile_row"], entry["_tile_col"]
            th, tw = self.tile_size
            # Intersect tile [r, r+th) × [c, c+tw) with cell [buf, buf+cell)
            r0 = max(r, buf) - buf
            r1 = min(r + th, buf + cell_h) - buf
            c0 = max(c, buf) - buf
            c1 = min(c + tw, buf + cell_w) - buf
            if r1 <= r0 or c1 <= c0:
                return 0.0
            sub = arr[r0:r1, c0:c1]
        else:
            sub = arr

        total = sub.size
        if total == 0:
            return 0.0
        return float(np.count_nonzero(sub == 1)) / float(total)



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

        sampler = None
        shuffle = True
        # Data-level class rebalancing: oversample burned tiles on the TRAIN
        # split only (val/test untouched → metrics stay comparable).
        if self.burned_oversample_factor and self.burned_oversample_factor > 1.0:
            weights = self._get_train_sample_weights()
            if weights is not None:
                sampler = WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True,
                )
                shuffle = False  # mutually exclusive with a sampler

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=shuffle,
            sampler=sampler,
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