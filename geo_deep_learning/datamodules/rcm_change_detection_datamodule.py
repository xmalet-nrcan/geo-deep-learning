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

# bands_stats = {
#     'mean': [1.0088686544882763, 22.678325648034726, 4820.030168929148, -578.1138439754548, 174.35119966169816,
#              4645.179761547494, 5178.970253993203, 4074.12440505587, 1427.3155618129722, 517.5479435073069,
#              1945.2480656061873, 514.8092047489475, 425.98675130681056, 8939.542957169055],
#     'std': [0.17514777918322952, 4.602293040200134, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
#             np.nan, np.nan, np.nan,
#             np.nan],
#     'min': [1.0, 0.0, 446.0, -9810.0, 0.0, 81.0, 14.0, 93.0, 2.0, 2.0, 5.0, -9340.0, -9584.0, -8947.0],
#     'max': [9.0, 112.0, 9985.0, 9969.0, 6358.0, 9971.0, 9553.0, 9979.0, 32766.0, 32766.0, 32766.0, 9484.0,
#             9901.0, 9999.0]
# }

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
            mean: list[float] | None = None,
            std: list[float] | None = None,
            bands: Optional[List[int]] = None,
            band_names: Optional[List[str]] = None,
            satellite_pass: Optional[str] = None,
            beams: Optional[List[str]] = None,
            split_ratios=(0.70, 0.15, 0.15),
            split_on_columns: Optional[str | list] = None,
            data_type_max: Optional[int] = None,
            dataset_class: Type[RCMChangeDetectionDataset] = RCMChangeDetectionDataset,

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
        self.dataset_class = dataset_class

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
            beams=self.beams
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
        split_on_columns=['db_nbac_fire_id'],
    split_ratios=(0.8, 0.1, 0.1))
    dataset.setup()

    tdl = dataset.train_dataset
    val = dataset.val_dataset
    test = dataset.test_dataset

    print(f"Final split counts: "
          f"train={len(tdl)} ({len(tdl) / len(dataset.dataset.files):.2%}), "
          f"val={len(val)} ({len(val) / len(dataset.dataset.files):.2%}), "
          f"test={len(test)} ({len(test) / len(dataset.dataset.files):.2%})")

    print("cells, fires, group_pre, group_post")

    for n, d in (['train', tdl], ['val', val], ['test', test]):
        cells, fires, group_pre, group_post = set(), set(), set(), set()
        print("creating for ", n)
        for i in d:
            cells.add(i['cell_id'])
            fires.add(i['db_nbac_fire_id'])
            group_pre.add(i['group_id_pre'])
            group_post.add(i['group_id_post'])
        print(f"{len(cells)}, {len(fires)}, {len(group_pre)}, {len(group_post)}")
        with open(f'C:\\Users\\xmalet\\PycharmProjects\\geo-deep-learning\\data\\{n}_cells.txt', 'w') as f:
            f.write('"cell_id" in (')
            for c in cells:
                f.write(f"'{str(c)}'" + ', ')
            f.write(")")
        with open(f'C:\\Users\\xmalet\\PycharmProjects\\geo-deep-learning\\data\\{n}_fires.txt', 'w') as f:
            f.write('"db_nbac_fire_id" in (')
            for c in fires:
                f.write(str(c) + ', ')
            f.write(')')
        with open(f'C:\\Users\\xmalet\\PycharmProjects\\geo-deep-learning\\data\\{n}_group_pre.txt', 'w') as f:
            f.write('"group_id" in (')
            for c in group_pre:
                f.write(f"{str(c)}" + ', ')
            f.write(')')
        with open(f'C:\\Users\\xmalet\\PycharmProjects\\geo-deep-learning\\data\\{n}_group_post.txt', 'w') as f:
            f.write('"group_id" in (')
            for c in group_post:
                f.write(f"{str(c)}" + ', ')
            f.write(')')
