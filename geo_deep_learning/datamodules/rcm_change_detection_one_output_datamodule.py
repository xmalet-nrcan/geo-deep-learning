from geo_deep_learning.datamodules.rcm_change_detection_datamodule import RcmChangeDetectionDataModule
from geo_deep_learning.datasets.rcm_change_detection_dataset_one_output import RCMChangeDetectionDatasetOneOutput


class RcmChangeDetectionOneOutputDataModule(RcmChangeDetectionDataModule):
    def setup(self, stage: str | None = None) -> None:
        self.dataset = RCMChangeDetectionDatasetOneOutput(
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
