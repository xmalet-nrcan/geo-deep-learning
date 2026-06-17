import logging
from pathlib import Path
from typing import Optional, List, Any

from geo_deep_learning.datasets.rcm_change_detection_dataset import RCMChangeDetectionDataset, SatellitePass

logger = logging.getLogger("RCM-PrePost RCMChangeDetectionOnPredictDataset")
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s - %(name)s - [%(levelname)s] ] - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

class RCMChangeDetectionOnPredictDataset(RCMChangeDetectionDataset):
    def __init__(self, csv_root_folder: str, patches_root_folder: str, split_or_csv_file_name: str = None,
                 norm_stats: dict[str, list[float]] | None = None, bands: Optional[List[int]] = None,
                 band_names: Optional[List[str]] = None, satellite_pass: Optional[str | SatellitePass] = None,
                 beams: Optional[List[str]] = None,
                 dataset_years: Optional[list[int]] = None,
                 separate_metadata: bool = True,
                 ) -> None:
        super().__init__(csv_root_folder, patches_root_folder, split_or_csv_file_name, norm_stats, bands, band_names,
                         satellite_pass, beams, dataset_years, separate_metadata=separate_metadata)

    def _load_files(self) -> list[dict[str, str]]:
        df_csv = self._get_input_dataset_as_dataframe()

        files = []
        for (img_pre,
             img,
             pair_id,
             group_id_pre,
             group_id_post,
             event_id,
             cell_id,
             group_date_pre,
             group_date_post,
             beam,
             sat_pass,
             event_start_date) in df_csv[
            ['pre_input_file',
             'post_input_file',
             "pair_id",
             'group_id_pre',
             'group_id_post',
             'event_id',
             'cell_id',
             'group_date_pre',
             'group_date_post',
             'beam',
             'sat_pass',
             'event_start_date']
        ].itertuples(index=False):
            img_pre_path = img_pre.replace("$ROOT_PATH", self.patches_root_folder).strip()
            img_post_path = img.replace("$ROOT_PATH", self.patches_root_folder).strip()
            if Path(img_pre_path).exists() and Path(img_post_path).exists():
                files.append({
                    "image_pre": img_pre.replace("$ROOT_PATH", self.patches_root_folder).strip(),
                    "image": img.replace("$ROOT_PATH", self.patches_root_folder).strip(),
                    # "mask": self._get_mask_path(cell_id, group_date_post),
                    "water_mask": self._get_water_mask_path(cell_id),
                    "pair_id": pair_id,
                    "cell_id": cell_id,
                    "event_id": event_id,
                    "group_date_pre": group_date_pre,
                    "group_date_post": group_date_post,
                    "beam": beam,
                    "sat_pass": sat_pass,
                    "group_id_pre": group_id_pre,
                    "group_id_post": group_id_post,
                    "event_start_date": event_start_date,
                })

        logger.info(
            f"Loaded {len(files)} files for {len(df_csv)} rows. Nb of event : {len(df_csv['event_id'].unique())}")

        return files

    def _get_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """Metadata specific to the predict CSV schema."""
        return {
            "pair_id": data["pair_id"],
            "event_id": data["event_id"],
            "event_start_date": data["event_start_date"],
            "group_id_pre": data["group_id_pre"],
            "group_id_post": data["group_id_post"],
            "group_date_pre": data["group_date_pre"],
            "group_date_post": data["group_date_post"],
        }

    def _get_pre_post_name(self, data: dict[str, str]) -> str:
        """Build identifier adapted to predict CSV columns."""
        return (
            f"{data['cell_id']}_event_{data['event_id']}_pair_{data['pair_id']}|"
            f"{'ASC' if data['sat_pass'] == SatellitePass.ASCENDING else 'DESC'}-{data['beam'].name}|"
            f"({data['group_id_pre']}){data['group_date_pre']}_"
            f"({data['group_id_post']}){data['group_date_post']}|"
            f"event_({data['event_id']})_{data['event_start_date']}"
        )

    def __getitem__(self, index: int) -> dict:
        return super().__getitem__(index)
