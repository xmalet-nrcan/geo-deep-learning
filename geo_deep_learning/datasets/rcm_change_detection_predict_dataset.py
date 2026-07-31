"""Predict-time dataset for RCM change detection.

Differs from the training dataset only in the CSV schema (predict columns:
pair_id, pre_input_file, event_id, …) and the metadata returned.
All spatial-context buffer and tiling logic is inherited from the base class.
"""
import logging
from pathlib import Path
from typing import Optional, List, Any

from geo_deep_learning.datasets.rcm_change_detection_dataset import (
    RCMChangeDetectionDataset,
    SatellitePass,
    Beams,
)

logger = logging.getLogger("RCM-PrePost RCMChangeDetectionOnPredictDataset")
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s - %(name)s - [%(levelname)s] ] - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

class RCMChangeDetectionOnPredictDataset(RCMChangeDetectionDataset):
    """Predict-time dataset with optional spatial context from neighbouring grid cells.

    Inherits all buffer and tiling logic from :class:`RCMChangeDetectionDataset`.
    Differs only in the CSV schema used by :meth:`_build_raw_file_list` and in
    the metadata keys returned by :meth:`_get_metadata`.
    """

    def __init__(
        self,
        csv_root_folder: str,
        patches_root_folder: str,
        split_or_csv_file_name: str = None,
        norm_stats: dict[str, list[float]] | None = None,
        bands: Optional[List[int]] = None,
        band_names: Optional[List[str]] = None,
        satellite_pass: Optional[str | SatellitePass] = None,
        beams: Optional[List[str]] = None,
        dataset_years: Optional[list[int]] = None,
        separate_metadata: bool = True,
        tile_size: tuple[int, int] | None = None,
        tile_stride: tuple[int, int] | None = None,
        predict_overlap_buffer: int = 0,
    ) -> None:
        super().__init__(
            csv_root_folder, patches_root_folder, split_or_csv_file_name,
            norm_stats, bands, band_names, satellite_pass, beams,
            dataset_years, separate_metadata=separate_metadata,
            tile_size=tile_size, tile_stride=tile_stride,
            predict_overlap_buffer=predict_overlap_buffer,
        )

    # ------------------------------------------------------------------
    # Predict CSV schema
    # ------------------------------------------------------------------

    def _build_raw_file_list(self) -> list[dict[str, str]]:
        """Load the raw per-sample file dicts from the predict CSV schema.

        Predict CSV columns differ from the training CSV:
        ``pre_input_file`` / ``post_input_file`` instead of ``pre_path`` /
        ``post_path``; ``pair_id`` / ``event_id`` instead of
        ``db_nbac_fire_id``; no ``mask`` column.
        """
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
                    "image_pre": img_pre_path,
                    "image": img_post_path,
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
            f"Loaded {len(files)} files for {len(df_csv)} rows. "
            f"Nb of event : {len(df_csv['event_id'].unique())}"
        )

        return files

    # ------------------------------------------------------------------
    # Metadata helpers (predict-specific)
    # ------------------------------------------------------------------

    def _get_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """Metadata specific to the predict CSV schema."""
        return {
            "pair_id": int(data["pair_id"]) if data["pair_id"] is not None else -1,
            "event_id": int(data["event_id"]) if data["event_id"] is not None else -1,
            "event_start_date": str(data["event_start_date"]) if data["event_start_date"] is not None else "",
            "group_id_pre": int(data["group_id_pre"]) if data["group_id_pre"] is not None else -1,
            "group_id_post": int(data["group_id_post"]) if data["group_id_post"] is not None else -1,
            "group_date_pre": str(data["group_date_pre"]) if data["group_date_pre"] is not None else "",
            "group_date_post": str(data["group_date_post"]) if data["group_date_post"] is not None else "",
        }

    def _get_pre_post_name(self, data: dict[str, str]) -> str:
        """Build identifier adapted to predict CSV columns."""
        beam = data['beam']
        beam_name = beam.name if isinstance(beam, Beams) else str(beam)
        return (
            f"{data['cell_id']}_event_{data['event_id']}_pair_{data['pair_id']}|"
            f"{'ASC' if data['sat_pass'] == SatellitePass.ASCENDING else 'DESC'}-{beam_name}|"
            f"({data['group_id_pre']}){data['group_date_pre']}_"
            f"({data['group_id_post']}){data['group_date_post']}|"
            f"event_({data['event_id']})_{data['event_start_date']}"
        )