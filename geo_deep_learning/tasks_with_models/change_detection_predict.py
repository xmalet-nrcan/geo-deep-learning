"""Predict-time behaviour of the change-detection task.

:class:`PredictOutputMixin` holds everything that happens after training:
``predict_step`` (with TTA), per-tile / overlap-blended GeoTIFF writing, the
run manifest and the cross-cell merges. It is mixed into
:class:`~geo_deep_learning.tasks_with_models.change_detection_changeformer.ChangeDetectionChangeFormer`
and relies on these host attributes:

* ``change_detection_model``, ``weights_from_checkpoint_path``
* ``predict_output_dir``, ``predict_output_layout``, ``predict_run_name``
* ``predict_write_merged_all``, ``merged_all_min_pairs``, ``merged_all_prob_threshold``
* ``probability_zone_thresholds``, ``probability_zone_class_index``
* ``forward`` (called by :meth:`_tta_forward`) and ``self.trainer``
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
import torch
from rasterio.transform import Affine
from torch import Tensor

from geo_deep_learning.datasets.rcm_change_detection_dataset import NO_DATA
from geo_deep_learning.tools.losses.change_detection_loss import MASK_VALID_THRESHOLD
from geo_deep_learning.utils.geotiff_merge import (
    extract_scalar,
    group_merge_key,
    merge_predictions,
    parse_crs,
    prediction_output_filename,
    transform_coeffs,
)
from geo_deep_learning.utils.tile_reassembly import reassemble_overlapping_tiles
from geo_deep_learning.utils.tiling import buffer_valid_region, scalar_at

logger = logging.getLogger(__name__)

# Default ascending probability thresholds defining exclusive confidence bands
# for probability-zone vectorization (see ``predict_step``/``_new_manifest``).
# Pixels below the first threshold (0.3) are left unclassified/unzoned.
DEFAULT_PROBABILITY_ZONE_THRESHOLDS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Sentinel written to probability GeoTIFFs for invalid/masked pixels.
# Probabilities live in [0, 1], so a negative value is an unambiguous nodata.
# It MUST stay below the first probability-zone threshold (and equal to scanfire's
# ``vectorize.PROBABILITY_NODATA``): a larger value such as ``NO_DATA`` (32767) would
# be binned in the top confidence band, vectorizing masked pixels (water, invalid).
PROBABILITY_NODATA = -1.0

# Predict output directory layouts (see ``predict_output_layout`` in ``__init__``).
PREDICT_LAYOUT_PRODUCTION = "production"
PREDICT_LAYOUT_BENCHMARK = "benchmark"
PREDICT_OUTPUT_LAYOUTS = (PREDICT_LAYOUT_PRODUCTION, PREDICT_LAYOUT_BENCHMARK)

# Benchmark layout: per-run manifests live under ``<dir>/_runs/<predict_date>/``.
BENCHMARK_RUNS_DIRNAME = "_runs"
BENCHMARK_LATEST_MANIFEST = "manifest_latest.json"


class PredictOutputMixin:
    """``predict_step`` + GeoTIFF / manifest writing for the change-detection task."""

    def predict_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        """Run prediction step with optional Test-Time Augmentation (TTA).

        TTA applies 4 self-inverse geometric transforms (identity, horizontal
        flip, vertical flip, and both flips i.e. a 180° rotation), runs
        inference on each, inverts the transform, and averages the softmax
        probabilities. This reduces noise-related false positives —
        particularly important for SAR data where speckle can cause spurious
        detections. Note: unlike the training augmentations, 90°/270°
        rotations are *not* included here (see :meth:`_tta_forward`).
        """
        x_pre = batch["image_pre"]
        x_post = batch["image"]

        with torch.no_grad():
            logits = self._tta_forward(
                x_pre, x_post,
                sat_pass=batch.get("sat_pass_value"),
                beam=batch.get("beam_value"),
                pre_season=batch.get("pre_season"),
                post_season=batch.get("post_season"),
                time_delta=batch.get("time_delta_bin"),
                processing_year=batch.get("processing_year"),
            )

        # Convertir en probabilités et en classes prédites (binaire: 2 classes
        # 0=no-change/1=change, ou multiclasse: identique via softmax+argmax)
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        y_pred = torch.argmax(probs, dim=1)  # [B, H, W]

        # Positive-class probability surface, used downstream to vectorize
        # probability zones (exclusive confidence bands) rather than a single
        # binary mask — see ``probability_zone_thresholds``/``_new_manifest``.
        y_prob = probs[:, self.probability_zone_class_index, :, :].clone()

        # --- Exclure les pixels invalides et l'eau avec NO_DATA (32767) ---
        # ``mask-common`` et ``water_mask`` sont appliqués, voir :meth:`_invalid_pixel_mask`.
        invalid_mask = self._invalid_pixel_mask(batch, like=y_pred)
        y_pred = y_pred.masked_fill(invalid_mask, NO_DATA)
        y_prob = y_prob.masked_fill(invalid_mask, PROBABILITY_NODATA)

        # Retourner un dict avec tout ce qu'il faut pour sauvegarder après
        result = {
            "predictions": y_pred,  # [B, H, W] classes entières
            "probability": y_prob,  # [B, H, W] proba classe positive, pour zonage
            "probabilities": probs,  # [B, C, H, W] probabilités par classe
            "logits": logits,  # [B, C, H, W] logits bruts
            "pre_post_name": batch["pre_post_name"],
            "cell_id": batch["cell_id"],
            "profile": batch["profile"],  # profil rasterio pour écriture GeoTIFF
            "original_height": batch["original_height"],
            "original_width": batch["original_width"],
        }

        # --- Propager le masque commun pour le re-masquage après blending overlap ---
        if "mask-common" in batch:
            result["mask_common"] = batch["mask-common"]
        if "water_mask" in batch:
            result["water_mask"] = batch["water_mask"]


        # --- Propager les métadonnées optionnelles (event_id, db_nbac_fire_id, etc.) ---
        for key in ("pair_id",
                    "event_id",
                    "db_nbac_fire_id",
                    "event_start_date",
                    "event_end_date",
                    "beam",
                    "sat_pass",
                    "output_name",
                    'group_date_pre',
                    'group_date_post',
                    'group_id_pre',
                    'group_id_post',
                    'tile_row_start',
                    'tile_col_start',
                    'source_height',
                    'source_width',
                    'buffer_size',
                    'cell_orig_height',
                    'cell_orig_width'):
            if key in batch:
                result[key] = batch[key]

        if batch.get("has_mask", torch.tensor(False)).any():
            result["mask"] = batch["mask"]

        return result

    @staticmethod
    def _invalid_pixel_mask(batch: dict[str, Any], *, like: Tensor) -> Tensor:
        """Boolean ``[B, H, W]`` mask of pixels whose prediction must be ``NO_DATA``.

        Union of the per-pixel layers carried by the dataset samples:

        - ``mask-common``: SAR acquisition validity (``< MASK_VALID_THRESHOLD`` → invalid);
        - ``water_mask``: static water raster (``> 0`` → water).

        Each layer is applied independently, so water can never be vectorized as
        burn even when ``mask-common`` does not contain it.
        """
        invalid = torch.zeros_like(like, dtype=torch.bool)
        if "mask-common" in batch:
            invalid |= batch["mask-common"].squeeze(1) < MASK_VALID_THRESHOLD  # [B, 1, H, W] bool ou float
        if "water_mask" in batch:
            invalid |= batch["water_mask"].squeeze(1) > 0  # eau = valeur > 0
        return invalid

    def _tta_forward(
            self,
            x_pre: Tensor,
            x_post: Tensor,
            sat_pass: Tensor | None = None,
            beam: Tensor | None = None,
            **metadata_kwargs: Tensor,
    ) -> Tensor:
        """Average predictions over geometric transformations."""
        transforms = [
            lambda t: t,
            lambda t: torch.flip(t, dims=(-1,)),
            lambda t: torch.flip(t, dims=(-2,)),
            lambda t: torch.flip(t, dims=(-2, -1)),
        ]

        probs_sum: Tensor | None = None

        for transform in transforms:
            aug_pre = transform(x_pre)
            aug_post = transform(x_post)

            # This must call the forward path without TTA.
            out = self(
                aug_pre,
                aug_post,
                sat_pass=sat_pass,
                beam=beam,
                **metadata_kwargs,
            )

            if isinstance(out, (list, tuple)):
                out = out[-1]

            if out.shape[1] == 1:
                probabilities = torch.sigmoid(out)
            else:
                probabilities = torch.softmax(out, dim=1)

            # These transformations are their own inverse.
            probabilities = transform(probabilities)

            probs_sum = (
                probabilities
                if probs_sum is None
                else probs_sum + probabilities
            )

        avg_probs = probs_sum / len(transforms)
        eps = 1e-7

        if avg_probs.shape[1] == 1:
            return torch.logit(avg_probs.clamp(eps, 1.0 - eps))

        return torch.log(avg_probs.clamp_min(eps))

    # ------------------------------------------------------------------
    # Saving predictions (GeoTIFF + manifest + cross-tile merging)
    #
    # Overlap-blended tile reassembly (``create_blend_window``,
    # ``build_source_profile``, ``reassemble_overlapping_tiles``) and the
    # generic GeoTIFF-merging routines (``safe_merge``, ``chunked_merge``,
    # ``merge_predictions``, filename helpers, …) live in
    # ``geo_deep_learning.utils.geotiff_merge`` /
    # ``geo_deep_learning.utils.tile_reassembly`` — imported at the top of
    # this module — since they are pure, reusable I/O logic with no
    # dependency on the LightningModule itself.
    # ------------------------------------------------------------------

    def _new_manifest(self, base_dir: Path, predict_date: str, *, overlap_blended: bool = False) -> dict[str, Any]:
        """Create the skeleton of the prediction manifest written alongside GeoTIFFs."""
        manifest: dict[str, Any] = {
            "prediction_date": predict_date,
            "model_name": self.change_detection_model,
            "checkpoint": str(self.weights_from_checkpoint_path or ""),
            "base_dir": str(base_dir),
            # Thresholds used to define exclusive probability-zone bands, consumed
            # by scanfire's ``vectorize_probability_zones``/``SegmentationIngestionService``
            # when a ``probability_tif_path`` is present in a prediction entry.
            "probability_thresholds": self.probability_zone_thresholds,
            "predictions": [],
        }
        if overlap_blended:
            manifest["overlap_blended"] = True
        if self.predict_output_layout != PREDICT_LAYOUT_PRODUCTION:
            manifest["output_layout"] = self.predict_output_layout
        if self.predict_run_name:
            manifest["run_name"] = self.predict_run_name
        return manifest

    @staticmethod
    def _write_manifest_file(manifest: dict[str, Any], base_dir: Path) -> None:
        manifest_path = base_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info("Saved prediction manifest to %s", manifest_path)

    def _write_manifests(self, manifest: dict[str, Any], base_dir: Path, predict_date: str) -> None:
        """Write the run manifest according to :attr:`predict_output_layout`.

        - ``production``: ``<base_dir>/manifest.json`` (unchanged, read by scanfire).
        - ``benchmark``: ``<base_dir>/_runs/<predict_date>/manifest.json`` (history,
          one per run) + a copy at ``<base_dir>/manifest_latest.json``.
        """
        if self.predict_output_layout == PREDICT_LAYOUT_PRODUCTION:
            self._write_manifest_file(manifest, base_dir)
            return

        run_dir = base_dir / BENCHMARK_RUNS_DIRNAME / predict_date
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest_file(manifest, run_dir)
        latest_path = base_dir / BENCHMARK_LATEST_MANIFEST
        with open(latest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info("Saved latest benchmark manifest to %s", latest_path)

    def _resolve_predict_base_dir(self) -> Path:
        """Root directory of predict outputs, according to :attr:`predict_output_layout`.

        - ``production``: ``predict_output_dir`` (or ``trainer.default_root_dir``)
          with a ``predictions`` sub-folder appended when missing (historical
          behaviour, expected by scanfire).
        - ``benchmark``: ``predict_output_dir`` used as-is (typically
          ``<benchmark_output_root>/<model_name>``); falls back to
          ``<default_root_dir>/benchmark/<predict_run_name or change_detection_model>``.
        """
        if self.predict_output_layout == PREDICT_LAYOUT_PRODUCTION:
            if self.predict_output_dir is not None:
                base_dir = Path(self.predict_output_dir)
                if base_dir.name != "predictions":
                    base_dir = base_dir / "predictions"
                return base_dir
            return Path(self.trainer.default_root_dir) / "predictions"

        if self.predict_output_dir is not None:
            return Path(self.predict_output_dir)
        run_name = self.predict_run_name or str(self.change_detection_model)
        return Path(self.trainer.default_root_dir) / "benchmark" / run_name

    def _prediction_dirs(
            self,
            base_dir: Path,
            event_id: str,
            predict_date: str,
            cell_id: str,
    ) -> tuple[Path, Path]:
        """Return ``(tile_dir, merge_dir)`` for one prediction.

        ``tile_dir`` receives the per-cell GeoTIFF (+ ``_prob``); ``merge_dir``
        receives the cross-cell merges (per pre/post pair, ``merged_all.tif``).

        - ``production``: ``<base>/<event_id>/<predict_date>/<cell_id>`` and
          ``<base>/<event_id>/<predict_date>``.
        - ``benchmark``: both ``<base>/<event_id>`` (cell-level names already
          contain ``_cell-<id>``; merged names do not → no collision).
        """
        if self.predict_output_layout == PREDICT_LAYOUT_PRODUCTION:
            merge_dir = base_dir / str(event_id) / predict_date
            return merge_dir / str(cell_id), merge_dir
        event_dir = base_dir / str(event_id)
        return event_dir, event_dir

    @staticmethod
    def _write_single_geotiff_prediction(  # noqa: PLR0913
            *,
            pred_np: np.ndarray,
            profile: dict[str, Any],
            tile_dir: Path,
            merge_dir: Path,
            cell_id: str,
            pair_id: str | None,
            event_id: str,
            event_start_date: str | None,
            event_end_date: str | None,
            beam: str | None,
            sat_pass: str | None,
            group_id_pre: str,
            group_id_post: str,
            group_date_pre: str,
            group_date_post: str,
            output_name: str | None,
            legacy_name: str,
            manifest: dict[str, Any],
            group_tile_paths: dict[tuple[str, ...], list[Path]],
            event_all_tile_paths: dict[str, list[Path]],
            overlap_blended: bool = False,
            prob_np: np.ndarray | None = None,
    ) -> Path:
        """Write one prediction GeoTIFF, register it for merging, and append its manifest entry.

        Shared by both the per-tile path (:meth:`_write_prediction_batch`)
        and the overlap-blended path (:meth:`_save_assembled_predictions`)
        in :meth:`on_predict_end`, which used to duplicate this logic
        (~100 lines) — including a less defensive affine-transform parser
        in the per-tile path (see
        :func:`geo_deep_learning.utils.geotiff_merge.transform_coeffs`).

        When ``prob_np`` is provided, a sibling single-band ``float32``
        probability GeoTIFF (positive/burn-class probability, nodata
        :data:`PROBABILITY_NODATA`) is written next to the class raster and
        referenced in the manifest entry as ``probability_tif_path``, for
        downstream probability-zone vectorization.

        ``tile_dir`` / ``merge_dir`` come from :meth:`_prediction_dirs` (layout
        dependent): the raster is written to ``tile_dir`` and registered for the
        cross-cell merges written to ``merge_dir``.
        """
        tile_dir.mkdir(parents=True, exist_ok=True)

        out_name = prediction_output_filename(output_name, pair_id=pair_id, legacy_name=legacy_name)
        out_path = tile_dir / out_name

        with rio.open(str(out_path), "w", **profile) as dst:
            dst.write(pred_np[np.newaxis, :, :])

        logger.info("Saved prediction to %s (%dx%d)", out_path, pred_np.shape[1], pred_np.shape[0])

        prob_path: Path | None = None
        if prob_np is not None:
            prob_path = out_path.with_name(f"{out_path.stem}_prob{out_path.suffix}")
            prob_profile = {**profile, "dtype": "float32", "nodata": PROBABILITY_NODATA, "count": 1}
            with rio.open(str(prob_path), "w", **prob_profile) as dst:
                dst.write(prob_np[np.newaxis, :, :].astype(np.float32))
            logger.info("Saved probability raster to %s", prob_path)

        event_date_key = str(merge_dir)
        merge_key = group_merge_key(
            event_date_key, event_id, event_start_date, event_end_date,
            group_id_pre, group_date_pre, group_id_post, group_date_post,
            beam, sat_pass,
        )
        group_tile_paths[merge_key].append(out_path)
        event_all_tile_paths[event_date_key].append(out_path)

        entry = {
            "pair_id": pair_id,
            "event_id": event_id,
            "event_start_date": event_start_date,
            "event_end_date": event_end_date,
            "cell_id": cell_id,
            "beam": beam,
            "sat_pass": sat_pass,
            "group_id_pre": group_id_pre,
            "group_id_post": group_id_post,
            "group_date_pre": group_date_pre,
            "group_date_post": group_date_post,
            "output_name": out_name,
            "tif_path": str(out_path),
        }
        if prob_path is not None:
            entry["probability_tif_path"] = str(prob_path)
        if overlap_blended:
            entry["overlap_blended"] = True
        manifest["predictions"].append(entry)
        return out_path

    def _merge_predictions(
            self,
            group_tile_paths: dict[tuple[str, ...], list[Path]],
            event_all_tile_paths: dict[str, list[Path]],
    ) -> None:
        """Run the cross-cell merges with this module's ``merged_all`` settings."""
        merge_predictions(
            group_tile_paths, event_all_tile_paths,
            write_merged_all=self.predict_write_merged_all,
            merged_all_min_pairs=self.merged_all_min_pairs,
            merged_all_prob_threshold=self.merged_all_prob_threshold,
        )

    def _save_assembled_predictions(
            self,
            assembled: dict[str, dict[str, Any]],
            base_dir: Path,
            predict_date: str,
    ) -> None:
        """Save overlap-blended source-level predictions as GeoTIFFs and merge.

        Mirrors the structure of the per-tile path in :meth:`on_predict_end`
        (via the shared :meth:`_write_single_geotiff_prediction` helper):
        individual GeoTIFFs → manifest JSON → group merge → global merge.
        """
        group_tile_paths: dict[tuple[str, ...], list[Path]] = defaultdict(list)
        event_all_tile_paths: dict[str, list[Path]] = defaultdict(list)
        manifest = self._new_manifest(base_dir, predict_date, overlap_blended=True)

        for source_key, info in assembled.items():
            safe_name = Path(source_key.replace("|", "_").replace("/", "_")).stem
            cell_id = str(info["cell_id"])
            event_id = str(info.get("event_id", "unknown_event"))
            tile_dir, merge_dir = self._prediction_dirs(base_dir, event_id, predict_date, cell_id)
            self._write_single_geotiff_prediction(
                pred_np=info["predictions"],  # [H, W] uint16
                profile=info["profile"],
                tile_dir=tile_dir,
                merge_dir=merge_dir,
                cell_id=cell_id,
                pair_id=info.get("pair_id"),
                event_id=event_id,
                event_start_date=info.get("event_start_date"),
                event_end_date=info.get("event_end_date"),
                beam=info.get("beam"),
                sat_pass=info.get("sat_pass"),
                group_id_pre=str(info.get("group_id_pre", "all")),
                group_id_post=str(info.get("group_id_post", "all")),
                group_date_pre=str(info.get("group_date_pre", "all")),
                group_date_post=str(info.get("group_date_post", "all")),
                output_name=info.get("output_name"),
                legacy_name=safe_name,
                manifest=manifest,
                group_tile_paths=group_tile_paths,
                event_all_tile_paths=event_all_tile_paths,
                overlap_blended=True,
                prob_np=info.get("probability"),
            )

        self._write_manifests(manifest, base_dir, predict_date)
        # Merge across cells / groups (same logic as per-tile path)
        self._merge_predictions(group_tile_paths, event_all_tile_paths)
        logger.info("All blended predictions saved to %s", base_dir)

    def on_predict_end(self) -> None:
        """Appelé après que tous les predict_step soient terminés.

        Structure de sortie (``predict_output_layout``) :

        - ``production`` (défaut, consommée par scanfire) ::

            output_dir / predictions / EVENT_ID / PREDICTION_DATE / cell_id / image.tif
            output_dir / predictions / EVENT_ID / PREDICTION_DATE / merged.tif
            output_dir / predictions / manifest.json

        - ``benchmark`` (comparaison multi-modèles) ::

            output_dir / EVENT_ID / image.tif (+ _prob.tif, + fusion par paire pre/post)
            output_dir / _runs / PREDICTION_DATE / manifest.json
            output_dir / manifest_latest.json
        """
        predictions = self.trainer.predict_loop.predictions
        if not predictions:
            logger.warning("No predictions to save.")
            return

        # --- Base output directory ---
        predict_date = datetime.now().strftime("%Y%m%d_%H%M")
        base_dir = self._resolve_predict_base_dir()

        base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving predictions to %s (layout=%s)", base_dir, self.predict_output_layout)

        # --- Try overlap-based tile reassembly (cosine blending) ---
        dm = getattr(self.trainer, "datamodule", None)
        assembled, used_blending = reassemble_overlapping_tiles(
            predictions,
            getattr(dm, "tile_size", None),
            getattr(dm, "tile_stride", None),
            no_data_value=NO_DATA,
            positive_class_index=self.probability_zone_class_index,
            probability_nodata=PROBABILITY_NODATA,
        )
        if used_blending and assembled:
            logger.info("Using overlap blending for %d source images.", len(assembled))
            self._save_assembled_predictions(assembled, base_dir, predict_date)
            return

        # --- Phase 1 : écrire chaque tuile individuelle ---
        # On collecte les chemins par (event_id, predict_date) pour le merge
        group_tile_paths: dict[tuple[str, ...], list[Path]] = defaultdict(list)
        event_all_tile_paths: dict[str, list[Path]] = defaultdict(list)
        manifest = self._new_manifest(base_dir, predict_date)

        for batch_result in predictions:
            self._write_prediction_batch(
                batch_result, base_dir, predict_date, manifest, group_tile_paths, event_all_tile_paths,
            )

        self._write_manifests(manifest, base_dir, predict_date)
        self._merge_predictions(group_tile_paths, event_all_tile_paths)
        logger.info("All predictions saved to %s", base_dir)

    def _crop_prediction_to_cell(
            self,
            pred_np: np.ndarray,
            transform: Affine,
            batch_result: dict[str, Any],
            tile_h: int,
            tile_w: int,
            index: int,
    ) -> tuple[np.ndarray | None, Affine, int, int]:
        """Crop the neighbour-context buffer ring off a saved prediction tile.

        Mirrors the training-time masking in :meth:`_mask_buffer_zone`: without
        this, pixels computed from a *neighbouring* cell's territory (added
        only to give the model spatial context — see ``predict_overlap_buffer``)
        leaked into the saved GeoTIFF, overlapping the neighbour cell's own,
        independently-produced prediction once everything is mosaicked into
        ``merged.tif``. No-op when buffer metadata is absent (unbuffered runs).

        Returns ``(None, transform, 0, 0)`` when the tile lies entirely inside
        the buffer ring (owns zero true-cell pixels) — the caller should then
        skip writing that tile.
        """
        buf_raw = batch_result.get("buffer_size")
        cell_h_raw = batch_result.get("cell_orig_height")
        cell_w_raw = batch_result.get("cell_orig_width")
        if buf_raw is None or cell_h_raw is None or cell_w_raw is None:
            return pred_np, transform, tile_h, tile_w

        buf = scalar_at(buf_raw, index)
        if buf <= 0:
            return pred_np, transform, tile_h, tile_w

        oh = scalar_at(cell_h_raw, index)
        ow = scalar_at(cell_w_raw, index)
        tile_row_starts = batch_result.get("tile_row_start")
        tile_col_starts = batch_result.get("tile_col_start")
        tr = scalar_at(tile_row_starts, index) if tile_row_starts is not None else 0
        tc = scalar_at(tile_col_starts, index) if tile_col_starts is not None else 0

        row_start, row_end, col_start, col_end = buffer_valid_region(buf, tr, tc, tile_h, tile_w, oh, ow)
        if row_end <= row_start or col_end <= col_start:
            return None, transform, 0, 0

        cropped = pred_np[row_start:row_end, col_start:col_end]
        shifted = transform * Affine.translation(col_start, row_start)
        return cropped, shifted, row_end - row_start, col_end - col_start

    def _write_prediction_batch(  # noqa: PLR0913
            self,
            batch_result: dict[str, Any],
            base_dir: Path,
            predict_date: str,
            manifest: dict[str, Any],
            group_tile_paths: dict[tuple[str, ...], list[Path]],
            event_all_tile_paths: dict[str, list[Path]],
    ) -> None:
        """Write every sample of one predict batch (per-tile, non-blended path)."""
        y_pred = batch_result["predictions"]  # [B, H_padded, W_padded]
        y_prob = batch_result.get("probability")  # [B, H_padded, W_padded] float32 ou None
        names = batch_result["pre_post_name"]
        batch_cell_id = batch_result["cell_id"]
        batch_profiles = batch_result["profile"]
        orig_heights = batch_result["original_height"]  # Tensor [B] ou list
        orig_widths = batch_result["original_width"]  # Tensor [B] ou list
        batch_size = y_pred.shape[0]

        batch_pair_ids = batch_result.get("pair_id")
        batch_output_names = batch_result.get("output_name")
        # event_id : peut être un Tensor, une list, ou absent.
        # Fallback pour le training dataset qui a db_nbac_fire_id.
        batch_event_ids = batch_result.get("event_id")
        if batch_event_ids is None:
            batch_event_ids = batch_result.get("db_nbac_fire_id")
        batch_group_id_pre = batch_result.get("group_id_pre")
        batch_group_id_post = batch_result.get("group_id_post")
        batch_group_date_pre = batch_result.get("group_date_pre")
        batch_group_date_post = batch_result.get("group_date_post")
        batch_event_start_dates = batch_result.get("event_start_date")
        batch_event_end_dates = batch_result.get("event_end_date")
        batch_beams = batch_result.get("beam")
        batch_sat_passes = batch_result.get("sat_pass")

        for i in range(batch_size):
            cell_id = batch_cell_id[i]
            sample_name = names[i].replace("\n", "").replace("|", "_").replace("/", "_")

            # --- Récupérer les dimensions originales ---
            orig_h = scalar_at(orig_heights, i)
            orig_w = scalar_at(orig_widths, i)

            # --- Découper le padding (crop au coin supérieur-gauche) ---
            pred_np = y_pred[i, :orig_h, :orig_w].cpu().numpy().astype(np.uint16)

            # --- Reconstruire le profil rasterio ---
            crs_val = batch_profiles["crs"][i] if isinstance(batch_profiles["crs"], (list, tuple)) else batch_profiles["crs"]
            coeffs = transform_coeffs(batch_profiles["transform"], i)
            logger.debug("Transform coefficients: %s", coeffs)

            # --- Retirer l'anneau de contexte voisin (predict_overlap_buffer) ---
            pred_np, transform, out_h, out_w = self._crop_prediction_to_cell(
                pred_np, Affine(*coeffs), batch_result, orig_h, orig_w, i,
            )
            if pred_np is None:
                logger.debug("Tile %s owns no true-cell pixels (fully inside buffer ring); skipping.", sample_name)
                continue

            # --- Same crop applied to the probability surface, if available ---
            prob_np = None
            if y_prob is not None:
                raw_prob_np = y_prob[i, :orig_h, :orig_w].cpu().numpy().astype(np.float32)
                prob_np, _, _, _ = self._crop_prediction_to_cell(
                    raw_prob_np, Affine(*coeffs), batch_result, orig_h, orig_w, i,
                )

            profile_i = {
                "driver": "GTiff",
                "dtype": "uint16",
                "count": 1,
                "nodata": 32767,
                "height": out_h,  # ← dimensions ORIGINALES, pas paddées
                "width": out_w,  # ← dimensions ORIGINALES, pas paddées
                "crs": parse_crs(crs_val),
                "transform": transform,
            }

            event_id = extract_scalar(batch_event_ids, i, default="unknown_event")
            tile_dir, merge_dir = self._prediction_dirs(base_dir, event_id, predict_date, cell_id)
            self._write_single_geotiff_prediction(
                pred_np=pred_np,
                profile=profile_i,
                tile_dir=tile_dir,
                merge_dir=merge_dir,
                cell_id=cell_id,
                pair_id=extract_scalar(batch_pair_ids, i, default=None),
                event_id=event_id,
                event_start_date=extract_scalar(batch_event_start_dates, i, default=None),
                event_end_date=extract_scalar(batch_event_end_dates, i, default=None),
                beam=extract_scalar(batch_beams, i, default=None),
                sat_pass=extract_scalar(batch_sat_passes, i, default=None),
                group_id_pre=extract_scalar(batch_group_id_pre, i, default="all"),
                group_id_post=extract_scalar(batch_group_id_post, i, default="all"),
                group_date_pre=extract_scalar(batch_group_date_pre, i, default="all"),
                group_date_post=extract_scalar(batch_group_date_post, i, default="all"),
                output_name=extract_scalar(batch_output_names, i, default=""),
                legacy_name=f"cell-{cell_id}_{sample_name}",
                manifest=manifest,
                group_tile_paths=group_tile_paths,
                event_all_tile_paths=event_all_tile_paths,
                prob_np=prob_np,
            )
