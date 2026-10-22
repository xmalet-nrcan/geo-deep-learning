"""Validation / test figure logging for the change-detection task."""

import logging
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from torch import Tensor

from geo_deep_learning.tools.losses.change_detection_loss import (
    IGNORE_MASK_INDEX,
    MASK_VALID_THRESHOLD,
)
from geo_deep_learning.tools.visualization import visualize_prediction

logger = logging.getLogger(__name__)

# Only visualize samples with at least this fraction of valid pixels labeled
# "burned" (class 1) — keeps validation/test figures informative.
MIN_BURNED_RATIO_FOR_VISUALIZATION = 0.10


def _select_rgb_bands(num_bands: int, *, use_metadata_film: bool) -> list[int]:
    """Pick up to 3 evenly spaced data bands used as pseudo-RGB."""
    if use_metadata_film:
        # No metadata bands in image — all bands are data
        available = list(range(num_bands))
    else:
        # Legacy: skip band 0 (COMMON_MASK) and last 2 (SAT_PASS, BEAM).
        data_band_start = 1  # skip COMMON_MASK
        data_band_end = max(num_bands - 2, data_band_start + 1)  # skip SAT_PASS, BEAM
        available = list(range(data_band_start, data_band_end))
    if len(available) >= 3:  # noqa: PLR2004
        return [available[0], available[len(available) // 2], available[-1]]
    return available[:3]


def _log_figure(trainer: Trainer, fig: plt.Figure, artifact_file: str, *, epoch_suffix: bool) -> None:
    """Send ``fig`` to the trainer's MLflow / TensorBoard logger."""
    if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "log_figure"):
        trainer.logger.experiment.log_figure(
            figure=fig,
            artifact_file=artifact_file,
            run_id=getattr(trainer.logger, "run_id", None),
        )
    elif isinstance(trainer.logger, TensorBoardLogger):
        trainer.logger.experiment.add_figure(
            tag=artifact_file,
            figure=fig,
            global_step=trainer.current_epoch if epoch_suffix else 0,
        )
    else:
        logger.warning("Logger does not support figure logging.")


def log_change_detection_figures(  # noqa: PLR0913
        trainer: Trainer,
        batch: dict[str, Any],
        outputs: Tensor,
        max_samples: int,
        *,
        num_classes: int,
        changed_num_classes: int,
        class_colors: list[str] | None,
        use_metadata_film: bool,
        artifact_prefix: str = "val",
        epoch_suffix: bool = True,
) -> int:
    """Log figures comparing input diff, ground truth, and prediction.

    Generates side-by-side images of:
      - Absolute difference between post and pre images (3 selected bands)
      - Ground truth mask (if available)
      - Model prediction (with water/no-data pixels shown distinctly)

    Args:
        trainer: Lightning trainer (used for logger access)
        batch: Batch dict with keys: image, image_pre, pre_post_name, mask,
               has_mask, mask-common
        outputs: Model logits [B, C, H, W]
        max_samples: Max number of samples to visualize
        num_classes: ``num_classes`` of the task (colour map size).
        changed_num_classes: Number of output classes (index of the grey
            "invalid" colour).
        class_colors: Optional per-class colours.
        use_metadata_film: Whether metadata bands are absent from the image.
        artifact_prefix: Prefix for artifact path ("test" or "val")
        epoch_suffix: Whether to add epoch info to artifact filename

    Returns:
        Number of samples actually visualized
    """
    if batch is None or outputs is None:
        return 0

    try:
        image_batch = batch["image"]
        pre_image_batch = batch["image_pre"]
        batch_image_name = batch["pre_post_name"]
        has_mask_flags = batch.get("has_mask", torch.tensor([True] * len(image_batch)))
        mask_batch = batch["mask"].squeeze(1).long()
        common_mask = batch.get("mask-common")  # [B, 1, H, W] or None

        num_samples = min(max_samples, len(image_batch))
        num_logged = 0

        rgb_indices = _select_rgb_bands(image_batch.shape[1], use_metadata_film=use_metadata_film)

        for i in range(len(image_batch)):
            if num_logged >= num_samples:
                break

            # --- Filter: only visualize samples with enough burned pixels ---
            # (see MIN_BURNED_RATIO_FOR_VISUALIZATION at module level)
            has_real_mask = has_mask_flags[i] if isinstance(
                has_mask_flags, (list, torch.Tensor)) else has_mask_flags
            if not has_real_mask:
                continue  # no ground truth → skip

            mask_i_for_filter = mask_batch[i]  # [H, W], values: 0=unburn, 1=burn, 255=ignore
            if common_mask is not None:
                valid_pixels = (common_mask[i].squeeze(0) > MASK_VALID_THRESHOLD)  # [H, W]
            else:
                valid_pixels = (mask_i_for_filter != IGNORE_MASK_INDEX)

            valid_count = valid_pixels.sum()
            if valid_count == 0:
                continue

            burned_count = ((mask_i_for_filter == 1) & valid_pixels).sum()
            burned_ratio = burned_count.float() / valid_count.float()
            if burned_ratio < MIN_BURNED_RATIO_FOR_VISUALIZATION:
                continue
            image_post = image_batch[i]
            image_pre = pre_image_batch[i]
            image_name = batch_image_name[i].replace('\n', '')

            # Compute absolute difference on selected bands
            image_diff = torch.abs(image_post - image_pre)
            vis_image = image_diff[rgb_indices, :, :]  # [3, H, W] or fewer
            # Normalize to [0, 1] for matplotlib (avoids clipping warning)
            vmin = vis_image.min()
            vmax = vis_image.max()
            if vmax - vmin > 1e-6:
                vis_image = (vis_image - vmin) / (vmax - vmin)
            else:
                vis_image = torch.zeros_like(vis_image)

            # Prediction with water/no-data masking
            pred = torch.argmax(outputs[i], dim=0)  # [H, W]
            if common_mask is not None:
                invalid = (common_mask[i].squeeze(0) < MASK_VALID_THRESHOLD)  # [H, W]
                # Use a distinct value for visualization of masked pixels
                pred = pred.clone()
                pred[invalid] = changed_num_classes  # = 2 → index du gris dans la colormap

            # Ground truth mask (always present here — filtered above)
            mask_i = mask_batch[i]

            fig = visualize_prediction(
                image=vis_image,
                mask=mask_i,
                prediction=pred,

                sample_name=image_name[:80],  # truncate long names
                num_classes=num_classes,
                class_colors=class_colors,
            )

            # Build artifact path
            # Use a short, filesystem-safe name
            safe_name = Path(image_name[:60].replace('|', '_').replace('/', '_')).stem
            base_path = f"{artifact_prefix}/{safe_name}"
            if epoch_suffix and trainer is not None:
                artifact_file = f"{base_path}/idx_{i}_epoch_{trainer.current_epoch}.png"
            else:
                artifact_file = f"{base_path}/idx_{i}.png"

            _log_figure(trainer, fig, artifact_file, epoch_suffix=epoch_suffix)

            # Explicitly close figure to prevent memory leak
            plt.close(fig)
            num_logged += 1

    except Exception:
        logger.exception("Error in visualization logging")
        return 0

    return num_logged
