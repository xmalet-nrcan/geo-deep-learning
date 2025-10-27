"""Segmentation SegFormer model."""

import logging
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import kornia as krn
import torch
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper

from geo_deep_learning.models.segmentation.segformer import SegFormerSegmentationModel
from geo_deep_learning.tasks_with_models.segmentation_segformer import SegmentationSegformer
from geo_deep_learning.tools.visualization import visualize_prediction
from geo_deep_learning.utils.models import load_weights_from_checkpoint
from geo_deep_learning.utils.tensors import denormalization

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)


class ChangeDetectionSegmentationSegformer(SegmentationSegformer):
    """Segmentation SegFormer model."""

    def _apply_aug(self) -> AugmentationSequential:
        """Augmentation pipeline."""

        return AugmentationSequential(
                krn.augmentation.RandomHorizontalFlip(p=0.5, keepdim=True),
                krn.augmentation.RandomVerticalFlip(p=0.5, keepdim=True),
                krn.augmentation.RandomRotation90(
                    times=(1, 3),
                    p=0.5,
                    align_corners=True,
                    keepdim=True,
                ),
                data_keys=None,
                random_apply=1, )

    def on_before_batch_transfer(
        self,
        batch: dict[str, Any],
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:

        aug = AugmentationSequential(krn.augmentation.PadTo(size=self.image_size,
                                     pad_mode='constant',
                                     pad_value=0,
                                     keepdim=False), data_keys=None)
        transformed = aug({
            "image": batch["image"],
            "mask": batch["mask"]})
        batch.update(transformed)
        return batch



    def forward(self, image: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(image)

    def _log_visualizations(  # noqa: PLR0913
            self,
            trainer: Trainer,
            batch: dict[str, Any],
            outputs: Tensor,
            max_samples: int,
            artifact_prefix: str = "val",
            *,
            epoch_suffix: bool = True,
    ) -> None:
        """
        SegFormer-specific log visualizations.

        Args:
            trainer: Lightning trainer
            batch: Batch data containing image, mask, image_name, mean, std
            outputs: Model predictions
            max_samples: Maximum number of samples to visualize
            artifact_prefix: Prefix for artifact path ("test" or "val")
            epoch_suffix: Whether to add epoch info to artifact filename

        Returns:
            Number of samples actually visualized

        """
        if batch is None or outputs is None:
            return 0

        try:
            logger.info("Logging visualizations")
            image_batch = batch["image"]
            mask_batch = batch["mask"].squeeze(1).long()
            batch_image_name = batch["image_name"]
            mean_batch = batch["mean"]
            std_batch = batch["std"]
            num_samples = min(max_samples, len(image_batch))
            for i in range(num_samples):
                image = image_batch[i]
                image_name = batch_image_name[i]
                # mean = mean_batch[i]
                # std = std_batch[i]
                # image = denormalization(image, mean=mean, std=std)

                fig = visualize_prediction(
                    image=image,
                    mask=mask_batch[i],
                    prediction=outputs[i],
                    sample_name=image_name,
                    num_classes=self.num_classes,
                    class_colors=self.class_colors,
                )
                base_path = f"{artifact_prefix}/{Path(image_name).stem}"
                if epoch_suffix and trainer is not None:
                    artifact_file = (
                        f"{base_path}/idx_{i}_epoch_{trainer.current_epoch}.png"
                    )
                else:
                    artifact_file = f"{base_path}/idx_{i}.png"
                if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "log_figure"):
                    # MLflowLogger
                    trainer.logger.experiment.log_figure(
                        figure=fig,
                        artifact_file=artifact_file,
                        run_id=getattr(trainer.logger, "run_id", None),
                    )
                elif isinstance(trainer.logger, TensorBoardLogger):
                    # TensorBoardLogger
                    trainer.logger.experiment.add_figure(
                        tag=artifact_file,
                        figure=fig,
                        global_step=trainer.current_epoch if epoch_suffix else 0,
                    )
                else:
                    logger.warning("Logger does not support figure logging.")
        except Exception:
            logger.exception("Error in SegFormer visualization")
        else:
            return num_samples
