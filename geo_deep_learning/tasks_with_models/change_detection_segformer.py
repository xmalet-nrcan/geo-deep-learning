"""Segmentation SegFormer model."""
import logging
import warnings
from pathlib import Path
from typing import Any, Callable

import kornia as krn
import torch
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import Trainer
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from lightning.pytorch.loggers import TensorBoardLogger
# from segmentation_models_pytorch.losses import JaccardLoss
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torchmetrics import JaccardIndex, F1Score
from torchmetrics.classification import BinaryJaccardIndex

from geo_deep_learning.tasks_with_models.segmentation_segformer import SegmentationSegformer
from geo_deep_learning.tools.visualization import visualize_prediction

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)


class ChangeDetectionSegmentationSegformer(SegmentationSegformer):
    """Segmentation SegFormer model."""

    def __init__(self, encoder: str, *,
                 image_size: tuple[int, int],
                 in_channels: int,
                 num_classes: int,
                 max_samples: int,
                 loss: Callable,
                 optimizer: OptimizerCallable = torch.optim.Adam,
                 scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
                 scheduler_config: dict[str, Any] | None = None,
                 use_dynamic_encoder: bool = False,
                 freeze_layers: list[str] | None = None,
                 weights: str | None = None,
                 class_labels: list[str] | None = None,
                 class_colors: list[str] | None = None,
                 weights_from_checkpoint_path: str | None = None,
                 **kwargs: object) -> None:
        super().__init__(encoder, image_size=image_size, in_channels=in_channels, num_classes=num_classes,
                         max_samples=max_samples, loss=loss, optimizer=optimizer, scheduler=scheduler,
                         scheduler_config=scheduler_config, use_dynamic_encoder=use_dynamic_encoder,
                         freeze_layers=freeze_layers, weights=weights, class_labels=class_labels,
                         class_colors=class_colors, weights_from_checkpoint_path=weights_from_checkpoint_path, **kwargs)

        # self.ce_loss = JaccardLoss(mode='binary', smooth=1e-6, eps=1e-7)
        self.ce_loss = BCEWithLogitsLoss()
        num_classes = self.num_classes if self.num_classes > 1 else 2
        task_type = "multiclass" if num_classes > 2 else "binary"
        if num_classes == 2:
            self.train_iou = BinaryJaccardIndex(threshold=0.3)
            self.val_iou = BinaryJaccardIndex(threshold=0.3)
            self.test_iou = BinaryJaccardIndex(threshold=0.3)
        else:
            self.train_iou = JaccardIndex(task=task_type, num_classes=num_classes)
            self.val_iou = JaccardIndex(task=task_type, num_classes=num_classes)
            self.test_iou = JaccardIndex(task=task_type, num_classes=num_classes)

        self.train_f1 = F1Score(task=task_type, num_classes=num_classes)
        self.val_f1 = F1Score(task=task_type, num_classes=num_classes)
        self.test_f1 = F1Score(task=task_type, num_classes=num_classes)

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
            data_keys=None, )

    def on_before_batch_transfer(
            self,
            batch: dict[str, Any],
            dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:

        aug = AugmentationSequential(krn.augmentation.PadTo(size=self.image_size,
                                                            pad_mode='constant', pad_value=0,
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
    ) -> int:
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
            logger.info("Batch size: %d", len(batch["image"]))
            image_batch = batch["image"]
            mask_batch = batch["mask"].squeeze(1).long()
            batch_image_name = batch["image_name"]
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
            return 0
        else:
            return num_samples

    # --- TRAINING ---
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Run one training step on a batch."""
        x = batch["image"]
        y = batch["mask"].squeeze(1)  # (B, H, W)
        batch_size = x.size(0)

        # Get the targets in the right format
        y_long = y.long()
        y_float = y.unsqueeze(1).float()

        # Forward
        outputs = self(x)
        logits = outputs.out

        # --- Main loss ---
        try:
            main_loss = self.loss(logits, y_long) + self.ce_loss(logits, y_long)
        except ValueError:
            # Si self.ce_loss attend un float (comme JaccardLoss)
            main_loss = self.loss(logits, y_long) + self.ce_loss(logits, y_float)

        # --- Auxiliary losses (si disponibles) ---
        aux_loss = torch.zeros((), device=y.device, dtype=main_loss.dtype)
        if hasattr(outputs, "aux") and outputs.aux:
            for key, weight in getattr(self, "aux_weight", {}).items():
                if not weight or key not in outputs.aux:
                    continue

                logits_aux = outputs.aux[key]
                try:
                    aux_loss += weight * (
                            self.loss(logits_aux, y_long) + self.ce_loss(logits_aux, y_long)
                    )
                except ValueError:
                    aux_loss += weight * (
                            self.loss(logits_aux, y_long) + self.ce_loss(logits_aux, y_float)
                    )

        total_loss = main_loss + aux_loss

        # --- Logging ---
        self.log(
            "train_loss",
            total_loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # --- Calcul des métriques différé (pour éviter de casser autograd) ---
        with torch.no_grad():
            if self.num_classes == 1:
                preds = (logits.sigmoid() > self.threshold).long().squeeze(1)
            else:
                preds = logits.softmax(dim=1).argmax(dim=1)

            # On accumule les prédictions pour calculer les métriques à la fin
            self.train_iou.update(preds, y_long)

            self.train_f1.update(preds, y_long)
            self.log(
                "train_iou_step",
                self.train_iou(preds, y_long),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

        return total_loss

    def on_train_epoch_end(self) -> None:
        """Aggregate and log metrics at the end of each training epoch."""
        try:
            train_iou = self.train_iou.compute()
            train_f1 = self.train_f1.compute()

            self.log("train_iou", train_iou, prog_bar=True, sync_dist=True)
            self.log("train_f1", train_f1, prog_bar=True, sync_dist=True)
        finally:
            # Important : reset pour la prochaine époque
            self.train_iou.reset()
            self.train_f1.reset()



    # --- VALIDATION ---
    def validation_step(self, batch, batch_idx):
        y_hat = super().validation_step(batch, batch_idx)
        y = batch["mask"].squeeze(1).long()

        self.val_iou(y_hat, y)
        self.val_f1(y_hat, y)

        self.log("val_iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return y_hat

    # --- TEST ---
    def test_step(self, batch, batch_idx):
        super().test_step(batch, batch_idx)
        x, y = batch["image"], batch["mask"].squeeze(1).long()
        outputs = self(x)

        if self.num_classes == 1:
            preds = (outputs.out.sigmoid().squeeze(1) > self.threshold).long()
        else:
            preds = outputs.out.softmax(dim=1).argmax(dim=1)
        self.test_iou(preds, y)
        self.test_f1(preds, y)

        self.log("test_iou", self.test_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
