"""Segmentation DINOv3 with Mask2Former model."""

import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import kornia as krn
import torch
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper

from geo_deep_learning.models.segmentation.dinov3 import DINOv3SegmentationModel
from geo_deep_learning.tools.target_converters import semantic_to_instance_masks
from geo_deep_learning.tools.utils import denormalization, load_weights_from_checkpoint
from geo_deep_learning.tools.visualization import visualize_prediction

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)


class SegmentationDINOv3(LightningModule):
    """Segmentation DINOv3 with Mask2Former decoder."""

    def __init__(  # noqa: PLR0913
        self,
        num_classes: int,
        *,
        image_size: tuple[int, int],
        max_samples: int,
        criterion: Callable,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        scheduler_config: dict[str, Any] | None = None,
        class_labels: list[str] | None = None,
        class_colors: list[str] | None = None,
        weights_from_checkpoint_path: str | None = None,
        aux_loss_weight: float = 0.4,
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])
        self.num_classes = num_classes
        self.image_size = image_size
        self.weights_from_checkpoint_path = weights_from_checkpoint_path
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {"interval": "epoch"}
        self.class_colors = class_colors
        self.max_samples = max_samples
        self.threshold = 0.5
        self.aux_loss_weight = aux_loss_weight
        self.criterion = criterion
        num_classes_metric = num_classes + 1 if num_classes == 1 else num_classes
        self.iou_metric = MeanIoU(
            num_classes=num_classes_metric,
            per_class=True,
            input_format="index",
            include_background=True,
        )
        self.labels = (
            [str(i) for i in range(num_classes_metric)]
            if class_labels is None
            else class_labels
        )
        self.iou_classwise_metric = ClasswiseWrapper(
            self.iou_metric,
            labels=self.labels,
        )
        self._total_samples_visualized = 0
        self.train_samples_count = 0
        self.val_samples_count = 0
        self.test_samples_count = 0

    def _apply_aug(self) -> AugmentationSequential:
        """Augmentation pipeline."""
        random_resized_crop_zoom_in = krn.augmentation.RandomResizedCrop(
            size=self.image_size,
            scale=(1.0, 2.0),
            p=0.5,
            align_corners=False,
            keepdim=True,
        )
        random_resized_crop_zoom_out = krn.augmentation.RandomResizedCrop(
            size=self.image_size,
            scale=(0.5, 1.0),
            p=0.5,
            align_corners=False,
            keepdim=True,
        )

        return AugmentationSequential(
            krn.augmentation.RandomHorizontalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomVerticalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomRotation90(
                times=(1, 3),
                p=0.5,
                align_corners=True,
                keepdim=True,
            ),
            random_resized_crop_zoom_in,
            random_resized_crop_zoom_out,
            data_keys=None,
            random_apply=1,
        )

    def configure_model(self) -> None:
        """Configure model."""
        self.model = DINOv3SegmentationModel(num_classes=self.num_classes)
        if self.weights_from_checkpoint_path:
            map_location = self.device
            load_parts = self.hparams.get("load_parts")
            logger.info(
                "Loading weights from checkpoint: %s",
                self.weights_from_checkpoint_path,
            )
            load_weights_from_checkpoint(
                self.model,
                self.weights_from_checkpoint_path,
                load_parts=load_parts,
                map_location=map_location,
            )

    def configure_optimizers(self) -> list[list[dict[str, Any]]]:
        """Configure optimizers."""
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)

        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        """Forward pass."""
        return self.model(image)

    def on_after_batch_transfer(
        self,
        batch: dict[str, Any],
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        """On after batch transfer."""
        if not self.trainer.training:
            return batch
        device = batch["image"].device
        aug = self._apply_aug()
        batch_aug = aug({"image": batch["image"], "mask": batch["mask"]})
        for key in ["image", "mask"]:
            if key in batch_aug and batch_aug[key].device != device:
                batch[key] = batch_aug[key].to(device, non_blocking=True)
            elif key in batch_aug:
                batch[key] = batch_aug[key]
        return batch

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run training step."""
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        self.train_samples_count += batch_size
        y = y.squeeze(1).long()
        outputs = self(x)
        targets = semantic_to_instance_masks(y, self.num_classes)
        loss_dict = self.criterion(outputs, targets)
        loss = sum(
            loss_dict[k] * self.criterion.weight_dict[k]
            for k in ["loss_ce", "loss_mask", "loss_dice"]
            if k in loss_dict
        )
        aux_loss_keys = [
            k for k in loss_dict if k not in ["loss_ce", "loss_mask", "loss_dice"]
        ]
        if aux_loss_keys:
            aux_loss = sum(loss_dict[k] for k in aux_loss_keys)
            loss = loss + self.aux_loss_weight * aux_loss
            self.log(
                "train_aux_loss",
                aux_loss,
                batch_size=batch_size,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=False,
            )
        for k, v in loss_dict.items():
            if k in ["loss_ce", "loss_mask", "loss_dice"]:
                self.log(
                    f"train_{k}",
                    v,
                    batch_size=batch_size,
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=False,
                )
        self.log(
            "train_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=False,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        """On train epoch end."""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        logger.info(
            "Training epoch complete. Processed %d samples",
            self.train_samples_count,
        )
        self.train_samples_count = 0

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run validation step."""
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        self.val_samples_count += batch_size
        y = y.squeeze(1).long()
        outputs = self(x)
        targets = semantic_to_instance_masks(y, self.num_classes)
        loss_dict = self.criterion(outputs, targets)
        loss = sum(
            loss_dict[k] * self.criterion.weight_dict[k]
            for k in ["loss_ce", "loss_mask", "loss_dice"]
            if k in loss_dict
        )
        self.log(
            "val_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=False,
        )
        pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
        pred_masks = outputs["pred_masks"]  # [B, Q, H, W]
        return self._convert_to_semantic(pred_logits, pred_masks, y.shape[-2:])

    def on_validation_epoch_end(self) -> None:
        """On validation epoch end."""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        logger.info(
            "Validation epoch complete. Processed %d samples",
            self.val_samples_count,
        )
        self.val_samples_count = 0

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run test step."""
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        self.test_samples_count += batch_size
        y = y.squeeze(1).long()

        outputs = self(x)
        targets = semantic_to_instance_masks(y, self.num_classes)
        loss_dict = self.criterion(outputs, targets)
        loss = sum(
            loss_dict[k] * self.criterion.weight_dict[k]
            for k in ["loss_ce", "loss_mask", "loss_dice"]
            if k in loss_dict
        )
        pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
        pred_masks = outputs["pred_masks"]  # [B, Q, H, W]
        y_hat = self._convert_to_semantic(pred_logits, pred_masks, y.shape[-2:])
        metrics = self.iou_classwise_metric(y_hat, y)
        metrics["test_loss"] = loss

        # Visualization
        if self._total_samples_visualized < self.max_samples:
            remaining_samples = self.max_samples - self._total_samples_visualized
            samples_to_visualize = min(remaining_samples, len(x))
            samples_visualized = self._log_visualizations(
                trainer=self.trainer,
                batch=batch,
                outputs=y_hat,
                max_samples=samples_to_visualize,
                artifact_prefix="test",
                epoch_suffix=False,
            )
            self._total_samples_visualized += samples_visualized

        self.log_dict(
            metrics,
            batch_size=batch_size,
            prog_bar=False,
            logger=True,
            on_step=False,
            sync_dist=True,
            rank_zero_only=False,
        )

    def on_test_epoch_end(self) -> None:
        """On test epoch end."""
        logger.info(
            "Test epoch complete. Processed %d samples",
            self.test_samples_count,
        )
        self.test_samples_count = 0

    def _convert_to_semantic(
        self,
        pred_logits: Tensor,
        pred_masks: Tensor,
        target_size: tuple[int, int],
    ) -> Tensor:
        """
        Convert Mask2Former outputs to semantic segmentation format.

        Args:
            pred_logits: Class predictions [B, Q, C+1]
            pred_masks: Mask predictions [B, Q, H, W]
            target_size: Target spatial size (H, W)

        Returns:
            Semantic segmentation predictions [B, H, W]

        """
        # Upsample masks to target size if needed
        if pred_masks.shape[-2:] != target_size:
            pred_masks = torch.nn.functional.interpolate(
                pred_masks,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # Get class probabilities (exclude no-object class)
        pred_probs = pred_logits.softmax(dim=-1)[..., :-1]  # [B, Q, C]
        pred_masks = pred_masks.sigmoid()  # [B, Q, H, W]

        # Combine class probabilities with mask predictions
        # For each class c: sum over queries q of (class_prob[q,c] * mask[q])
        # Output: [B, C, H, W]
        semseg = torch.einsum("bqc,bqhw->bchw", pred_probs, pred_masks)

        # Get final prediction by taking argmax over classes
        return semseg.argmax(dim=1)  # [B, H, W]

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
        DINOv3-specific log visualizations.

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
                mean = mean_batch[i]
                std = std_batch[i]
                image = denormalization(image, mean=mean, std=std)

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
                trainer.logger.experiment.log_figure(
                    figure=fig,
                    artifact_file=artifact_file,
                    run_id=trainer.logger.run_id,
                )
        except Exception:
            logger.exception("Error in DINOv3 visualization")
        else:
            return num_samples
