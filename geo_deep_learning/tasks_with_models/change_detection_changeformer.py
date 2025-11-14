"""Segmentation SegFormer model."""

import logging
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import kornia as krn
import torch
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import  TensorBoardLogger
from segmentation_models_pytorch.utils.losses import BCEWithLogitsLoss
from segmentation_models_pytorch.losses import FocalLoss

from torch import Tensor
from torchmetrics import JaccardIndex, F1Score
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper

from geo_deep_learning.datasets.rcm_change_detection_dataset import NO_DATA, BandName  # noqa: F401
from geo_deep_learning.models.change_detection.change_detection_model import ChangeDetectionModel
from geo_deep_learning.tools.visualization import visualize_prediction
from geo_deep_learning.utils.models import load_weights_from_checkpoint

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)


# class DataAugmentation(nn.Module):
#
#     """Data Augmentation Module."""
#
#     def __init__(self, image_size: tuple[int, int]) -> None:
#         """Initialize the augmentation module."""
#         super().__init__()
#         self.aug = AugmentationSequential(
#             krn.augmentation.PadTo(size=image_size,
#                                    pad_mode='constant',
#                                    pad_value=0,
#                                    keepdim=False),
#             krn.augmentation.RandomHorizontalFlip(p=0.5, keepdim=True),
#             krn.augmentation.RandomVerticalFlip(p=0.5, keepdim=True),
#             krn.augmentation.RandomRotation90(
#                 times=(1, 3),
#                 p=0.5,
#                 align_corners=True,
#                 keepdim=True,
#             ),
#             data_keys=None,
#         )
#     @torch.no_grad()
#     def forward(self, x: Tensor) -> Tensor:
#         """Forward pass."""
#         return self.aug(x)


class ChangeDetectionChangeFormer(LightningModule):
    """Change Detection with ChangeFormer V6 model."""

    def __init__(  # noqa: PLR0913
            self,
            change_detection_model: str,
            *,
            image_size: tuple[int, int],
            num_classes: int,
            max_samples: int,
            loss: Callable,
            optimizer: OptimizerCallable = torch.optim.Adam,
            scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
            scheduler_config: dict[str, Any] | None = None,
            weights: str | None = None,
            class_labels: list[str] | None = None,
            class_colors: list[str] | None = None,
            weights_from_checkpoint_path: str | None = None,
            in_channels: int | None = None,
            **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters()
        self.change_detection_model = change_detection_model
        self.in_channels = in_channels

        self.num_classes = num_classes  # Should be 2
        self.image_size = image_size
        self.max_samples = max_samples

        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {"interval": "epoch"}

        self.weights = weights
        self.weights_from_checkpoint_path = weights_from_checkpoint_path

        self.class_colors = class_colors
        self.threshold = 0.5

        self.changed_num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.iou_metric = MeanIoU(
            num_classes=self.changed_num_classes,
            per_class=True,
            input_format="index",
            include_background=True,
        )
        self.labels = (
            [str(i) for i in range(self.changed_num_classes)]
            if class_labels is None
            else class_labels
        )
        self.iou_classwise_metric = ClasswiseWrapper(
            self.iou_metric,
            labels=self.labels,
        )
        self._total_samples_visualized = 0

        self.ce_loss = FocalLoss(mode='binary')
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
        transformed = aug({"image_pre": batch["image_pre"],
                           "image_post": batch["image_post"],
                           "image": batch["image_post"],
                           "mask": batch["mask"]})
        batch.update(transformed)
        return batch

    def configure_model(self) -> None:
        """Configure model."""
        self.model = ChangeDetectionModel(
            change_detection_model=self.change_detection_model,
            in_channels=self.in_channels,
            out_channels=self.num_classes  + 1 if self.num_classes == 1 else self.num_classes,
        )

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
        if (
                self.hparams["scheduler"]["class_path"]
                == "torch.optim.lr_scheduler.OneCycleLR"
        ):
            max_lr = (
                self.hparams.get("scheduler", {}).get("init_args", {}).get("max_lr")
            )
            stepping_batches = self.trainer.estimated_stepping_batches
            if stepping_batches > -1:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=stepping_batches,
                )
            elif (
                    stepping_batches == -1
                    and getattr(self.trainer.datamodule, "epoch_size", None) is not None
            ):
                batch_size = self.trainer.datamodule.batch_size
                epoch_size = self.trainer.datamodule.epoch_size
                accumulate_grad_batches = self.trainer.accumulate_grad_batches
                max_epochs = self.trainer.max_epochs
                steps_per_epoch = math.ceil(
                    epoch_size / (batch_size * accumulate_grad_batches),
                )
                buffer_steps = int(steps_per_epoch * accumulate_grad_batches)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    steps_per_epoch=steps_per_epoch + buffer_steps,
                    epochs=max_epochs,
                )
            else:
                stepping_batches = (
                    self.hparams.get("scheduler", {})
                    .get("init_args", {})
                    .get("total_steps")
                )
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=stepping_batches,
                )
        else:
            scheduler = self.scheduler(optimizer)

        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def forward(self, image_pre: Tensor, image_post: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(image_pre, image_post)[-1]  # Because ChangeFormer output a list in its forward pass.

    # # TODO : Modifier pour avoir image pre/post transformées
    # def on_before_batch_transfer(
    #         self,
    #         batch: dict[str, Any],
    #         dataloader_idx: int,  # noqa: ARG002
    # ) -> dict[str, Any]:
    #     """On before batch transfer."""
    #     if self.trainer.training:
    #         aug = self._apply_aug()
    #
    #         transformed = aug({"image_pre": batch["image_pre"],
    #                           "image_post": batch["image_post"],
    #                             "image": batch["image_post"],
    #                           "mask": batch["mask"]})
    #         batch.update(transformed)
    #     return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training:
            return batch
        aug = self._apply_aug()
        device = batch["image"].device

        transformed = aug({"image_pre": batch["image_pre"],
                           "image_post": batch["image_post"],
                           "image": batch["image_post"],
                           "mask": batch["mask"]})
        for key in ["image", "mask", "image_pre", "image_post"]:
            if key in transformed:
                batch[key] = transformed[key].to(device, non_blocking=True)
        return batch

    # TODO : Modifier pour avoir image pre/post
    def training_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run training step."""
        x_pre, x_post, y,one_hot,  logits, loss, batch_size = self._forward_and_get_loss(batch)
        # --- Logging ---
        self.log(
            "train_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # --- Calcul des métriques différé (pour éviter de casser autograd) ---
        with torch.no_grad():
            # On accumule les prédictions pour calculer les métriques à la fin
            self.train_iou.update(logits, one_hot)

            self.train_f1.update(logits, one_hot)
            self.log(
                "train_iou_step",
                self.train_iou(logits, one_hot),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

            self.log("train_iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    # TODO : Modifier pour avoir image pre/post
    def validation_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run validation step."""
        x_pre, x_post, y,one_hot,  y_hat, loss, batch_size = self._forward_and_get_loss(batch)

        self.log(
            "val_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )


        self.val_iou(y_hat, one_hot)
        self.val_f1(y_hat, one_hot)

        self.log("val_iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return y_hat

    def test_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run test step."""
        x_pre, x_post, y,one_hot, y_hat, loss, batch_size = self._forward_and_get_loss(batch)
        # Convert logits to class predictions
        y_pred = torch.argmax(y_hat, dim=1)
        y_true = torch.argmax(one_hot, dim=1)

        metrics = self.iou_classwise_metric(y_pred, y_true)

        metrics["test_loss"] = loss

        if self._total_samples_visualized < self.max_samples:
            remaining_samples = self.max_samples - self._total_samples_visualized
            samples_to_visualize = min(remaining_samples, len(x_post))
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
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=False,
        )

        self.test_iou(y_hat, one_hot)
        self.test_f1(y_hat, one_hot)

        self.log("test_iou", self.test_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def _forward_and_get_loss(self,batch: dict[str, Any]) -> tuple[Any, Any,Any,  Any, Any, Any, Any]:
        x_pre, x_post = batch["image_pre"], batch["image_post"]
        y = batch["mask"]
        batch_size = x_post.shape[0]

        logits = self(x_pre, x_post)
        y_float = y.float()


        y_one_hot = y.squeeze(1) if y.dim() == 4 else y
        one_hot = torch.nn.functional.one_hot(y_one_hot.long(), num_classes=self.num_classes+1 if self.num_classes==1 else self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).contiguous().float()

        # --- Main loss ---
        main_loss = self.loss(logits.contiguous(),one_hot) + self.ce_loss(logits.contiguous(), one_hot)
 

        return x_pre, x_post, y_float,one_hot, logits, main_loss, batch_size

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
                    prediction=torch.argmax(outputs[i], dim=1),
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
