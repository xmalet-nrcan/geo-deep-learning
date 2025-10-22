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
from lightning.pytorch.loggers import MLFlowLogger

from torch import Tensor
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
        print(self.hparams)
        self.in_channels = in_channels
        # bands = [ 'M','LOCALINCANGLE','RL','RR','S0', 'SP1','SP2','SP3','RFDI' ] #self.hparams['data']["band_names"]
        # if bands is None:
        #     self.in_channels = len(BandName) + 3
        # else:
        #     bands = [BandName[i] for i in bands]
        #     if BandName.BITMASK_CROPPED in bands:
        #         self.in_channels = len(bands) + 3  # (COMMON MASK, bands, SAT_PASS, BEAM)
        #     else:
        #         self.in_channels = len(bands) + 4  # (COMMON MASK,BITMASK_CROPPED, bands, SAT_PASS, BEAM, )
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


    def _apply_aug(self) -> AugmentationSequential:
        """Augmentation pipeline."""

        pad_to_patch_size = krn.augmentation.PadTo(size=self.image_size,
                                                   pad_mode='constant',
                                                   pad_value=0,
                                                   keepdim=False)



        return AugmentationSequential(
            pad_to_patch_size,
            krn.augmentation.RandomHorizontalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomVerticalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomRotation90(
                times=(1, 3),
                p=0.5,
                align_corners=True,
                keepdim=True,
            ),
            data_keys=None,
            random_apply=1,
        )

    def configure_model(self) -> None:
        """Configure model."""
        self.model = ChangeDetectionModel(
            change_detection_model=self.change_detection_model,
            in_channels=self.in_channels,
            out_channels=self.num_classes,
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
        return self.model(image_pre, image_post)

    # TODO : Modifier pour avoir image pre/post transformées
    def on_before_batch_transfer(
            self,
            batch: dict[str, Any],
            dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        """On before batch transfer."""
        if self.trainer.training:
            aug = self._apply_aug()
            transformed = aug({"image_pre": batch["image_pre"],
                              "image_post": batch["image_post"],
                              "mask": batch["mask"]})
            batch.update(transformed)
        return batch

    # TODO : Modifier pour avoir image pre/post
    def training_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run training step."""
        x_pre, x_post, y, y_hat, loss, batch_size = self._forward_and_get_loss(batch)


        self.log(
            "train_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )

        return loss

    # TODO : Modifier pour avoir image pre/post
    def validation_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run validation step."""
        x_pre, x_post, y, y_hat, loss, batch_size = self._forward_and_get_loss(batch)

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
        if self.num_classes == 1:
            y_hat = (y_hat.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat = y_hat.softmax(dim=1).argmax(dim=1)

        return y_hat

    def test_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run test step."""
        x_pre, x_post, y, y_hat, loss, batch_size = self._forward_and_get_loss(batch)

        if self.num_classes == 1:
            y_hat = (y_hat.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat = y_hat.softmax(dim=1).argmax(dim=1)

        metrics = self.iou_classwise_metric(y_hat, y)
        self.iou_classwise_metric.reset()
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
            rank_zero_only=True,
        )

    def _forward_and_get_loss(self, batch: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any, Any]:
        x_pre, x_post = batch["image_pre"], batch["image_post"]
        y = batch["mask"]
        batch_size = x_post.shape[0]
        y = y.squeeze(1).long()
        y_hat = self(x_pre, x_post)
        y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
        if y_hat.shape[1] == 1 and y.ndim == 3:
            y = y.unsqueeze(1)  # Shape: [B, 1, H, W]
        loss = self.loss(y_hat, y)
        return x_pre, x_post, y, y_hat, loss, batch_size

    def _log_visualizations(  # noqa: PLR0913
            self,
            trainer: Trainer,
            batch: dict[str, Any],
            outputs: Tensor,
            max_samples: int,
            artifact_prefix: str = "val",
            *,
            epoch_suffix: bool = True,
    ) -> Optional[int]:
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
            max_batch = batch["max"]
            num_samples = min(max_samples, len(image_batch))
            for i in range(num_samples):
                image = image_batch[i]
                image_name = batch_image_name[i]
                # TODO : RESTORE WHEN CHECKED
                mean = mean_batch[i]
                std = std_batch[i]
                # image = denormalization(image, mean=mean, std=std,data_type_max=max_batch)
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
                if isinstance(trainer.logger, MLFlowLogger):

                    trainer.logger.experiment.log_figure(
                        figure=fig,
                        artifact_file=artifact_file,
                        run_id=trainer.logger.run_id,
                    )
              
        except Exception:
            logger.exception("Error in SegFormer visualization")
        else:
            return num_samples
