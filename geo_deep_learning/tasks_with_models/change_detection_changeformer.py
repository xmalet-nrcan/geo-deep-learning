"""Segmentation SegFormer model."""

from pathlib import Path

import kornia as krn
import logging
import math
import numpy as np
import rasterio as rio
import torch
import torch.nn.functional as F
import warnings
from collections.abc import Callable
from datetime import datetime
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from rasterio.transform import Affine
from torch import Tensor
from torchmetrics import JaccardIndex, F1Score
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper
from typing import Any

from geo_deep_learning.datasets.rcm_change_detection_dataset import NO_DATA, BandName  # noqa: F401
from geo_deep_learning.models.change_detection.change_detection_model import ChangeDetectionModel
from geo_deep_learning.tools.visualization import visualize_prediction
from geo_deep_learning.utils.models import load_weights_from_checkpoint

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)
IGNORE_MASK_INDEX=255

class ChangeDetectionChangeFormer(LightningModule):
    """Change Detection with ChangeFormer V6 model."""

    def __init__(  # noqa: PLR0913
            self,
            change_detection_model: str,
            *,
            image_size: tuple[int, int],
            num_classes: int,
            max_samples: int,
            main_loss: Callable,
            secondary_loss: Callable,
            loss_ratio=(1.0, 1.0),
            optimizer: OptimizerCallable = torch.optim.Adam,
            scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
            scheduler_config: dict[str, Any] | None = None,
            weights: str | None = None,
            burned_class_weight: float = 1.0,
            class_labels: list[str] | None = None,
            class_colors: list[str] | None = None,
            weights_from_checkpoint_path: str | None = None,
            in_channels: int | None = None,
            threshold: float = 0.5,
            predict_output_dir: str | None = None,  # For Outputs
            **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters()
        self.change_detection_model = change_detection_model
        self.in_channels = in_channels
        self.burned_class_weight = float(burned_class_weight)
        if self.burned_class_weight < 1.0:
            msg = "burned_class_weight must be >= 1.0"
            raise ValueError(msg)
        self.num_classes = num_classes  # Should be 2
        self.image_size = image_size
        self.max_samples = max_samples

        self.main_loss = main_loss
        self.secondary_loss = secondary_loss
        self.loss_ratio = loss_ratio

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {"interval": "epoch"}

        self.weights = weights
        self.weights_from_checkpoint_path = weights_from_checkpoint_path

        self.class_colors = class_colors
        self.threshold = threshold

        self.changed_num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.labels = (
            [str(i) for i in range(self.changed_num_classes)]
            if class_labels is None
            else class_labels
        )
        # ----- Validation -----
        self.val_iou_metric = MeanIoU(
            num_classes=self.changed_num_classes,
            per_class=True,
            input_format="index",
            include_background=True,
        )

        self.val_iou_classwise = ClasswiseWrapper(
            self.val_iou_metric,
            labels=self.labels,
        )

        # ----- Test -----
        self.test_iou_metric = MeanIoU(
            num_classes=self.changed_num_classes,
            per_class=True,
            input_format="index",
            include_background=True,
        )

        self.test_iou_classwise = ClasswiseWrapper(
            self.test_iou_metric,
            labels=self.labels,
        )

        self._total_samples_visualized = 0

        num_classes = self.num_classes if self.num_classes > 1 else 2
        task_type = "multiclass" if num_classes > 2 else "binary"

        if num_classes == 2:
            self.train_iou = BinaryJaccardIndex(threshold=self.threshold, ignore_index=IGNORE_MASK_INDEX)
            self.val_iou = BinaryJaccardIndex(threshold=self.threshold, ignore_index=IGNORE_MASK_INDEX)
            self.test_iou = BinaryJaccardIndex(threshold=self.threshold, ignore_index=IGNORE_MASK_INDEX)
        else:
            self.train_iou = JaccardIndex(task=task_type, num_classes=num_classes)
            self.val_iou = JaccardIndex(task=task_type, num_classes=num_classes)
            self.test_iou = JaccardIndex(task=task_type, num_classes=num_classes)

        self.train_f1 = F1Score(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX)
        self.val_f1 = F1Score(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX)
        self.test_f1 = F1Score(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX)

        if num_classes == 2:
            self.train_precision = BinaryPrecision(threshold=self.threshold, ignore_index=IGNORE_MASK_INDEX )
            self.val_precision = BinaryPrecision(threshold=self.threshold, ignore_index=IGNORE_MASK_INDEX )
            self.test_precision = BinaryPrecision(threshold=self.threshold, ignore_index=IGNORE_MASK_INDEX )
            self.train_recall = BinaryRecall(threshold=self.threshold, ignore_index=IGNORE_MASK_INDEX )
            self.val_recall = BinaryRecall(threshold=self.threshold, ignore_index=IGNORE_MASK_INDEX )
            self.test_recall = BinaryRecall(threshold=self.threshold, ignore_index=IGNORE_MASK_INDEX )
        else:
            from torchmetrics import Precision, Recall
            self.train_precision = Precision(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX )
            self.val_precision = Precision(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX )
            self.test_precision = Precision(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX )
            self.train_recall = Recall(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX )
            self.val_recall = Recall(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX )
            self.test_recall = Recall(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX )

        self.predict_output_dir = predict_output_dir

    def _apply_geo_aug(self) -> AugmentationSequential:
        """Geometric augmentations (applied to images + masks)."""
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
        )

    def _apply_intensity_aug(self) -> AugmentationSequential:
        """Intensity augmentations (applied to images only)."""
        return AugmentationSequential(
            krn.augmentation.RandomGaussianNoise(mean=0.0, std=0.05, p=0.3, keepdim=True),
            krn.augmentation.RandomGaussianBlur(
                kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.3, keepdim=True
            ),
            krn.augmentation.RandomErasing(
                scale=(0.02, 0.1), ratio=(0.3, 3.3), p=0.3, keepdim=True
            ),
            data_keys=None,
        )

    def on_before_batch_transfer(
            self,
            batch: dict[str, Any],
            dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        aug = AugmentationSequential(
            krn.augmentation.PadTo(size=self.image_size,
                                   pad_mode='constant',
                                   pad_value=0,
                                   keepdim=False),
            data_keys=None,
        )

        keys_to_pad = {"image_pre": batch["image_pre"],
                       "image": batch["image"]}

        # En predict, mask et mask-common sont toujours présents dans votre dataset
        # car __getitem__ les retourne toujours
        for mask_names in ['mask', 'mask-common', 'water_mask']:
            if mask_names in batch:
                if mask_names == 'mask':
                    keys_to_pad[mask_names] = batch[mask_names]
                else:
                    keys_to_pad[mask_names] = batch[mask_names].to(torch.float32)

        transformed = aug(keys_to_pad)
        batch.update(transformed)
        return batch

    def configure_model(self) -> None:
        """Configure model."""
        self.model = ChangeDetectionModel(
            change_detection_model=self.change_detection_model,
            in_channels=self.in_channels,
            out_channels=self.num_classes + 1 if self.num_classes == 1 else self.num_classes,
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
            init_args = self.hparams.get("scheduler", {}).get("init_args", {})
            max_lr = init_args.get("max_lr")
            # Récupérer les paramètres optionnels du YAML
            extra_kwargs = {}
            for key in ("pct_start", "anneal_strategy", "div_factor", "final_div_factor",
                        "three_phase", "cycle_momentum"):
                if key in init_args:
                    extra_kwargs[key] = init_args[key]

            stepping_batches = self.trainer.estimated_stepping_batches
            if stepping_batches > -1:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=stepping_batches,
                    **extra_kwargs,
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
                    **extra_kwargs,
                )
            else:
                total_steps = init_args.get("total_steps")
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=total_steps,
                    **extra_kwargs,
                )
        else:
            scheduler = self.scheduler(optimizer)

        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def forward(self, image_pre: Tensor, image_post: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(image_pre, image_post)[-1]  # Because ChangeFormer output a list in its forward pass.

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training:
            return batch
        device = batch["image"].device

        # 1. Geometric augmentations on images + masks together
        geo_aug = self._apply_geo_aug()
        keys_to_aug = {
            "image_pre": batch["image_pre"],
            "image": batch["image"],
        }
        if "mask-common" in batch:
            keys_to_aug["mask-common"] = batch["mask-common"].to(torch.float32)
        if "mask" in batch:
            keys_to_aug["mask"] = batch["mask"]

        transformed = geo_aug(keys_to_aug)
        for key in transformed:
            batch[key] = transformed[key].to(device, non_blocking=True)

        # 2. Intensity augmentations on images only
        intensity_aug = self._apply_intensity_aug()
        for img_key in ["image_pre", "image"]:
            batch[img_key] = intensity_aug({img_key: batch[img_key]})[img_key]
        return batch

    # TODO : Modifier pour avoir image pre/post
    def training_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run training step."""
        x_pre, x_post, y, one_hot, logits, main_loss, loss, ce_loss, batch_size = self._forward_and_get_loss(batch)
        # --- Logging ---
        self.log(
            "train_loss",
            main_loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log("main_loss", main_loss, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("ce_loss", ce_loss, on_epoch=True, sync_dist=True, batch_size=batch_size)

        # --- Calcul des métriques différé (pour éviter de casser autograd) ---
        with torch.no_grad():
            common_mask = batch["mask-common"]  # [B, 1, H, W]
            valid_preds, valid_targets = self._extract_valid_pixels(logits, one_hot, common_mask)

            # On accumule les prédictions pour calculer les métriques à la fin
            valid_mask = valid_preds != IGNORE_MASK_INDEX
            if valid_mask.any():
                vp = valid_preds[valid_mask]
                vt = valid_targets[valid_mask]
                self.train_iou.update(vp, vt)
                self.train_f1.update(vp, vt)
                self.train_precision.update(vp, vt)
                self.train_recall.update(vp, vt)

        return main_loss

    @staticmethod
    def _extract_valid_pixels(
            logits: Tensor,
            one_hot: Tensor,
            common_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Extract valid pixels for metric computation.

        Invalid pixels are set to IGNORE_MASK_INDEX (255) so that metrics
        configured with ignore_index=IGNORE_MASK_INDEX will skip them.

        Args:
            logits: [B, C, H, W] model output logits
            one_hot: [B, C, H, W] one-hot encoded targets
            common_mask: [B, 1, H, W] validity mask (1=valid, 0=invalid)

        Returns:
            all_preds: [M] predicted class indices (invalid pixels = IGNORE_MASK_INDEX)
            all_targets: [M] target class indices (invalid pixels = IGNORE_MASK_INDEX)
        """
        # valid_pixels : [B, H, W] booléen
        valid_pixels = (common_mask.squeeze(1) > 0.5)  # robust to float imprecision

        # Prédictions et targets en indices de classe : [B, H, W]
        preds = torch.argmax(logits, dim=1)  # [B, H, W]
        targets = torch.argmax(one_hot, dim=1)  # [B, H, W]

        # Mettre les pixels invalides à IGNORE_MASK_INDEX pour qu'ils soient ignorés par les métriques
        preds[~valid_pixels] = IGNORE_MASK_INDEX
        targets[~valid_pixels] = IGNORE_MASK_INDEX

        # Retourner tous les pixels aplatis — les métriques avec ignore_index ignoreront IGNORE_MASK_INDEX
        return preds.flatten(), targets.flatten()

    def on_train_epoch_end(self):
        self.log("train_iou", self.train_iou.compute(), prog_bar=True, sync_dist=True)
        self.log("train_f1", self.train_f1.compute(), prog_bar=True, sync_dist=True)
        self.log("train_precision", self.train_precision.compute(), prog_bar=True, sync_dist=True)
        self.log("train_recall", self.train_recall.compute(), prog_bar=True, sync_dist=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)

        self.train_iou.reset()
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def validation_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run validation step."""
        has_mask = batch.get("has_mask", torch.tensor([True]))
        if not has_mask.any():
            return None  # skip ce batch
        x_pre, x_post, y, one_hot, logits, main_loss, loss, ce_loss, batch_size = self._forward_and_get_loss(batch)

        self.log(
            "val_loss",
            main_loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        with torch.no_grad():
            # Masquer les pixels invalides avant de mettre à jour les métriques
            common_mask = batch["mask-common"]  # [B, 1, H, W]
            valid_preds, valid_targets = self._extract_valid_pixels(logits, one_hot, common_mask)

            if valid_preds.numel() > 0:
                # Filtrer les pixels valides (exclure IGNORE_MASK_INDEX)
                # car BinaryJaccardIndex et MeanIoU n'acceptent pas de valeurs hors [0, num_classes-1]
                valid_mask = valid_preds != IGNORE_MASK_INDEX
                if valid_mask.any():
                    vp = valid_preds[valid_mask]
                    vt = valid_targets[valid_mask]
                    self.val_iou_classwise.update(vp, vt)
                    self.val_iou(vp, vt)
                    self.val_f1(vp, vt)
                    self.val_precision(vp, vt)
                    self.val_recall(vp, vt)

        return logits

    def on_validation_epoch_end(self):
        # Classwise IoU
        classwise_iou = self.val_iou_classwise.compute()
        for class_name, value in classwise_iou.items():
            self.log(f"val_iou_{class_name}", value, prog_bar=False, sync_dist=True)

        # Global metrics
        val_iou = self.val_iou.compute()
        val_f1 = self.val_f1.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()

        self.log("val_iou", val_iou, prog_bar=True, sync_dist=True)
        self.log("val_f1", val_f1, prog_bar=True, sync_dist=True)
        self.log("val_precision", val_precision, prog_bar=True, sync_dist=True)
        self.log("val_recall", val_recall, prog_bar=True, sync_dist=True)
        # In binary setup this recall corresponds to class 1 (burned).
        self.log("val_recall_burn", val_recall, prog_bar=True, sync_dist=True)

        # Reset all
        self.val_iou_classwise.reset()
        self.val_iou.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def test_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run test step."""

        has_mask = batch.get("has_mask", torch.tensor([True]))
        if not has_mask.any():
            return None

        x_pre, x_post, y, one_hot, logits, main_loss, loss, ce_loss, batch_size = self._forward_and_get_loss(batch)
        # Convert logits to class predictions
        y_pred = torch.argmax(logits, dim=1)
        y_true = torch.argmax(one_hot, dim=1)

        # --- Update metrics ---
        with torch.no_grad():
            common_mask = batch["mask-common"]  # [B, 1, H, W]
            valid_preds, valid_targets = self._extract_valid_pixels(logits, one_hot, common_mask)

            if valid_preds.numel() > 0:
                valid_mask = valid_preds != IGNORE_MASK_INDEX
                if valid_mask.any():
                    vp = valid_preds[valid_mask]
                    vt = valid_targets[valid_mask]
                    self.test_iou_classwise.update(vp, vt)
                    self.test_iou.update(vp, vt)
                    self.test_f1.update(vp, vt)
                    self.test_precision.update(vp, vt)
                    self.test_recall.update(vp, vt)

        # --- Log test loss (epoch-aggregated) ---
        self.log(
            "test_loss",
            main_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # --- Visualisations ---
        if self._total_samples_visualized < self.max_samples:
            remaining = self.max_samples - self._total_samples_visualized
            samples_to_visualize = min(remaining, len(x_post))

            self._total_samples_visualized += self._log_visualizations(
                trainer=self.trainer,
                batch=batch,
                outputs=logits,
                max_samples=samples_to_visualize,
                artifact_prefix="test",
                epoch_suffix=False,
            )

    def on_test_epoch_end(self):
        # --- Classwise IoU ---
        classwise_metrics = self.test_iou_classwise.compute()
        for class_name, value in classwise_metrics.items():
            self.log(
                f"test_iou_{class_name}",
                value,
                prog_bar=False,
                sync_dist=True,
            )

        # --- Global metrics ---
        self.log("test_iou", self.test_iou.compute(), prog_bar=True, sync_dist=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True, sync_dist=True)
        self.log("test_precision", self.test_precision.compute(), prog_bar=True, sync_dist=True)
        self.log("test_recall", self.test_recall.compute(), prog_bar=True, sync_dist=True)
        # --- Reset metrics ---
        self.test_iou_classwise.reset()
        self.test_iou.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()

    def _forward_and_get_loss(self, batch: dict[str, Any]) -> tuple[ Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        x_pre, x_post = batch["image_pre"], batch["image"]
        y = batch["mask"]
        common_data_mask = batch["mask-common"]

        batch_size = x_post.shape[0]
        # Vérif entrées images
        if not torch.isfinite(x_pre).all():
            raise RuntimeError("x_pre contains NaN/Inf")
        if not torch.isfinite(x_post).all():
            raise RuntimeError("x_post contains NaN/Inf")

        # S'assurer que le masque commun est bien en float et sans NaN
        common_data_mask = common_data_mask.to(dtype=torch.float32)
        common_data_mask = torch.nan_to_num(common_data_mask, nan=0.0, posinf=1.0, neginf=0.0)
        if logger.isEnabledFor(logging.DEBUG):
            with torch.no_grad():
                logger.debug(
                    "x_pre stats: min=%.4f, max=%.4f, mean=%.4f",
                    x_pre.min().item(), x_pre.max().item(), x_pre.mean().item(),
                )
                logger.debug(
                    "x_post stats: min=%.4f, max=%.4f, mean=%.4f",
                    x_post.min().item(), x_post.max().item(), x_post.mean().item(),
                )
        # --- Remplacer les images quasi-vides par du bruit faible ---
        # pour éviter NaN dans LayerNorm (variance ~ 0 → gradient explose)
        valid_ratio = common_data_mask.flatten(1).mean(dim=1)  # [B]
        min_valid_ratio = 0.05  # au moins 10% de pixels valides
        bad_mask = valid_ratio < min_valid_ratio  # [B] booléen
        if bad_mask.any():
            n_bad = bad_mask.sum().item()
            logger.warning(
                "Patching %d/%d samples with <%.0f%% valid pixels (ratios: %s)",
                n_bad, batch_size, min_valid_ratio * 100,
                [f"{r:.3f}" for r, b in zip(valid_ratio.tolist(), bad_mask.tolist()) if b],
            )
            # Remplir les samples quasi-vides avec du bruit uniforme [0, 0.01]
            # pour que LayerNorm ait une variance > 0
            noise = torch.rand_like(x_pre[0:1]) * 0.01
            for idx in bad_mask.nonzero(as_tuple=True)[0]:
                x_pre[idx] = noise[0]
                x_post[idx] = noise[0]
                # Mettre le masque à 0 pour exclure ces samples de la loss
                common_data_mask[idx] = 0.0
                y[idx] = 0

        logits = self(x_pre, x_post)  # [B, C, H, W]
        y_float = y.float()
        logits_no_nan = torch.nan_to_num(logits, nan=1e15, posinf=1.0, neginf=0.0)
        num_classes = self.changed_num_classes

        # Vérifier les logits (NaN résiduel)
        if not torch.isfinite(logits).all():
            with torch.no_grad():
                logger.warning(
                    "Logits contain non-finite values — skipping batch. "
                    "pre_names=%s, post_names=%s",
                    batch.get('image_pre_name', 'N/A'),
                    batch.get('image_name_post', 'N/A'),
                )
            zero_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
            logits_safe = torch.zeros_like(logits)
            dummy_one_hot = torch.zeros(
                (batch_size, num_classes, x_post.shape[2], x_post.shape[3]),
                device=logits.device, dtype=logits.dtype, )

            return x_pre, x_post, y.float(), dummy_one_hot, logits_safe, zero_loss, zero_loss, zero_loss, batch_size

        # Préparation du one-hot
        y_one_hot = y.squeeze(1) if y.dim() == 4 else y
        y_one_hot = y_one_hot.clamp(min=0, max=num_classes - 1)  # clamp aussi les 255 → num_classes-1
        one_hot = torch.nn.functional.one_hot(y_one_hot.long(), num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).contiguous().float()

        # common_data_mask : [B, 1, H, W], 1=valide, 0=invalide (eau, no-data, padding)
        # Expand le masque pour matcher les dimensions des logits et du one-hot
        loss_mask = common_data_mask.to(dtype=torch.float32)
        if loss_mask.dim() == 4 and loss_mask.shape[1] == 1:
            loss_mask_expanded = loss_mask.expand_as(logits_no_nan)  # [B, C, H, W]
        else:
            loss_mask_expanded = loss_mask.unsqueeze(1).expand_as(logits_no_nan)

        # Appliquer le masque : mettre à 0 les logits et targets pour les pixels invalides
        # afin qu'ils ne contribuent pas à la loss
        masked_logits = logits_no_nan * loss_mask_expanded
        masked_one_hot = one_hot * loss_mask_expanded

        # Vérifier qu'il reste des pixels valides
        valid_sum = common_data_mask.sum()
        if valid_sum == 0:
            # Eviter NaN si la loss divise par le nombre de pixels
            main_loss = torch.tensor(0.0, device=logits_no_nan.device, dtype=logits_no_nan.dtype, requires_grad=True)
            ce_loss = torch.tensor(0.0, device=logits_no_nan.device, dtype=logits_no_nan.dtype, requires_grad=True)
            return x_pre, x_post, y_float, one_hot, logits_no_nan, main_loss, main_loss, ce_loss, batch_size

        w_ml, w_sl = self.loss_ratio

        # Vérifier entrées de la loss
        if not torch.isfinite(one_hot).all():
            raise RuntimeError("One-hot targets contain non-finite values (NaN/Inf).")

        # --- Losses ---
        ce_loss = self.secondary_loss(masked_logits.contiguous(), masked_one_hot)
        loss = self.main_loss(masked_logits.contiguous(), masked_one_hot)
        # burned_fn_penalty = self._burned_false_negative_penalty(
        #     logits=logits_no_nan,
        #     one_hot=one_hot,
        #     valid_mask=common_data_mask,
        # )
        main_loss = w_sl * ce_loss + w_ml * loss # + burned_fn_penalty

        # Dernière vérification
        if not torch.isfinite(main_loss):
            raise RuntimeError(
                f"Computed loss is NaN/Inf. "
                f"ce_loss={ce_loss.detach().cpu().item()}, "
                f"loss={loss.detach().cpu().item()}"
            )

        return x_pre, x_post, y_float, one_hot, logits_no_nan, main_loss, loss, ce_loss, batch_size

    def _burned_false_negative_penalty(
            self,
            logits: Tensor,
            one_hot: Tensor,
            valid_mask: Tensor,
    ) -> Tensor:
        """Add extra penalty on positive (burned) pixels to reduce false negatives."""
        if self.burned_class_weight <= 1.0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        if logits.shape[1] > 1:
            burned_logits = logits[:, 1:2, :, :]
            burned_targets = one_hot[:, 1:2, :, :]
        else:
            burned_logits = logits
            burned_targets = one_hot

        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.to(dtype=logits.dtype)

        pos_weight = torch.tensor(
            self.burned_class_weight,
            device=logits.device,
            dtype=logits.dtype,
        )
        penalty_map = F.binary_cross_entropy_with_logits(
            burned_logits,
            burned_targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        penalty_map = penalty_map * valid_mask
        valid_pixels = valid_mask.sum().clamp_min(1.0)
        return penalty_map.sum() / valid_pixels

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
        """Log visualization figures comparing input diff, ground truth, and prediction.

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

            # Determine which bands to use for RGB visualization.
            # Skip band 0 (COMMON_MASK) and last 2 (SAT_PASS, BEAM).
            # Pick up to 3 data bands from the middle for a meaningful composite.
            num_bands = image_batch.shape[1]
            data_band_start = 1  # skip COMMON_MASK
            data_band_end = max(num_bands - 2, data_band_start + 1)  # skip SAT_PASS, BEAM
            available = list(range(data_band_start, data_band_end))
            # Take 3 evenly spaced bands (or fewer if not enough)
            if len(available) >= 3:
                step = max(1, len(available) // 3)
                rgb_indices = [available[0], available[len(available) // 2], available[-1]]
            else:
                rgb_indices = available[:3]

            for i in range(num_samples):
                image_post = image_batch[i]
                image_pre = pre_image_batch[i]
                image_name = batch_image_name[i].replace('\n', '')

                # Compute absolute difference on selected bands
                image_diff = torch.abs(image_post - image_pre)
                vis_image = image_diff[rgb_indices, :, :]  # [3, H, W] or fewer

                # Prediction with water/no-data masking
                pred = torch.argmax(outputs[i], dim=0)  # [H, W]
                if common_mask is not None:
                    invalid = (common_mask[i].squeeze(0) < 0.5)  # [H, W]
                    # Use a distinct value (255) for visualization of masked pixels
                    pred = pred.clone()
                    effective_num_classes = self.num_classes + 1 if self.num_classes == 1 else self.num_classes
                    pred[invalid] = effective_num_classes  # = 2 → index du gris dans la colormap

                # Ground truth mask
                has_real_mask = has_mask_flags[i] if isinstance(
                    has_mask_flags, (list, torch.Tensor)) else has_mask_flags
                mask_i = mask_batch[i] if has_real_mask else None

                fig = visualize_prediction(
                    image=vis_image,
                    mask=mask_i,
                    prediction=pred,
                    sample_name=image_name[:80],  # truncate long names
                    num_classes=self.num_classes,
                    class_colors=self.class_colors,
                )

                # Build artifact path
                # Use a short, filesystem-safe name
                safe_name = Path(image_name[:60].replace('|', '_').replace('/', '_')).stem
                base_path = f"{artifact_prefix}/{safe_name}"
                if epoch_suffix and trainer is not None:
                    artifact_file = f"{base_path}/idx_{i}_epoch_{trainer.current_epoch}.png"
                else:
                    artifact_file = f"{base_path}/idx_{i}.png"

                # Log to appropriate logger
                if hasattr(trainer.logger, "experiment") and hasattr(
                        trainer.logger.experiment, "log_figure"):
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

                # Explicitly close figure to prevent memory leak
                plt.close(fig)
                num_logged += 1

        except Exception:
            logger.exception("Error in visualization logging")
            return 0

        return num_logged

    def predict_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        """Run prediction step (inference only, no loss/metrics)."""

        # TODO : Masquer l'eau

        x_pre = batch["image_pre"]
        x_post = batch["image"]

        # Forward pass → logits [B, C, H, W]
        with torch.no_grad():
            logits = self(x_pre, x_post)

        # Convertir en probabilités et en classes prédites
        if self.num_classes == 1:
            # Binaire : 2 classes (0=no-change, 1=change)
            probs = torch.softmax(logits, dim=1)  # [B, 2, H, W]
            y_pred = torch.argmax(probs, dim=1)  # [B, H, W]
        else:
            probs = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(probs, dim=1)

        # --- Masquer l'eau avec NO_DATA (32767) ---
        # mask-common inclut déjà le masque d'eau (combiné dans le dataset)
        # On peut aussi utiliser water_mask directement pour être explicite
        if "mask-common" in batch:
            common_mask = batch["mask-common"]  # [B, 1, H, W] bool ou float
            # common_mask == 1 → pixel valide, == 0 → pixel à masquer (eau, no-data, etc.)
            invalid_mask = (common_mask.squeeze(1) == 0)  # [B, H, W]
            y_pred = y_pred.masked_fill(invalid_mask, NO_DATA)
        elif "water_mask" in batch:
            water_mask = batch["water_mask"]  # [B, 1, H, W]
            is_water = (water_mask.squeeze(1) > 0)  # eau = valeur > 0
            y_pred = y_pred.masked_fill(is_water, NO_DATA)

        # Retourner un dict avec tout ce qu'il faut pour sauvegarder après
        result = {
            "predictions": y_pred,  # [B, H, W] classes entières
            "probabilities": probs,  # [B, C, H, W] probabilités par classe
            "logits": logits,  # [B, C, H, W] logits bruts
            "pre_post_name": batch["pre_post_name"],
            "cell_id": batch["cell_id"],
            "profile": batch["profile"],  # profil rasterio pour écriture GeoTIFF
            "original_height": batch["original_height"],
            "original_width": batch["original_width"],
        }

        # --- Propager les métadonnées optionnelles (event_id, db_nbac_fire_id, etc.) ---
        for key in ("pair_id",
                    "event_id",
                    "db_nbac_fire_id",
                    'group_date_pre',
                    'group_date_post',
                    'group_id_pre',
                    'group_id_post'):
            if key in batch:
                result[key] = batch[key]

        if batch.get("has_mask", torch.tensor(False)).any():
            result["mask"] = batch["mask"]

        return result

    def on_predict_end(self) -> None:
        """Appelé après que tous les predict_step soient terminés.

        Structure de sortie :
            output_dir / predictions / EVENT_ID / PREDICTION_DATE / cell_id / image.tif
            output_dir / predictions / EVENT_ID / PREDICTION_DATE / merged.tif
        """
        from collections import defaultdict
        predictions = self.trainer.predict_loop.predictions
        if not predictions:
            logger.warning("No predictions to save.")
            return

        # --- Base output directory ---
        predict_date = datetime.now().strftime("%Y%m%d_%H%M")
        logger.info(f"Saving predictions to -- {self.predict_output_dir}")
        if self.predict_output_dir is not None:
            base_dir = Path(self.predict_output_dir)
            if base_dir.name != "predictions":
                base_dir = base_dir / "predictions"
        else:
            base_dir = Path(self.trainer.default_root_dir) / "predictions"

        # --- Phase 1 : écrire chaque tuile individuelle ---
        # On collecte les chemins par (event_id, predict_date) pour le merge
        group_tile_paths: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
        event_all_tile_paths: dict[str, list[Path]] = defaultdict(list)

        logger.info(f"Saving predictions to {base_dir}")
        base_dir.mkdir(parents=True, exist_ok=True)

        # --- Écrire le manifeste JSON pour l'ingestion DB ---
        manifest = {
            "prediction_date": predict_date,
            "model_name": self.change_detection_model,
            "checkpoint": str(self.weights_from_checkpoint_path or ""),
            "base_dir": str(base_dir),
            "predictions": [],
        }

        for batch_result in predictions:
            batch_pair_ids = batch_result.get("pair_id")
            batch_cell_id = batch_result['cell_id']
            y_pred = batch_result["predictions"]  # [B, H_padded, W_padded]
            names = batch_result["pre_post_name"]
            batch_profiles = batch_result["profile"]
            orig_heights = batch_result["original_height"]  # Tensor [B] ou list
            orig_widths = batch_result["original_width"]  # Tensor [B] ou list
            batch_size = y_pred.shape[0]
            # event_id : peut être un Tensor, une list, ou absent
            batch_event_ids = batch_result.get("event_id")
            # Fallback pour le training dataset qui a db_nbac_fire_id
            if batch_event_ids is None:
                batch_event_ids = batch_result.get("db_nbac_fire_id")

            batch_group_id_pre = batch_result.get("group_id_pre")
            batch_group_id_post = batch_result.get("group_id_post")
            batch_group_date_pre = batch_result.get("group_date_pre")
            batch_group_date_post = batch_result.get("group_date_post")

            for i in range(batch_size):
                cell_id = batch_cell_id[i]
                sample_name = names[i].replace('\n', '').replace('|', '_').replace('/', '_')
                pair_id = self._extract_scalar(batch_pair_ids, i, default=None)
                event_id = self._extract_scalar(batch_event_ids, i, default="unknown_event")
                group_id_pre = self._extract_scalar(batch_group_id_pre, i, default="all")
                group_id_post = self._extract_scalar(batch_group_id_post, i, default="all")
                group_date_pre = self._extract_scalar(batch_group_date_pre, i, default="all")
                group_date_post = self._extract_scalar(batch_group_date_post, i, default="all")

                # --- Récupérer les dimensions originales ---
                orig_h = orig_heights[i].item() if isinstance(orig_heights, torch.Tensor) else int(orig_heights[i])
                orig_w = orig_widths[i].item() if isinstance(orig_widths, torch.Tensor) else int(orig_widths[i])

                # --- Découper le padding (crop au coin supérieur-gauche) ---
                pred_np = y_pred[i, :orig_h, :orig_w].cpu().numpy().astype(np.uint16)

                # --- Reconstruire le profil rasterio ---
                crs_val = batch_profiles["crs"][i] if isinstance(batch_profiles["crs"], (list, tuple)) else \
                batch_profiles["crs"]

                transform_raw = batch_profiles["transform"]

                t_list = [transform_raw[k][i].item() for k in range(6)]

                print(t_list)

                profile_i = {
                    "driver": "GTiff",
                    "dtype": "uint16",
                    "count": 1,
                    "nodata": 32767,
                    "height": orig_h,  # ← dimensions ORIGINALES, pas paddées
                    "width": orig_w,  # ← dimensions ORIGINALES, pas paddées
                    "crs": crs_val,
                    "transform": Affine(*t_list),
                }
                # --- Chemin : base / EVENT_ID / PREDICTION_DATE / cell_id / image.tif ---
                event_date_dir = base_dir / event_id / predict_date
                tile_dir = event_date_dir / cell_id
                tile_dir.mkdir(parents=True, exist_ok=True)
                out_path = tile_dir / f"{pair_id}-{sample_name}.tif"

                with rio.open(str(out_path), "w", **profile_i) as dst:
                    dst.write(pred_np[np.newaxis, :, :])

                # Collecter pour les merges (APRÈS le with)
                event_date_key = str(event_date_dir)
                group_tile_paths[(event_date_key, str(group_id_pre), str(group_id_post))].append(out_path)
                event_all_tile_paths[event_date_key].append(out_path)

                logger.info("Saved prediction to %s (%dx%d)", out_path, orig_w, orig_h)

                manifest["predictions"].append({
                    "pair_id": pair_id,
                    "event_id": event_id,
                    "cell_id": cell_id,
                    "group_id_pre": group_id_pre,
                    "group_id_post": group_id_post,
                    "group_date_pre": group_date_pre,
                    "group_date_post": group_date_post,
                    "tif_path": str(out_path),
                })

                manifest_path = base_dir / "manifest.json"
                import json
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2, default=str)
                logger.info("Saved prediction manifest to %s", manifest_path)

        self._merge_predictions(group_tile_paths, event_all_tile_paths)
        logger.info("All predictions saved to %s", base_dir)

    @staticmethod
    def _extract_scalar(batch_field, index: int, default: str = "unknown") -> str:
        """Extract a scalar string value from a batched field at position index."""
        if batch_field is None:
            return default
        if isinstance(batch_field, torch.Tensor):
            return str(batch_field[index].item())
        if isinstance(batch_field, (list, tuple)):
            return str(batch_field[index])
        return str(batch_field)

    @staticmethod
    def _merge_predictions(
            group_tile_paths: dict[tuple[str, str, str], list[Path]],
            event_all_tile_paths: dict[str, list[Path]],
    ) -> None:
        """Merge tiles in two passes:
        1. Per group_id_pre/group_id_post pair → merged_group_{pre}_{post}.tif
        2. All tiles in the event/date dir    → merged_all.tif
        """
        from rasterio.merge import merge as rio_merge

        # --- Pass 1 : merge par paire (group_id_pre, group_id_post) ---
        for (event_date_dir_str, group_pre, group_post), tile_paths in group_tile_paths.items():
            event_date_dir = Path(event_date_dir_str)
            if len(tile_paths) < 2:
                # Rien à merger s'il n'y a qu'une seule tuile
                logger.info("Skipping merge for group %s/%s (only %d tile)",
                            group_pre, group_post, len(tile_paths))
                continue

            merged_name = f"merged_group_{group_pre}_{group_post}.tif"
            logger.info("Merging %d tiles → %s/%s", len(tile_paths), event_date_dir, merged_name)

            datasets_to_merge = []
            try:
                datasets_to_merge = [rio.open(str(p)) for p in tile_paths]
                mosaic, mosaic_transform = rio_merge(datasets_to_merge)

                merge_profile = datasets_to_merge[0].profile.copy()
                merge_profile.update({
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": mosaic_transform,
                })

                merged_path = event_date_dir / merged_name
                with rio.open(str(merged_path), "w", **merge_profile) as dst:
                    dst.write(mosaic)

                logger.info("Saved merged group to %s (%dx%d)",
                            merged_path, mosaic.shape[2], mosaic.shape[1])

            except Exception:
                logger.exception("Failed to merge group %s/%s in %s",
                                 group_pre, group_post, event_date_dir)
            finally:
                for ds in datasets_to_merge:
                    try:
                        ds.close()
                    except Exception:
                        pass

        # --- Pass 2 : merge global par EVENT_ID / PREDICTION_DATE ---
        for event_date_dir_str, tile_paths in event_all_tile_paths.items():
            event_date_dir = Path(event_date_dir_str)
            if len(tile_paths) < 2:
                logger.info("Skipping global merge for %s (only %d tile)",
                            event_date_dir, len(tile_paths))
                continue

            logger.info("Merging all %d tiles → %s/merged_all.tif", len(tile_paths), event_date_dir)

            datasets_to_merge = []
            try:
                datasets_to_merge = [rio.open(str(p)) for p in tile_paths]
                mosaic, mosaic_transform = rio_merge(datasets_to_merge)

                merge_profile = datasets_to_merge[0].profile.copy()
                merge_profile.update({
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": mosaic_transform,
                })

                merged_path = event_date_dir / "merged_all.tif"
                with rio.open(str(merged_path), "w", **merge_profile) as dst:
                    dst.write(mosaic)

                logger.info("Saved global merge to %s (%dx%d)",
                            merged_path, mosaic.shape[2], mosaic.shape[1])

            except Exception:
                logger.exception("Failed to create global merge in %s", event_date_dir)
            finally:
                for ds in datasets_to_merge:
                    try:
                        ds.close()
                    except Exception:
                        pass
