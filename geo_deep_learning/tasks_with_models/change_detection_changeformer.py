"""Change Detection with ChangeFormer model for RCM SAR data."""

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
IGNORE_MASK_INDEX = 255

# Deep supervision weights for ChangeFormer's 5 output heads (c4→c1→final).
# Intermediate heads get decreasing weight; the final head gets the most.
DEEP_SUPERVISION_WEIGHTS = [0.1, 0.1, 0.15, 0.2, 1.0]

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
            deep_supervision: bool = True,
            deep_supervision_weights: list[float] | None = None,
            speckle_noise_std: float = 0.15,
            use_metadata_film: bool = True,
            film_embed_dim: int = 32,
            use_cbam: bool = False,
            cbam_reduction: int = 4,
            use_channel_dropout: bool = False,
            channel_dropout_prob: float = 0.1,
            use_dfa: bool = False,
            dfa_gate_hidden: int = 16,
            **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Initialize the model.

        Args:
            deep_supervision: Use all ChangeFormer decoder heads for loss computation.
                The decoder produces 5 outputs (4 intermediate + 1 final); when enabled,
                losses from intermediate heads are weighted and summed.
            deep_supervision_weights: Per-head loss weights (length must match decoder
                output count, typically 5). Defaults to DEEP_SUPERVISION_WEIGHTS.
            speckle_noise_std: Std-dev of multiplicative SAR speckle noise augmentation.
                Set to 0 to disable. Only applied during training.
            use_metadata_film: When True, SAT_PASS and BEAM are NOT expected as
                image bands.  Instead they are read from ``batch["sat_pass_value"]``
                and ``batch["beam_value"]`` and used for FiLM conditioning.
                Requires ``separate_metadata=True`` on the DataModule.
            film_embed_dim: Embedding dimension for the FiLM conditioner.
            use_cbam: Apply CBAM (Channel & Spatial Attention) after FiLM.
                Helps focus on relevant bands and spatial regions.
            cbam_reduction: Channel attention reduction ratio for CBAM.
            use_channel_dropout: Randomly drop input channels during training.
                Improves robustness to noisy SAR bands.
            channel_dropout_prob: Probability of dropping each channel.
            use_dfa: Apply Difference Feature Attention on decoder outputs.
                Gates unreliable intermediate predictions.
            dfa_gate_hidden: Hidden dim for DFA gating MLP.
        """
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

        # Deep supervision (use all ChangeFormer decoder heads)
        self.deep_supervision = deep_supervision
        self.deep_supervision_weights = deep_supervision_weights or DEEP_SUPERVISION_WEIGHTS
        self.speckle_noise_std = speckle_noise_std
        self.use_metadata_film = use_metadata_film
        self.film_embed_dim = film_embed_dim

        # Additional conditioning / regularization modules
        self.use_cbam = use_cbam
        self.cbam_reduction = cbam_reduction
        self.use_channel_dropout = use_channel_dropout
        self.channel_dropout_prob = channel_dropout_prob
        self.use_dfa = use_dfa
        self.dfa_gate_hidden = dfa_gate_hidden

        self.changed_num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.labels = (
            [str(i) for i in range(self.changed_num_classes)]
            if class_labels is None
            else class_labels
        )

        # --- Cache augmentation modules (avoid re-creating each batch) ---
        self._geo_aug = self._build_geo_aug()
        self._intensity_aug = self._build_intensity_aug()
        self._pad_aug = AugmentationSequential(
            krn.augmentation.PadTo(
                size=self.image_size,
                pad_mode='constant',
                pad_value=0,
                keepdim=False,
            ),
            data_keys=None,
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

    @staticmethod
    def _build_geo_aug() -> AugmentationSequential:
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

    @staticmethod
    def _build_intensity_aug() -> AugmentationSequential:
        """Intensity augmentations for SAR data (applied to images only).

        Key difference from optical: SAR speckle is *multiplicative*, so the
        primary noise augmentation multiplies by (1 + N(0, std)) instead of
        adding Gaussian noise.  We also keep a small additive Gaussian term and
        random erasing for robustness.
        """
        return AugmentationSequential(
            krn.augmentation.RandomGaussianNoise(mean=0.0, std=0.03, p=0.2, keepdim=True),
            krn.augmentation.RandomGaussianBlur(
                kernel_size=(3, 3), sigma=(0.1, 1.5), p=0.2, keepdim=True
            ),
            krn.augmentation.RandomErasing(
                scale=(0.02, 0.08), ratio=(0.3, 3.3), p=0.2, keepdim=True
            ),
            data_keys=None,
        )

    @staticmethod
    def _apply_speckle_noise(image: Tensor, std: float) -> Tensor:
        """Apply multiplicative speckle noise typical of SAR imagery.

        Speckle in SAR is modelled as multiplicative: I_noisy = I * (1 + ε)
        where ε ~ N(0, std).  This preserves the sign and scale of the signal
        while adding realistic radiometric variation.
        """
        if std <= 0:
            return image
        noise = 1.0 + torch.randn_like(image) * std
        return image * noise

    def on_before_batch_transfer(
            self,
            batch: dict[str, Any],
            dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
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

        transformed = self._pad_aug(keys_to_pad)
        batch.update(transformed)
        return batch

    def configure_model(self) -> None:
        """Configure model."""
        # Define metadata fields for FiLM conditioning
        film_metadata_fields = None
        if self.use_metadata_film:
            film_metadata_fields = {
                "sat_pass": 2,       # ASC / DESC
                "beam": 4,           # A / B / C / D
                "pre_season": 13,     # 13 , month number + 0 if undefined
                "post_season": 13,    # 13 , month number + 0 if undefined
                "time_delta": 5,     # 0-4d / 4-12d / 12-24d / 24-48d / 48d+
                "processing_year" : 3   #0 : undefined, 1: 2023, 2: <> 2023
            }

        self.model = ChangeDetectionModel(
            change_detection_model=self.change_detection_model,
            in_channels=self.in_channels,
            out_channels=self.num_classes + 1 if self.num_classes == 1 else self.num_classes,
            use_metadata_film=self.use_metadata_film,
            film_embed_dim=self.film_embed_dim,
            film_metadata_fields=film_metadata_fields,
            use_cbam=self.use_cbam,
            cbam_reduction=self.cbam_reduction,
            use_channel_dropout=self.use_channel_dropout,
            channel_dropout_prob=self.channel_dropout_prob,
            use_dfa=self.use_dfa,
            dfa_gate_hidden=self.dfa_gate_hidden,
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

    def forward(
            self,
            image_pre: Tensor,
            image_post: Tensor,
            sat_pass: Tensor | None = None,
            beam: Tensor | None = None,
            **metadata_kwargs: Tensor,
    ) -> Tensor | list[Tensor]:
        """Forward pass.

        Args:
            image_pre: Pre-event image [B, C, H, W].
            image_post: Post-event image [B, C, H, W].
            sat_pass: [B] satellite pass index (only when use_metadata_film=True).
            beam: [B] beam index (only when use_metadata_film=True).
            **metadata_kwargs: Additional FiLM metadata (e.g. season=[B]).

        Returns:
            When ``deep_supervision`` is enabled **and** the model is in
            training/validation mode, returns the full list of decoder outputs
            (one per scale + final).  Otherwise returns only the final
            prediction tensor.
        """
        outputs = self.model(image_pre, image_post, sat_pass=sat_pass, beam=beam, **metadata_kwargs)
        # ChangeFormer decoder returns a list: [p_c4, p_c3, p_c2, p_c1, final]
        if self.deep_supervision and self.training:
            return outputs  # list[Tensor]
        return outputs[-1]  # Tensor [B, C, H, W]

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training:
            return batch
        device = batch["image"].device

        # 1. Geometric augmentations on images + masks together
        keys_to_aug = {
            "image_pre": batch["image_pre"],
            "image": batch["image"],
        }
        if "mask-common" in batch:
            keys_to_aug["mask-common"] = batch["mask-common"].to(torch.float32)
        if "mask" in batch:
            keys_to_aug["mask"] = batch["mask"]

        transformed = self._geo_aug(keys_to_aug)
        for key in transformed:
            batch[key] = transformed[key].to(device, non_blocking=True)

        # 2. Intensity augmentations on images only (generic)
        for img_key in ["image_pre", "image"]:
            batch[img_key] = self._intensity_aug({img_key: batch[img_key]})[img_key]

        # 3. SAR-specific: multiplicative speckle noise
        if self.speckle_noise_std > 0:
            for img_key in ["image_pre", "image"]:
                batch[img_key] = self._apply_speckle_noise(
                    batch[img_key], self.speckle_noise_std
                )

        return batch

    # TODO : Modifier pour avoir image pre/post
    def training_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run training step."""
        x_pre, x_post, y, one_hot, logits, main_loss, focal_loss_val, lovasz_loss_val, final_head_loss, batch_size = self._forward_and_get_loss(batch)
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
        self.log("focal_loss", focal_loss_val, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("lovasz_loss", lovasz_loss_val, on_epoch=True, sync_dist=True, batch_size=batch_size)
        # Final-head-only loss: comparable to val_loss (which always uses the final head)
        self.log("train_loss_final_head", final_head_loss, on_epoch=True, sync_dist=True, batch_size=batch_size)

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

    def on_validation_epoch_start(self) -> None:
        """Reset visualization counter at the start of each validation epoch."""
        self._total_samples_visualized = 0

    def validation_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run validation step."""
        has_mask = batch.get("has_mask", torch.tensor([True]))
        if not has_mask.any():
            return None  # skip ce batch
        x_pre, x_post, y, one_hot, logits, main_loss, _focal, _lovasz, _final_head, batch_size = self._forward_and_get_loss(batch)

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

        # --- Visualisations en validation ---
        if self._total_samples_visualized < self.max_samples:
            remaining = self.max_samples - self._total_samples_visualized
            samples_to_visualize = min(remaining, len(x_post))
            self._total_samples_visualized += self._log_visualizations(
                trainer=self.trainer,
                batch=batch,
                outputs=logits,
                max_samples=samples_to_visualize,
                artifact_prefix="val",
                epoch_suffix=True,
            )

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

    def on_test_epoch_start(self) -> None:
        """Reset visualization counter at the start of each test epoch."""
        self._total_samples_visualized = 0

    def test_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run test step."""

        has_mask = batch.get("has_mask", torch.tensor([True]))
        if not has_mask.any():
            return None

        x_pre, x_post, y, one_hot, logits, main_loss, _focal, _lovasz, _final_head, batch_size = self._forward_and_get_loss(batch)
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

    def _forward_and_get_loss(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        x_pre, x_post = batch["image_pre"], batch["image"]
        y = batch["mask"]
        common_data_mask = batch["mask-common"]

        batch_size = x_post.shape[0]

        # --- Mask out the neighbour-context buffer zone (train_overlap_buffer) ---
        # When spatial context is loaded from adjacent cells the image is
        # (h + 2b) × (w + 2b) but the ground-truth label mask only covers
        # the central (h × w) region.  Zero the buffer zone in common_data_mask
        # so the loss is never computed there.
        # The dataset stores "buffer_size", "cell_orig_height", "cell_orig_width"
        # whenever a non-zero buffer was applied (train or predict).
        if "buffer_size" in batch:
            buf_raw = batch["buffer_size"]
            buf = int(buf_raw[0].item() if isinstance(buf_raw, torch.Tensor) else buf_raw[0])
            if buf > 0:
                common_data_mask = common_data_mask.clone()

                if "tile_row_start" in batch:
                    # --- Tiled + buffered ---
                    # Each tile is a crop of the expanded (cell + 2*buf)
                    # image.  The valid zone (central cell, excluding
                    # neighbour context) spans rows [buf, buf + orig_h) ×
                    # cols [buf, buf + orig_w) in the expanded image.
                    # We compute the intersection of this valid zone with
                    # each tile's coverage area and mask everything outside.
                    # Without this per-tile logic the old code masked buf
                    # pixels from ALL 4 edges of EVERY tile, which wrongly
                    # discarded up to 63 % of valid pixels on interior
                    # tiles that don't touch the buffer boundary at all.
                    tile_h, tile_w = common_data_mask.shape[2], common_data_mask.shape[3]
                    orig_h_batch = batch["cell_orig_height"]
                    orig_w_batch = batch["cell_orig_width"]

                    for i in range(batch_size):
                        tr = int(batch["tile_row_start"][i].item() if isinstance(batch["tile_row_start"], torch.Tensor) else batch["tile_row_start"][i])
                        tc = int(batch["tile_col_start"][i].item() if isinstance(batch["tile_col_start"], torch.Tensor) else batch["tile_col_start"][i])
                        oh = int(orig_h_batch[i].item() if isinstance(orig_h_batch, torch.Tensor) else orig_h_batch[i])
                        ow = int(orig_w_batch[i].item() if isinstance(orig_w_batch, torch.Tensor) else orig_w_batch[i])

                        # Valid rows/cols in tile-local coordinates
                        vr_start = max(0, buf - tr)
                        vr_end   = min(tile_h, buf + oh - tr)
                        vc_start = max(0, buf - tc)
                        vc_end   = min(tile_w, buf + ow - tc)

                        if vr_start > 0:
                            common_data_mask[i, :, :vr_start, :] = 0.0
                        if vr_end < tile_h:
                            common_data_mask[i, :, vr_end:, :] = 0.0
                        if vc_start > 0:
                            common_data_mask[i, :, :, :vc_start] = 0.0
                        if vc_end < tile_w:
                            common_data_mask[i, :, :, vc_end:] = 0.0
                else:
                    # --- Non-tiled + buffered ---
                    # The full expanded image: mask buf pixels from all edges.
                    img_h, img_w = common_data_mask.shape[2], common_data_mask.shape[3]
                    if buf < img_h:
                        common_data_mask[:, :, :buf, :] = 0.0
                        common_data_mask[:, :, img_h - buf:, :] = 0.0
                    if buf < img_w:
                        common_data_mask[:, :, :, :buf] = 0.0
                        common_data_mask[:, :, :, img_w - buf:] = 0.0

                # Propagate the modified mask so that metrics in
                # training_step / validation_step / test_step use the
                # same valid-pixel set as the loss.
                batch["mask-common"] = common_data_mask
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
        min_valid_ratio = 0.05  # au moins 5% de pixels valides
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

        raw_output = self(
            x_pre,
            x_post,
            sat_pass=batch.get("sat_pass_value"),
            beam=batch.get("beam_value"),
            pre_season=batch.get("pre_season"),
            post_season=batch.get("post_season"),
            time_delta=batch.get("time_delta_bin"),
        )

        # Deep supervision: raw_output is a list during training, single Tensor otherwise
        if isinstance(raw_output, list):
            all_outputs = raw_output  # [p_c4, p_c3, p_c2, p_c1, final]
            logits = all_outputs[-1]  # final prediction for metrics
        else:
            all_outputs = [raw_output]
            logits = raw_output

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

            return x_pre, x_post, y.float(), dummy_one_hot, logits_safe, zero_loss, zero_loss, zero_loss, zero_loss, batch_size

        # Préparation du one-hot
        y_one_hot = y.squeeze(1) if y.dim() == 4 else y
        y_one_hot = y_one_hot.clamp(min=0, max=num_classes - 1)  # clamp aussi les 255 → num_classes-1
        one_hot = torch.nn.functional.one_hot(y_one_hot.long(), num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).contiguous().float()

        # Mark invalid pixels with IGNORE_MASK_INDEX in targets.
        # Both FocalLoss and LovaszLoss support ignore_index=255 natively,
        # avoiding the previous mask-multiplication approach which:
        #   - biased Lovász sorting (invalid pixels got error=1, distorting gradients)
        #   - added phantom class-0 contributions to FocalLoss
        invalid_pixels = (common_data_mask < 0.5)  # [B, 1, H, W], True=invalid
        one_hot_for_loss = one_hot.clone()
        one_hot_for_loss.masked_fill_(invalid_pixels.expand_as(one_hot), IGNORE_MASK_INDEX)

        # Vérifier qu'il reste des pixels valides
        valid_sum = common_data_mask.sum()
        if valid_sum == 0:
            zero = torch.tensor(0.0, device=logits_no_nan.device, dtype=logits_no_nan.dtype, requires_grad=True)
            return x_pre, x_post, y_float, one_hot, logits_no_nan, zero, zero, zero, zero, batch_size

        w_ml, w_sl = self.loss_ratio

        # Vérifier entrées de la loss
        if not torch.isfinite(one_hot).all():
            raise RuntimeError("One-hot targets contain non-finite values (NaN/Inf).")

        # --- Deep supervision: compute loss on every decoder head ---
        target_h, target_w = logits_no_nan.shape[2], logits_no_nan.shape[3]

        if self.deep_supervision and len(all_outputs) > 1 and self.training:
            ds_weights = self.deep_supervision_weights
            # Ensure we have a weight for each head
            if len(ds_weights) < len(all_outputs):
                ds_weights = ds_weights + [1.0] * (len(all_outputs) - len(ds_weights))

            total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            total_focal = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            total_lovasz = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            weight_sum = sum(ds_weights[:len(all_outputs)])

            final_head_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            for head_idx, head_output in enumerate(all_outputs):
                head_logits = torch.nan_to_num(head_output, nan=1e15, posinf=1.0, neginf=0.0)
                # Resize intermediate heads to final resolution
                if head_logits.shape[2] != target_h or head_logits.shape[3] != target_w:
                    head_logits = F.interpolate(
                        head_logits, size=(target_h, target_w),
                        mode='bilinear', align_corners=False,
                    )

                head_lovasz = self.secondary_loss(head_logits.contiguous(), one_hot_for_loss)
                head_focal = self.main_loss(head_logits.contiguous(), one_hot_for_loss)
                head_loss = w_sl * head_lovasz + w_ml * head_focal

                total_loss = total_loss + ds_weights[head_idx] * head_loss
                total_focal = total_focal + ds_weights[head_idx] * head_focal
                total_lovasz = total_lovasz + ds_weights[head_idx] * head_lovasz

                # Save final head loss for fair train/val comparison
                if head_idx == len(all_outputs) - 1:
                    final_head_loss = head_loss

            main_loss = total_loss / weight_sum
            focal_loss_val = total_focal / weight_sum
            lovasz_loss_val = total_lovasz / weight_sum
        else:
            # Standard single-head loss (final_head_loss == main_loss)
            lovasz_loss_val = self.secondary_loss(logits_no_nan.contiguous(), one_hot_for_loss)
            focal_loss_val = self.main_loss(logits_no_nan.contiguous(), one_hot_for_loss)
            main_loss = w_sl * lovasz_loss_val + w_ml * focal_loss_val
            final_head_loss = main_loss

        # Burned-class false-negative penalty (uses burned_class_weight)
        burn_penalty = self._burned_false_negative_penalty(logits_no_nan, one_hot, common_data_mask)
        main_loss = main_loss + burn_penalty
        final_head_loss = final_head_loss + burn_penalty

        # Dernière vérification
        if not torch.isfinite(main_loss):
            raise RuntimeError(
                f"Computed loss is NaN/Inf. "
                f"focal_loss={focal_loss_val.detach().cpu().item()}, "
                f"lovasz_loss={lovasz_loss_val.detach().cpu().item()}, "
                f"burn_penalty={burn_penalty.detach().cpu().item()}"
            )

        return x_pre, x_post, y_float, one_hot, logits_no_nan, main_loss, focal_loss_val, lovasz_loss_val, final_head_loss, batch_size

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
            num_bands = image_batch.shape[1]
            if self.use_metadata_film:
                # No metadata bands in image — all bands are data
                available = list(range(num_bands))
            else:
                # Legacy: skip band 0 (COMMON_MASK) and last 2 (SAT_PASS, BEAM).
                data_band_start = 1  # skip COMMON_MASK
                data_band_end = max(num_bands - 2, data_band_start + 1)  # skip SAT_PASS, BEAM
                available = list(range(data_band_start, data_band_end))
            # Take 3 evenly spaced bands (or fewer if not enough)
            if len(available) >= 3:
                step = max(1, len(available) // 3)
                rgb_indices = [available[0], available[len(available) // 2], available[-1]]
            else:
                rgb_indices = available[:3]

            # Minimum burned pixel ratio to include a sample in visualizations
            min_burned_ratio = 0.10
            for i in range(len(image_batch)):
                if num_logged >= num_samples:
                    break

                # --- Filter: only visualize samples with ≥30% burned pixels ---
                has_real_mask = has_mask_flags[i] if isinstance(
                    has_mask_flags, (list, torch.Tensor)) else has_mask_flags
                if not has_real_mask:
                    continue  # no ground truth → skip

                mask_i_for_filter = mask_batch[i]  # [H, W], values: 0=unburn, 1=burn, 255=ignore
                if common_mask is not None:
                    valid_pixels = (common_mask[i].squeeze(0) > 0.5)  # [H, W]
                else:
                    valid_pixels = (mask_i_for_filter != IGNORE_MASK_INDEX)

                valid_count = valid_pixels.sum()
                if valid_count == 0:
                    continue

                burned_count = ((mask_i_for_filter == 1) & valid_pixels).sum()
                burned_ratio = burned_count.float() / valid_count.float()
                if burned_ratio < min_burned_ratio:
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
                    invalid = (common_mask[i].squeeze(0) < 0.5)  # [H, W]
                    # Use a distinct value (255) for visualization of masked pixels
                    pred = pred.clone()
                    effective_num_classes = self.num_classes + 1 if self.num_classes == 1 else self.num_classes
                    pred[invalid] = effective_num_classes  # = 2 → index du gris dans la colormap

                # Ground truth mask (always present here — filtered above)
                mask_i = mask_batch[i]

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
        """Run prediction step with optional Test-Time Augmentation (TTA).

        TTA applies geometric transforms (flips, 90° rotations), runs inference
        on each, inverts the transform, and averages the softmax probabilities.
        This reduces noise-related false positives — particularly important for
        SAR data where speckle can cause spurious detections.
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
            )

        # Convertir en probabilités et en classes prédites
        if self.num_classes == 1:
            # Binaire : 2 classes (0=no-change, 1=change)
            probs = torch.softmax(logits, dim=1)  # [B, 2, H, W]
            y_pred = torch.argmax(probs, dim=1)  # [B, H, W]
        else:
            probs = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(probs, dim=1)

        # --- Exclure les pixels invalides et l'eau avec NO_DATA (32767) ---
        # ``mask-common`` décrit la validité des acquisitions SAR, mais ne
        # contient pas nécessairement l'eau. Apply both masks even when the
        # common mask is available so water can never be vectorized as burn.
        invalid_mask = torch.zeros_like(y_pred, dtype=torch.bool)
        if "mask-common" in batch:
            common_mask = batch["mask-common"]  # [B, 1, H, W] bool ou float
            invalid_mask |= common_mask.squeeze(1) < 0.5
        if "water_mask" in batch:
            water_mask = batch["water_mask"]  # [B, 1, H, W]
            invalid_mask |= water_mask.squeeze(1) > 0  # eau = valeur > 0
        y_pred = y_pred.masked_fill(invalid_mask, NO_DATA)

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
                    'source_width'):
            if key in batch:
                result[key] = batch[key]

        if batch.get("has_mask", torch.tensor(False)).any():
            result["mask"] = batch["mask"]

        return result

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
    # Overlap blending for tile-based prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _create_blend_window(
        height: int,
        width: int,
        overlap_h: int,
        overlap_w: int,
    ) -> np.ndarray:
        """Create a 2D blending window with cosine ramps in overlap regions.

        Pixels in the non-overlapping center get weight 1.0.  Pixels in the
        overlap zone smoothly ramp from 0→1 using a raised-cosine profile,
        ensuring seamless transitions between adjacent tiles.

        Args:
            height: Tile height in pixels.
            width: Tile width in pixels.
            overlap_h: Vertical overlap in pixels (tile_h − stride_h).
            overlap_w: Horizontal overlap in pixels (tile_w − stride_w).

        Returns:
            2D ``float32`` array of shape ``[height, width]`` with values in ``(0, 1]``.
        """

        def _ramp(size: int, overlap: int) -> np.ndarray:
            win = np.ones(size, dtype=np.float32)
            if overlap > 0:
                ramp_vals = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
                ramp_vals = 0.5 * (1.0 - np.cos(np.pi * ramp_vals))
                win[:overlap] = ramp_vals
                win[-overlap:] = ramp_vals[::-1]
            return win

        win_h = _ramp(height, overlap_h)
        win_w = _ramp(width, overlap_w)
        window = np.outer(win_h, win_w)
        return np.maximum(window, 1e-6).astype(np.float32)

    def _reassemble_overlapping_tiles(
        self,
        predictions: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], bool]:
        """Reassemble overlapping tiles into source-level predictions with cosine blending.

        Detects whether tiling with overlap was used.  If so, groups tiles by
        source image (using ``pre_post_name`` minus the tile suffix) and blends
        their softmax probabilities with a 2D cosine window to eliminate tile
        seam artefacts.

        Args:
            predictions: List of batch prediction dicts from :meth:`predict_step`.

        Returns:
            Tuple of:
            - Dict mapping *source_key* → assembled prediction info (numpy arrays,
              GeoTIFF profile, metadata scalars).
            - ``True`` if overlap blending was applied, ``False`` otherwise.
        """
        from collections import defaultdict

        # --- Check if tiling metadata is present ---
        has_tiles = any("tile_row_start" in batch for batch in predictions)
        if not has_tiles:
            return {}, False

        dm = self.trainer.datamodule
        tile_size = getattr(dm, "tile_size", None)
        tile_stride = getattr(dm, "tile_stride", None)
        if tile_size is None or tile_stride is None:
            return {}, False

        tile_h, tile_w = tile_size
        stride_h, stride_w = tile_stride
        overlap_h = max(tile_h - stride_h, 0)
        overlap_w = max(tile_w - stride_w, 0)

        if overlap_h <= 0 and overlap_w <= 0:
            return {}, False  # No overlap → skip blending

        logger.info(
            "Overlap detected: tile=%s, stride=%s, overlap=(%d, %d). "
            "Reassembling tiles with cosine blending…",
            tile_size, tile_stride, overlap_h, overlap_w,
        )

        # --- Group tiles by source image ---
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for batch_result in predictions:
            names = batch_result["pre_post_name"]
            batch_size = len(names)

            for i in range(batch_size):
                name = names[i].replace("\n", "")
                source_key = name.split("|tile_")[0] if "|tile_" in name else name

                tile_info: dict[str, Any] = {
                    "name": name,
                    "probabilities": batch_result["probabilities"][i].cpu(),
                }

                # Dimensions (original = after tile crop, before padding)
                for dim_key in ("original_height", "original_width"):
                    v = batch_result[dim_key]
                    tile_info[dim_key] = v[i].item() if isinstance(v, torch.Tensor) else int(v[i])

                # Tile position & source dimensions
                for key in ("tile_row_start", "tile_col_start", "source_height", "source_width"):
                    if key in batch_result:
                        v = batch_result[key]
                        tile_info[key] = v[i].item() if isinstance(v, torch.Tensor) else int(v[i])

                # Scalar metadata
                for key in ("cell_id", "pair_id", "event_id", "db_nbac_fire_id",
                            "event_start_date", "event_end_date", "beam", "sat_pass", "output_name",
                            "group_date_pre", "group_date_post",
                            "group_id_pre", "group_id_post"):
                    if key in batch_result:
                        tile_info[key] = self._extract_scalar(batch_result[key], i, default="unknown")

                # Profile (GeoTIFF) — extract per-sample values from the collated profile.
                # PyTorch's default_collate *transposes* a Python list of length N:
                # a list of 9-element transform lists becomes a list of 9 tensors each
                # of shape (batch_size,).  So pv[k][i] yields the k-th coefficient of
                # the i-th sample — NOT pv[i] which would give the i-th coefficient
                # across all samples.  Tensors (from numpy-backed values) are batched
                # normally as (N, 9) and can be indexed with pv[i] directly.
                profile_raw: dict[str, Any] = {}
                for pk, pv in batch_result["profile"].items():
                    if pk == "transform":
                        if isinstance(pv, (list, tuple)):
                            # Transposed list: pv[k] is a tensor of shape (batch_size,)
                            # representing the k-th transform coefficient for all samples.
                            profile_raw[pk] = [
                                float(pv[k][i].item() if isinstance(pv[k], torch.Tensor)
                                      else pv[k][i])
                                for k in range(len(pv))
                            ]
                        elif isinstance(pv, torch.Tensor):
                            # Stacked (N, 9) tensor: pv[i] is the i-th sample's transform.
                            profile_raw[pk] = pv[i].tolist()
                        else:
                            profile_raw[pk] = pv
                    elif isinstance(pv, (list, tuple)):
                        profile_raw[pk] = pv[i]
                    elif isinstance(pv, torch.Tensor):
                        profile_raw[pk] = pv[i]
                    else:
                        profile_raw[pk] = pv
                tile_info["profile_raw"] = profile_raw

                # Common mask for NO_DATA
                if "mask_common" in batch_result:
                    tile_info["mask_common"] = batch_result["mask_common"][i].cpu()
                if "water_mask" in batch_result:
                    tile_info["water_mask"] = batch_result["water_mask"][i].cpu()

                groups[source_key].append(tile_info)

        # --- Reassemble each source image ---
        assembled: dict[str, dict[str, Any]] = {}
        blend_window_cache: dict[tuple[int, int], np.ndarray] = {}

        for source_key, tiles in groups.items():
            # Single-tile source without tile metadata → leave for per-tile path
            if len(tiles) == 1 and "tile_row_start" not in tiles[0]:
                continue

            source_h = tiles[0].get("source_height", tiles[0]["original_height"])
            source_w = tiles[0].get("source_width", tiles[0]["original_width"])
            num_classes = tiles[0]["probabilities"].shape[0]

            prob_accum = np.zeros((num_classes, source_h, source_w), dtype=np.float64)
            weight_accum = np.zeros((source_h, source_w), dtype=np.float64)
            mask_accum = np.zeros((source_h, source_w), dtype=np.float32)
            water_accum = np.zeros((source_h, source_w), dtype=bool)

            for tile in tiles:
                r = tile.get("tile_row_start", 0)
                c = tile.get("tile_col_start", 0)
                th = tile["original_height"]
                tw = tile["original_width"]

                # Blend window (cached by tile dimensions)
                win_key = (th, tw)
                if win_key not in blend_window_cache:
                    blend_window_cache[win_key] = self._create_blend_window(
                        th, tw, overlap_h, overlap_w,
                    )
                win = blend_window_cache[win_key]

                probs = tile["probabilities"].numpy()[:, :th, :tw]
                prob_accum[:, r:r + th, c:c + tw] += probs * win[np.newaxis, :, :]
                weight_accum[r:r + th, c:c + tw] += win

                # Combine common masks (OR logic: valid in any tile = valid)
                if "mask_common" in tile:
                    cm = tile["mask_common"].numpy()
                    if cm.ndim == 3:
                        cm = cm.squeeze(0)
                    mask_accum[r:r + th, c:c + tw] = np.maximum(
                        mask_accum[r:r + th, c:c + tw], cm[:th, :tw],
                    )
                if "water_mask" in tile:
                    wm = tile["water_mask"].numpy()
                    if wm.ndim == 3:
                        wm = wm.squeeze(0)
                    water_accum[r:r + th, c:c + tw] |= wm[:th, :tw] > 0

            # Normalize blended probabilities
            weight_accum = np.maximum(weight_accum, 1e-8)
            blended_probs = (prob_accum / weight_accum[np.newaxis, :, :]).astype(np.float32)

            # Final class prediction
            pred = np.argmax(blended_probs, axis=0).astype(np.uint16)

            # Re-apply NO_DATA mask
            invalid = (mask_accum < 0.5) | water_accum
            pred[invalid] = NO_DATA

            # Build source-level GeoTIFF profile
            first_tile = tiles[0]
            source_profile = self._build_source_profile(first_tile, source_h, source_w)

            assembled[source_key] = {
                "predictions": pred,
                "probabilities": blended_probs,
                "source_height": source_h,
                "source_width": source_w,
                "profile": source_profile,
                "cell_id": first_tile.get("cell_id", "unknown"),
                "pair_id": first_tile.get("pair_id"),
                "event_id": first_tile.get(
                    "event_id",
                    first_tile.get("db_nbac_fire_id", "unknown_event"),
                ),
                "event_start_date": first_tile.get("event_start_date"),
                "event_end_date": first_tile.get("event_end_date"),
                "beam": first_tile.get("beam"),
                "sat_pass": first_tile.get("sat_pass"),
                "output_name": first_tile.get("output_name"),
                "group_id_pre": first_tile.get("group_id_pre", "all"),
                "group_id_post": first_tile.get("group_id_post", "all"),
                "group_date_pre": first_tile.get("group_date_pre", "all"),
                "group_date_post": first_tile.get("group_date_post", "all"),
                "pre_post_name": source_key,
            }

        logger.info(
            "Reassembled %d source images from overlapping tiles.",
            len(assembled),
        )
        return assembled, bool(assembled)

    @staticmethod
    def _build_source_profile(
        tile_info: dict[str, Any],
        source_h: int,
        source_w: int,
    ) -> dict[str, Any]:
        """Reconstruct the full source image's GeoTIFF profile from a tile's profile.

        Inverts the tile-level transform translation so the saved GeoTIFF
        covers the original spatial extent.
        """
        from rasterio.crs import CRS as RioCRS

        raw_profile = tile_info["profile_raw"]
        r = tile_info.get("tile_row_start", 0)
        c = tile_info.get("tile_col_start", 0)

        # Recover transform coefficients — support list/tuple (9 or 6 elements),
        # 1-D tensor, numpy array, dict with int keys, or Affine object.
        transform_raw = raw_profile["transform"]
        if isinstance(transform_raw, dict):
            try:
                t_list = [transform_raw[k] for k in range(6)]
            except KeyError:
                t_list = [transform_raw.get(k, 0.0) for k in ("a", "b", "c", "d", "e", "f")]
        elif isinstance(transform_raw, (list, tuple)):
            t_list = list(transform_raw)
        else:
            # tensor, numpy array, Affine, …
            t_list = list(transform_raw)
        t_list = [t.item() if isinstance(t, torch.Tensor) else float(t) for t in t_list]

        if len(t_list) < 6:
            raise ValueError(
                f"Cannot reconstruct source profile: transform has only {len(t_list)} "
                f"element(s) — need ≥ 6. "
                f"type={type(transform_raw).__name__!r}, raw={transform_raw!r}"
            )

        # Undo tile translation: source_transform = tile_transform * translation(−c, −r)
        tile_transform = Affine(*t_list[:6])
        source_transform = tile_transform * Affine.translation(-c, -r)

        # Parse CRS
        crs_val = raw_profile.get("crs")
        try:
            crs_obj = RioCRS.from_user_input(crs_val) if crs_val else RioCRS.from_epsg(3979)
        except Exception:
            crs_obj = RioCRS.from_epsg(3979)

        return {
            "driver": "GTiff",
            "dtype": "uint16",
            "count": 1,
            "nodata": 32767,
            "height": source_h,
            "width": source_w,
            "crs": crs_obj,
            "transform": source_transform,
        }

    def _save_assembled_predictions(
        self,
        assembled: dict[str, dict[str, Any]],
        base_dir: Path,
        predict_date: str,
    ) -> None:
        """Save overlap-blended source-level predictions as GeoTIFFs and merge.

        Mirrors the structure of the per-tile path in :meth:`on_predict_end`:
        individual GeoTIFFs → manifest JSON → group merge → global merge.
        """
        from collections import defaultdict
        import json

        group_tile_paths: dict[tuple[str, ...], list[Path]] = defaultdict(list)
        event_all_tile_paths: dict[str, list[Path]] = defaultdict(list)

        manifest = {
            "prediction_date": predict_date,
            "model_name": self.change_detection_model,
            "checkpoint": str(self.weights_from_checkpoint_path or ""),
            "base_dir": str(base_dir),
            "overlap_blended": True,
            "predictions": [],
        }

        for source_key, info in assembled.items():
            pred_np = info["predictions"]  # [H, W] uint16
            profile_i = info["profile"]
            cell_id = str(info["cell_id"])
            pair_id = info.get("pair_id")
            event_id = str(info.get("event_id", "unknown_event"))
            group_id_pre = str(info.get("group_id_pre", "all"))
            group_id_post = str(info.get("group_id_post", "all"))
            group_date_pre = str(info.get("group_date_pre", "all"))
            group_date_post = str(info.get("group_date_post", "all"))
            event_start_date = info.get("event_start_date")
            event_end_date = info.get("event_end_date")
            beam = info.get("beam")
            sat_pass = info.get("sat_pass")
            safe_name = Path(source_key.replace("|", "_").replace("/", "_")).stem

            # Directory: base / EVENT_ID / PREDICTION_DATE / cell_id
            event_date_dir = base_dir / event_id / predict_date
            tile_dir = event_date_dir / cell_id
            tile_dir.mkdir(parents=True, exist_ok=True)

            out_name = self._prediction_output_filename(
                info.get("output_name"), pair_id=pair_id, legacy_name=safe_name,
            )
            out_path = tile_dir / out_name

            with rio.open(str(out_path), "w", **profile_i) as dst:
                dst.write(pred_np[np.newaxis, :, :])

            logger.info(
                "Saved blended prediction to %s (%dx%d)",
                out_path, pred_np.shape[1], pred_np.shape[0],
            )

            # Collect for merge
            event_date_key = str(event_date_dir)
            group_tile_paths[self._group_merge_key(
                event_date_key,
                event_id,
                event_start_date,
                event_end_date,
                group_id_pre,
                group_date_pre,
                group_id_post,
                group_date_post,
                beam,
                sat_pass,
            )].append(out_path)
            event_all_tile_paths[event_date_key].append(out_path)

            manifest["predictions"].append({
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
                "overlap_blended": True,
            })

        # Write manifest
        manifest_path = base_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info("Saved prediction manifest to %s", manifest_path)

        # Merge across cells / groups (same logic as per-tile path)
        self._merge_predictions(group_tile_paths, event_all_tile_paths)
        logger.info("All blended predictions saved to %s", base_dir)

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

        base_dir.mkdir(parents=True, exist_ok=True)

        # --- Try overlap-based tile reassembly (cosine blending) ---
        assembled, used_blending = self._reassemble_overlapping_tiles(predictions)
        if used_blending and assembled:
            logger.info(
                "Using overlap blending for %d source images.", len(assembled),
            )
            self._save_assembled_predictions(assembled, base_dir, predict_date)
            return

        # --- Phase 1 : écrire chaque tuile individuelle ---
        # On collecte les chemins par (event_id, predict_date) pour le merge
        group_tile_paths: dict[tuple[str, ...], list[Path]] = defaultdict(list)
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
            batch_output_names = batch_result.get("output_name")
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
            batch_event_start_dates = batch_result.get("event_start_date")
            batch_event_end_dates = batch_result.get("event_end_date")
            batch_beams = batch_result.get("beam")
            batch_sat_passes = batch_result.get("sat_pass")

            for i in range(batch_size):
                cell_id = batch_cell_id[i]
                sample_name = names[i].replace('\n', '').replace('|', '_').replace('/', '_')
                pair_id = self._extract_scalar(batch_pair_ids, i, default=None)
                event_id = self._extract_scalar(batch_event_ids, i, default="unknown_event")
                event_start_date = self._extract_scalar(batch_event_start_dates, i, default=None)
                event_end_date = self._extract_scalar(batch_event_end_dates, i, default=None)
                beam = self._extract_scalar(batch_beams, i, default=None)
                sat_pass = self._extract_scalar(batch_sat_passes, i, default=None)
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
                # Parse CRS back; default to EPSG:3979 if empty/invalid
                from rasterio.crs import CRS as RioCRS
                try:
                    crs_obj = RioCRS.from_user_input(crs_val) if crs_val else RioCRS.from_epsg(3979)
                except Exception:
                    logger.warning("Could not parse CRS '%s' for sample %d — using default EPSG:3979.", crs_val, i)
                    crs_obj = RioCRS.from_epsg(3979)

                transform_raw = batch_profiles["transform"]

                t_list = [transform_raw[k][i].item() for k in range(6)]

                logger.debug("Transform coefficients: %s", t_list)

                profile_i = {
                    "driver": "GTiff",
                    "dtype": "uint16",
                    "count": 1,
                    "nodata": 32767,
                    "height": orig_h,  # ← dimensions ORIGINALES, pas paddées
                    "width": orig_w,  # ← dimensions ORIGINALES, pas paddées
                    "crs": crs_obj,
                    "transform": Affine(*t_list),
                }
                # --- Chemin : base / EVENT_ID / PREDICTION_DATE / cell_id / image.tif ---
                event_date_dir = base_dir / event_id / predict_date
                tile_dir = event_date_dir / cell_id
                tile_dir.mkdir(parents=True, exist_ok=True)
                out_name = self._prediction_output_filename(
                    self._extract_scalar(batch_output_names, i, default=""),
                    pair_id=pair_id,
                    legacy_name=f"cell-{cell_id}_{sample_name}",
                )
                out_path = tile_dir / out_name

                with rio.open(str(out_path), "w", **profile_i) as dst:
                    dst.write(pred_np[np.newaxis, :, :])

                # Collecter pour les merges (APRÈS le with)
                event_date_key = str(event_date_dir)
                group_tile_paths[self._group_merge_key(
                    event_date_key,
                    event_id,
                    event_start_date,
                    event_end_date,
                    group_id_pre,
                    group_date_pre,
                    group_id_post,
                    group_date_post,
                    beam,
                    sat_pass,
                )].append(out_path)
                event_all_tile_paths[event_date_key].append(out_path)

                logger.info("Saved prediction to %s (%dx%d)", out_path, orig_w, orig_h)

                manifest["predictions"].append({
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
                })

        # Write manifest once after all batches are processed
        manifest_path = base_dir / "manifest.json"
        import json
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info("Saved prediction manifest to %s", manifest_path)

        self._merge_predictions(group_tile_paths, event_all_tile_paths)
        logger.info("All predictions saved to %s", base_dir)

    @staticmethod
    def _group_merge_key(
        event_date_dir: str,
        event_id: object,
        event_start_date: object,
        event_end_date: object,
        group_id_pre: object,
        group_date_pre: object,
        group_id_post: object,
        group_date_post: object,
        beam: object,
        sat_pass: object,
    ) -> tuple[str, ...]:
        """Return the complete provenance key for one cross-cell merge."""
        return tuple(map(str, (
            event_date_dir,
            event_id,
            event_start_date,
            event_end_date,
            group_id_pre,
            group_date_pre,
            group_id_post,
            group_date_post,
            beam,
            sat_pass,
        )))

    @staticmethod
    def _merge_date(value: object) -> str:
        """Normalize a date-like metadata value to ``YYYYMMDD`` for filenames."""
        value_as_string = str(value).strip()
        digits = "".join(char for char in value_as_string if char.isdigit())
        return digits[:8] if len(digits) >= 8 else "NA"

    @classmethod
    def _merged_group_filename(
        cls,
        event_id: str,
        event_start_date: str,
        event_end_date: str,
        group_id_pre: str,
        group_date_pre: str,
        group_id_post: str,
        group_date_post: str,
        beam: str,
        sat_pass: str,
    ) -> str:
        """Build the cross-cell merge name without a ``cell_id`` component."""
        return (
            f"event-{event_id}"
            f"_start-{cls._merge_date(event_start_date)}"
            f"_end_{cls._merge_date(event_end_date)}"
            f"_pre-g{group_id_pre}-{cls._merge_date(group_date_pre)}"
            f"_post-g{group_id_post}-{cls._merge_date(group_date_post)}"
            f"_beam-{beam}_pass-{sat_pass}.tif"
        )

    @staticmethod
    def _prediction_output_filename(
        output_name: str | None,
        *,
        pair_id: str | None,
        legacy_name: str,
        suffix: str = "",
    ) -> str:
        """Return a safe GeoTIFF filename, preferring the CSV output name.

        ``output_name`` is supplied by the SCANFIRE orchestrator and contains
        the event, group dates/IDs, cell, beam, and satellite pass.  The
        legacy fallback preserves compatibility with older prediction CSVs.
        The standard SCANFIRE name already includes ``cell-{cell_id}``; the
        legacy fallback receives the cell identifier from the caller as well.
        """
        if output_name:
            requested = Path(str(output_name)).name
            if requested.lower().endswith(".tif") and requested != ".tif":
                return f"{Path(requested).stem}{suffix}.tif"

        return f"{pair_id}-{legacy_name}{suffix}.tif" if pair_id else f"{legacy_name}{suffix}.tif"

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
    def _safe_merge(datasets, method="average"):
        """Merge raster datasets with averaging support for all rasterio versions.

        Standard GeoTIFFs are north-up (pixel height < 0). Some rasterio versions
        (e.g. 1.4.0) raise MergeError for these. Workaround: flip to positive pixel
        height in memory, merge, then flip the result back.

        When ``method='average'``, overlapping valid pixels are averaged using
        sum/count (compatible with all rasterio versions, since ``'average'``
        was only added in rasterio ≥ 1.4.x).

        Args:
            datasets: List of rasterio dataset readers to merge.
            method: ``'average'`` (default) averages valid (non-nodata) pixels.
                Any other value (``'first'``, ``'last'``, ``'min'``, ``'max'``)
                is passed directly to ``rasterio.merge``.
        """
        from rasterio.merge import merge as rio_merge
        from rasterio.transform import Affine
        from rasterio import MemoryFile
        import numpy as np

        if method == "average":
            return ChangeDetectionChangeFormer._merge_average(datasets)

        try:
            return rio_merge(datasets, method=method)
        except Exception as e:
            if "negative pixel height" not in str(e):
                raise
            return ChangeDetectionChangeFormer._merge_with_flip(
                datasets, method=method,
            )

    @staticmethod
    def _merge_average(datasets):
        """Merge datasets by averaging overlapping valid pixels.

        Uses two passes of ``rasterio.merge`` with ``method='sum'`` and
        ``method='count'`` to compute the average.  Falls back to the
        flip workaround if the rasterio version rejects negative pixel height.
        """
        from rasterio.merge import merge as rio_merge
        import numpy as np

        try:
            mosaic_sum, transform = rio_merge(datasets, method="sum")
            # rasterio.DatasetReader is random-access; no seek needed before the
            # second pass.
            mosaic_count, _ = rio_merge(datasets, method="count")
        except Exception as e:
            if "negative pixel height" not in str(e):
                raise
            mosaic_sum, transform = ChangeDetectionChangeFormer._merge_with_flip(
                datasets, method="sum",
            )
            mosaic_count, _ = ChangeDetectionChangeFormer._merge_with_flip(
                datasets, method="count",
            )

        # Average: sum / count, avoiding division by zero
        mask_no_coverage = (mosaic_count == 0)
        mosaic_count_safe = mosaic_count.astype(np.float64)
        mosaic_count_safe[mask_no_coverage] = 1.0
        mosaic = (mosaic_sum.astype(np.float64) / mosaic_count_safe)

        # Restore nodata where no tile contributed
        nodata = datasets[0].nodata
        if nodata is not None:
            mosaic[mask_no_coverage] = nodata

        mosaic = mosaic.astype(datasets[0].dtypes[0])
        return mosaic, transform

    @staticmethod
    def _merge_with_flip(datasets, method="first"):
        """Merge datasets after flipping to positive pixel height.

        Workaround for rasterio versions that reject north-up (negative pixel
        height) transforms.  Flips all datasets to positive pixel height in
        memory, merges, then flips the result back.
        """
        from rasterio.merge import merge as rio_merge
        from rasterio.transform import Affine
        from rasterio import MemoryFile

        mem_files = []
        flipped_datasets = []
        needs_flip = False

        for ds in datasets:
            if ds.transform.e < 0:
                needs_flip = True
                data = ds.read()[:, ::-1, :]  # flip vertically
                new_transform = Affine(
                    ds.transform.a, ds.transform.b, ds.transform.c,
                    ds.transform.d, -ds.transform.e,
                    ds.transform.f + ds.transform.e * ds.height,
                )
                profile = ds.profile.copy()
                profile['transform'] = new_transform
                memfile = MemoryFile()
                with memfile.open(**profile) as mem_dst:
                    mem_dst.write(data)
                flipped_datasets.append(memfile.open())
                mem_files.append(memfile)
            else:
                flipped_datasets.append(ds)

        mosaic, mosaic_transform = rio_merge(flipped_datasets, method=method)

        # Close flipped in-memory datasets
        for ds in flipped_datasets:
            if ds not in datasets:
                ds.close()
        for mf in mem_files:
            mf.close()

        # Flip result back to north-up (negative pixel height)
        if needs_flip:
            mosaic = mosaic[:, ::-1, :].copy()
            mosaic_transform = Affine(
                mosaic_transform.a, mosaic_transform.b, mosaic_transform.c,
                mosaic_transform.d, -mosaic_transform.e,
                mosaic_transform.f + mosaic_transform.e * mosaic.shape[1],
            )

        return mosaic, mosaic_transform

    @staticmethod
    def _chunked_merge(
        tile_paths: list[Path],
        output_path: Path,
        chunk_size: int = 100,
    ) -> None:
        """Merge many tiles without exceeding the OS open-file limit.

        When *tile_paths* contains more tiles than *chunk_size*, the merge is
        done in rounds: each chunk is merged into a temporary GeoTIFF, then
        the intermediate files are merged into the final output.  This avoids
        the ``Too many open files`` error that occurs when rasterio tries to
        hold hundreds of file descriptors simultaneously.

        Args:
            tile_paths: Paths to the individual prediction GeoTIFFs.
            output_path: Destination path for the merged result.
            chunk_size: Max number of files to open at once (default 100,
                conservative to account for FDs used by GDAL, Python, etc.).
        """
        import gc

        if len(tile_paths) <= chunk_size:
            # Small enough → single-pass merge
            ChangeDetectionChangeFormer._single_merge(tile_paths, output_path)
            return

        logger.info(
            "Batched merge: %d tiles in chunks of %d",
            len(tile_paths), chunk_size,
        )

        intermediate_paths: list[Path] = []
        tmp_dir = output_path.parent / "_merge_tmp"
        tmp_dir.mkdir(exist_ok=True)

        try:
            # --- Round 1: merge each chunk → intermediate file ---
            for chunk_idx in range(0, len(tile_paths), chunk_size):
                chunk = tile_paths[chunk_idx: chunk_idx + chunk_size]
                if len(chunk) == 1:
                    # Single tile, no merge needed – use directly
                    intermediate_paths.append(chunk[0])
                    continue

                intermediate_path = tmp_dir / f"_chunk_{chunk_idx}.tif"
                ChangeDetectionChangeFormer._single_merge(chunk, intermediate_path)
                intermediate_paths.append(intermediate_path)
                logger.debug(
                    "  Chunk %d–%d merged → %s",
                    chunk_idx, chunk_idx + len(chunk) - 1, intermediate_path.name,
                )
                # Force-release file descriptors held by rasterio / GDAL
                gc.collect()

            # --- Round 2: merge intermediates → final output ---
            if len(intermediate_paths) == 1:
                # Only one intermediate: just rename/copy
                import shutil
                shutil.move(str(intermediate_paths[0]), str(output_path))
            else:
                ChangeDetectionChangeFormer._single_merge(
                    intermediate_paths, output_path,
                )

        finally:
            # Clean up intermediate files
            for p in tmp_dir.glob("_chunk_*.tif"):
                try:
                    p.unlink()
                except OSError:
                    pass
            try:
                tmp_dir.rmdir()
            except OSError:
                pass

    @staticmethod
    def _single_merge(tile_paths: list[Path], output_path: Path) -> None:
        """Merge a list of tile GeoTIFFs into a single output file.

        All files in *tile_paths* are opened, merged via
        :func:`_safe_merge`, written to *output_path*, then closed.
        """
        datasets_to_merge = []
        try:
            datasets_to_merge = [rio.open(str(p)) for p in tile_paths]
            mosaic, mosaic_transform = ChangeDetectionChangeFormer._safe_merge(
                datasets_to_merge,
            )

            merge_profile = datasets_to_merge[0].profile.copy()
            merge_profile.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": mosaic_transform,
            })

            with rio.open(str(output_path), "w", **merge_profile) as dst:
                dst.write(mosaic)

            # Free large arrays immediately
            del mosaic

        finally:
            for ds in datasets_to_merge:
                try:
                    ds.close()
                except Exception:
                    pass

    @staticmethod
    def _merge_predictions(
            group_tile_paths: dict[tuple[str, ...], list[Path]],
            event_all_tile_paths: dict[str, list[Path]],
    ) -> None:
        """Merge tiles in two passes:
        1. Per event/pre-post pair/beam/pass → one self-describing GeoTIFF
        2. All tiles in the event/date dir    → merged_all.tif

        Uses :meth:`_chunked_merge` to handle large tile counts without
        exceeding the OS open-file descriptor limit.
        """
        import gc

        # --- Pass 1 : merge par paire pré/post et configuration SAR ---
        for (
            event_date_dir_str,
            event_id,
            event_start_date,
            event_end_date,
            group_pre,
            group_date_pre,
            group_post,
            group_date_post,
            beam,
            sat_pass,
        ), tile_paths in group_tile_paths.items():
            event_date_dir = Path(event_date_dir_str)
            if len(tile_paths) == 1:
                logger.info(
                    "Writing single-cell merge for event %s, group %s/%s, beam %s, pass %s",
                    event_id, group_pre, group_post, beam, sat_pass,
                )

            merged_name = ChangeDetectionChangeFormer._merged_group_filename(
                event_id,
                event_start_date,
                event_end_date,
                group_pre,
                group_date_pre,
                group_post,
                group_date_post,
                beam,
                sat_pass,
            )
            merged_path = event_date_dir / merged_name
            logger.info("Merging %d tiles → %s/%s", len(tile_paths), event_date_dir, merged_name)

            try:
                ChangeDetectionChangeFormer._chunked_merge(tile_paths, merged_path)
                logger.info("Saved merged group to %s", merged_path)
            except Exception:
                logger.exception(
                    "Failed to merge event %s, group %s/%s, beam %s, pass %s in %s",
                    event_id, group_pre, group_post, beam, sat_pass, event_date_dir,
                )

        # Force GC between passes to release all FDs from Pass 1
        gc.collect()

        # --- Pass 2 : merge global par EVENT_ID / PREDICTION_DATE ---
        for event_date_dir_str, tile_paths in event_all_tile_paths.items():
            event_date_dir = Path(event_date_dir_str)
            if len(tile_paths) < 2:
                logger.info("Skipping global merge for %s (only %d tile)",
                            event_date_dir, len(tile_paths))
                continue

            merged_path = event_date_dir / "merged_all.tif"
            logger.info("Merging all %d tiles → %s", len(tile_paths), merged_path)

            try:
                ChangeDetectionChangeFormer._chunked_merge(tile_paths, merged_path)
                logger.info("Saved global merge to %s", merged_path)
            except Exception:
                logger.exception("Failed to create global merge in %s", event_date_dir)


