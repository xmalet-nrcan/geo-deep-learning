"""Change Detection with ChangeFormer model for RCM SAR data."""

import json
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import kornia as krn
import numpy as np
import rasterio as rio
import torch
import torch.nn.functional as F
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from rasterio.transform import Affine
from torch import Tensor
from torchmetrics import F1Score, JaccardIndex, Precision, Recall
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper

from geo_deep_learning.datasets.rcm_change_detection_dataset import NO_DATA
from geo_deep_learning.models.change_detection.change_detection_model import ChangeDetectionModel
from geo_deep_learning.tools.visualization import visualize_prediction
from geo_deep_learning.utils.geotiff_merge import (
    extract_scalar,
    group_merge_key,
    merge_predictions,
    parse_crs,
    prediction_output_filename,
    transform_coeffs,
)
from geo_deep_learning.utils.models import load_weights_from_checkpoint
from geo_deep_learning.utils.tile_reassembly import reassemble_overlapping_tiles

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)
IGNORE_MASK_INDEX = 255

# Deep supervision weights for ChangeFormer's 5 output heads (c4→c1→final).
# Intermediate heads get decreasing weight; the final head gets the most.
DEEP_SUPERVISION_WEIGHTS = [0.1, 0.1, 0.15, 0.2, 1.0]

# Default ascending probability thresholds defining exclusive confidence bands
# for probability-zone vectorization (see ``predict_step``/``_new_manifest``).
# Pixels below the first threshold (0.3) are left unclassified/unzoned.
DEFAULT_PROBABILITY_ZONE_THRESHOLDS = (0.3, 0.5, 0.7, 0.8, 0.9, 1.0)

# Sentinel written to probability GeoTIFFs for invalid/masked pixels.
# Probabilities live in [0, 1], so a negative value is an unambiguous nodata.
PROBABILITY_NODATA = -1.0

# Only visualize samples with at least this fraction of valid pixels labeled
# "burned" (class 1) — keeps validation/test figures informative.
MIN_BURNED_RATIO_FOR_VISUALIZATION = 0.10

# Replacement value for NaN logits before they reach the loss functions.
# Using 0.0 (a neutral / low-confidence score) avoids injecting an
# artificially "certain" class prediction into the loss — see docstring on
# ``_sanitize_logits`` for details on why an extreme value here is unsafe.
NAN_LOGIT_REPLACEMENT = 0.0

# Threshold above which a ``mask-common`` / validity-mask pixel is considered
# valid (values are floats after augmentation/interpolation, not strict 0/1).
MASK_VALID_THRESHOLD = 0.5

# Samples with fewer than this fraction of valid pixels are patched with
# low-amplitude noise and excluded from the loss (see
# ``_patch_degenerate_samples``) to avoid near-zero LayerNorm variance.
MIN_VALID_RATIO_FOR_PATCH = 0.05


class _ForwardLossOutput(NamedTuple):
    """Result of :meth:`ChangeDetectionChangeFormer._forward_and_get_loss`.

    Kept as a ``NamedTuple`` (rather than a plain tuple) so each field is
    self-documenting while remaining fully compatible with the positional
    unpacking used at every call site (``training_step``, ``validation_step``,
    ``test_step``).
    """

    x_pre: Tensor
    x_post: Tensor
    y_float: Tensor
    one_hot: Tensor
    logits: Tensor
    main_loss: Tensor
    focal_loss: Tensor
    lovasz_loss: Tensor
    final_head_loss: Tensor
    batch_size: int


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
            use_signed_difference: bool = False,
            signed_difference_channels: int | None = None,
            signed_difference_normalize: bool = False,
            probability_zone_thresholds: list[float] | None = None,
            probability_zone_class_index: int = -1,
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
            use_signed_difference: Append the signed temporal difference (x1 - x2)
                as extra input channels before the encoder. Injects the direction
                of change (e.g. a drop of SAR backscatter = burned area). Works for
                every backbone (changeformer, changestar2, ...).
            signed_difference_channels: If set, a learnable 1x1 conv compresses the
                signed difference to this many channels. If None, the full signed
                difference (in_channels) is appended.
            signed_difference_normalize: If True, bound the signed-difference
                channels to [-1, 1] via tanh.
            probability_zone_thresholds: Ascending probability thresholds (in
                ``[0, 1]``) defining the exclusive confidence bands written to the
                prediction manifest for downstream probability-zone vectorization
                (see scanfire's ``SegmentationIngestionService``). Pixels whose
                probability falls below the first threshold are considered
                unclassified and excluded from zoning. Defaults to
                :data:`DEFAULT_PROBABILITY_ZONE_THRESHOLDS`.
            probability_zone_class_index: Channel index into the per-class softmax
                probabilities used as the "positive" (e.g. burn/change) class
                probability surface for zoning. Defaults to ``-1`` (last class),
                which is correct for the standard binary no-change/change setup.
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
        self.use_signed_difference = use_signed_difference
        self.signed_difference_channels = signed_difference_channels
        self.signed_difference_normalize = signed_difference_normalize
        self.probability_zone_thresholds = (
            list(probability_zone_thresholds)
            if probability_zone_thresholds
            else list(DEFAULT_PROBABILITY_ZONE_THRESHOLDS)
        )
        self.probability_zone_class_index = probability_zone_class_index

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

        classification_num_classes = self.num_classes if self.num_classes > 1 else 2
        task_type = "multiclass" if classification_num_classes > 2 else "binary"

        for split in ("train", "val", "test"):
            metrics = self._build_classification_metrics(classification_num_classes, task_type)
            setattr(self, f"{split}_iou", metrics["iou"])
            setattr(self, f"{split}_f1", metrics["f1"])
            setattr(self, f"{split}_precision", metrics["precision"])
            setattr(self, f"{split}_recall", metrics["recall"])

        self.predict_output_dir = predict_output_dir

    def _build_classification_metrics(self, num_classes: int, task_type: str) -> dict[str, Any]:
        """Instantiate IoU/F1/Precision/Recall metrics for one data split.

        ``ignore_index=IGNORE_MASK_INDEX`` is set on every metric (including
        the multiclass ``JaccardIndex``, which was previously missing it) so
        that all four metrics behave consistently even if an ignore-index
        pixel ever reaches ``.update()`` without being pre-filtered.
        """
        if num_classes == 2:  # noqa: PLR2004
            return {
                "iou": BinaryJaccardIndex(threshold=self.threshold, ignore_index=IGNORE_MASK_INDEX),
                "f1": F1Score(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX),
                "precision": BinaryPrecision(threshold=self.threshold, ignore_index=IGNORE_MASK_INDEX),
                "recall": BinaryRecall(threshold=self.threshold, ignore_index=IGNORE_MASK_INDEX),
            }
        return {
            "iou": JaccardIndex(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX),
            "f1": F1Score(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX),
            "precision": Precision(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX),
            "recall": Recall(task=task_type, num_classes=num_classes, ignore_index=IGNORE_MASK_INDEX),
        }

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
                "sat_pass": 2,        # ASC / DESC
                "beam": 4,            # A / B / C / D
                "pre_season": 13,     # 13, month number + 0 if undefined
                "post_season": 13,    # 13, month number + 0 if undefined
                "time_delta": 5,      # 0-4d / 4-12d / 12-24d / 24-48d / 48d+
                "processing_year": 3,  # 0: undefined, 1: 2023, 2: != 2023
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
            use_signed_difference=self.use_signed_difference,
            signed_difference_channels=self.signed_difference_channels,
            signed_difference_normalize=self.signed_difference_normalize,
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
            scheduler = self._build_onecycle_scheduler(optimizer)
        else:
            scheduler = self.scheduler(optimizer)

        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def _build_onecycle_scheduler(
            self, optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.OneCycleLR:
        """Build a ``OneCycleLR`` scheduler, inferring ``total_steps`` when needed.

        Lightning's ``trainer.estimated_stepping_batches`` is normally used,
        but falls back to computing steps from the DataModule's
        ``epoch_size`` (IterableDataset case) or, failing that, to an
        explicit ``total_steps`` from the YAML config.
        """
        init_args = self.hparams.get("scheduler", {}).get("init_args", {})
        max_lr = init_args.get("max_lr")
        extra_kwargs = {
            key: init_args[key]
            for key in (
                "pct_start", "anneal_strategy", "div_factor",
                "final_div_factor", "three_phase", "cycle_momentum",
            )
            if key in init_args
        }

        stepping_batches = self.trainer.estimated_stepping_batches
        if stepping_batches > -1:
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=stepping_batches,
                **extra_kwargs,
            )

        epoch_size = getattr(self.trainer.datamodule, "epoch_size", None)
        if stepping_batches == -1 and epoch_size is not None:
            batch_size = self.trainer.datamodule.batch_size
            accumulate_grad_batches = self.trainer.accumulate_grad_batches
            max_epochs = self.trainer.max_epochs
            steps_per_epoch = math.ceil(
                epoch_size / (batch_size * accumulate_grad_batches),
            )
            buffer_steps = int(steps_per_epoch * accumulate_grad_batches)
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                steps_per_epoch=steps_per_epoch + buffer_steps,
                epochs=max_epochs,
                **extra_kwargs,
            )

        total_steps = init_args.get("total_steps")
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            **extra_kwargs,
        )

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

    def on_after_batch_transfer(self, batch: dict[str, Any], dataloader_idx: int) -> dict[str, Any]:  # noqa: ARG002
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

        # 2. Intensity augmentations + SAR-specific multiplicative speckle noise,
        # both applied per image key (generic, images only).
        for img_key in ["image_pre", "image"]:
            batch[img_key] = self._intensity_aug({img_key: batch[img_key]})[img_key]
            if self.speckle_noise_std > 0:
                batch[img_key] = self._apply_speckle_noise(batch[img_key], self.speckle_noise_std)

        return batch

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
            self._update_split_metrics("train", batch, logits, one_hot)

        return main_loss

    def _update_split_metrics(
            self,
            split: str,
            batch: dict[str, Any],
            logits: Tensor,
            one_hot: Tensor,
            *,
            classwise: ClasswiseWrapper | None = None,
    ) -> None:
        """Update the IoU/F1/Precision/Recall metrics for one data split.

        Shared by ``training_step``/``validation_step``/``test_step``, which
        previously duplicated this block. Restricts to valid, tile-owned
        pixels (see :meth:`_apply_tile_ownership`) and skips the update
        entirely when no valid pixel remains in the batch. Uses ``.update()``
        (never ``metric(...)``) so no per-step ``compute()`` is wasted.
        """
        common_mask = self._apply_tile_ownership(batch, batch["mask-common"])
        valid_preds, valid_targets = self._extract_valid_pixels(logits, one_hot, common_mask)

        valid_mask = valid_preds != IGNORE_MASK_INDEX
        if not valid_mask.any():
            return

        vp, vt = valid_preds[valid_mask], valid_targets[valid_mask]
        if classwise is not None:
            classwise.update(vp, vt)
        getattr(self, f"{split}_iou").update(vp, vt)
        getattr(self, f"{split}_f1").update(vp, vt)
        getattr(self, f"{split}_precision").update(vp, vt)
        getattr(self, f"{split}_recall").update(vp, vt)

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
        valid_pixels = common_mask.squeeze(1) > MASK_VALID_THRESHOLD  # robust to float imprecision

        # Prédictions et targets en indices de classe : [B, H, W]
        preds = torch.argmax(logits, dim=1)  # [B, H, W]
        targets = torch.argmax(one_hot, dim=1)  # [B, H, W]

        # Mettre les pixels invalides à IGNORE_MASK_INDEX pour qu'ils soient ignorés par les métriques
        preds[~valid_pixels] = IGNORE_MASK_INDEX
        targets[~valid_pixels] = IGNORE_MASK_INDEX

        # Retourner tous les pixels aplatis — les métriques avec ignore_index ignoreront IGNORE_MASK_INDEX
        return preds.flatten(), targets.flatten()

    # ------------------------------------------------------------------
    # Tile-ownership mask (makes tiled metrics comparable to non-tiled)
    # ------------------------------------------------------------------

    @staticmethod
    def _axis_tile_starts(source: int, tile: int, stride: int) -> list[int]:
        """Reproduce the tile origins used at tiling time for one axis.

        Mirrors ``TiledChangeDetectionDataset._expand_files_with_tiles`` so the
        model can reconstruct, per sample, the full set of tile positions from
        ``source_height`` / ``source_width`` alone.
        """
        if source <= tile:
            return [0]
        starts = list(range(0, (source - tile) + 1, stride))
        last = source - tile
        if last not in starts:
            starts.append(last)
        return sorted(set(starts))

    @staticmethod
    def _scalar_at(field: Any, index: int) -> int:
        """Return ``int(field[index])``, handling both Tensor and list/tuple batches.

        Batched per-sample scalar fields (e.g. ``tile_row_start``,
        ``buffer_size``, ``original_height``) may arrive as a ``Tensor`` or as
        a plain Python list depending on the collate function. This was
        previously reimplemented ad hoc in three places
        (``_mask_buffer_zone``, ``_tile_ownership_mask``, ``_write_prediction_batch``).
        """
        value = field[index]
        return int(value.item()) if isinstance(field, torch.Tensor) else int(value)

    @staticmethod
    def _axis_ownership_end(starts: list[int], start: int, tile: int) -> int:
        """Local exclusive end index of the region a tile *exclusively* owns.

        A source pixel is owned by the tile with the **largest** start that
        still covers it.  In tile-local coordinates a tile therefore owns
        ``[0, next_start - start)`` (or the full tile for the last one),
        guaranteeing a partition of the source with no overlap and no gap.
        """
        nexts = [s for s in starts if s > start]
        return tile if not nexts else (min(nexts) - start)

    def _tile_ownership_mask(
            self,
            batch: dict[str, Any],
            ref_shape: torch.Size,
    ) -> Tensor | None:
        """Return a ``[B, 1, H, W]`` mask keeping only pixels each tile owns.

        Eliminates double-counting of overlapping tiles in the metrics so that
        tiled evaluation counts every source pixel exactly once — matching the
        non-tiled setup.  Returns ``None`` when tiling is inactive (no-op).
        """
        if "tile_row_start" not in batch or "source_height" not in batch:
            return None
        dm = getattr(self.trainer, "datamodule", None)
        tile_size = getattr(dm, "tile_size", None)
        tile_stride = getattr(dm, "tile_stride", None)
        if not tile_size or not tile_stride:
            return None

        tile_h, tile_w = tile_size
        stride_h, stride_w = tile_stride
        b, _, h, w = ref_shape
        mask = torch.zeros((b, 1, h, w), dtype=torch.float32)

        for i in range(b):
            r = self._scalar_at(batch["tile_row_start"], i)
            c = self._scalar_at(batch["tile_col_start"], i)
            sh = self._scalar_at(batch["source_height"], i)
            sw = self._scalar_at(batch["source_width"], i)

            row_starts = self._axis_tile_starts(sh, tile_h, stride_h)
            col_starts = self._axis_tile_starts(sw, tile_w, stride_w)
            row_end = min(self._axis_ownership_end(row_starts, r, tile_h), h)
            col_end = min(self._axis_ownership_end(col_starts, c, tile_w), w)

            if row_end > 0 and col_end > 0:
                mask[i, 0, :row_end, :col_end] = 1.0
        return mask

    def _apply_tile_ownership(self, batch: dict[str, Any], common_mask: Tensor) -> Tensor:
        """Restrict ``common_mask`` to owned pixels so overlapping tiles are counted once."""
        own = self._tile_ownership_mask(batch, common_mask.shape)
        if own is None:
            return common_mask
        return common_mask * own.to(common_mask.device, common_mask.dtype)

    # ------------------------------------------------------------------
    # Shared epoch-end metric logging helpers (used by on_{train,val,test}_epoch_end)
    # ------------------------------------------------------------------

    def _log_epoch_metrics(self, prefix: str) -> dict[str, Tensor]:
        """Compute, log, and return the IoU/F1/Precision/Recall metrics for one split.

        Returns the computed values keyed by short metric name so callers
        needing an extra derived log line (e.g. ``val_recall_burn``) can
        reuse them instead of calling ``.compute()`` a second time.
        """
        values = {
            "iou": getattr(self, f"{prefix}_iou").compute(),
            "f1": getattr(self, f"{prefix}_f1").compute(),
            "precision": getattr(self, f"{prefix}_precision").compute(),
            "recall": getattr(self, f"{prefix}_recall").compute(),
        }
        for name, value in values.items():
            self.log(f"{prefix}_{name}", value, prog_bar=True, sync_dist=True)
        return values

    def _log_classwise_iou(self, prefix: str, classwise_metric: ClasswiseWrapper) -> None:
        """Log one ``{prefix}_iou_{class_name}`` line per class."""
        for class_name, value in classwise_metric.compute().items():
            self.log(f"{prefix}_iou_{class_name}", value, prog_bar=False, sync_dist=True)

    @staticmethod
    def _reset_metrics(*metrics: Any) -> None:
        """Reset every metric passed in, in one call."""
        for metric in metrics:
            metric.reset()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)

        self._reset_metrics(self.train_iou, self.train_f1, self.train_precision, self.train_recall)

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
            self._update_split_metrics("val", batch, logits, one_hot, classwise=self.val_iou_classwise)

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

    def on_validation_epoch_end(self) -> None:
        self._log_classwise_iou("val", self.val_iou_classwise)
        values = self._log_epoch_metrics("val")
        # In binary setup this recall corresponds to class 1 (burned).
        self.log("val_recall_burn", values["recall"], prog_bar=True, sync_dist=True)

        self._reset_metrics(
            self.val_iou_classwise, self.val_iou, self.val_f1, self.val_precision, self.val_recall,
        )

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

        # --- Update metrics ---
        with torch.no_grad():
            self._update_split_metrics("test", batch, logits, one_hot, classwise=self.test_iou_classwise)

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

    def on_test_epoch_end(self) -> None:
        self._log_classwise_iou("test", self.test_iou_classwise)
        self._log_epoch_metrics("test")
        self._reset_metrics(
            self.test_iou_classwise, self.test_iou, self.test_f1, self.test_precision, self.test_recall,
        )

    def _forward_and_get_loss(
            self, batch: dict[str, Any],
    ) -> _ForwardLossOutput:
        """Run the forward pass and compute the (possibly deep-supervised) loss.

        Returns a :class:`_ForwardLossOutput` — behaves like a plain 10-tuple
        for the positional unpacking used at every call site, but each field
        is named for readability.
        """
        x_pre, x_post = batch["image_pre"], batch["image"]
        y = batch["mask"]
        batch_size = x_post.shape[0]

        common_data_mask = self._mask_buffer_zone(batch, batch["mask-common"], batch_size)
        # Propagate the modified mask so that metrics in training_step /
        # validation_step / test_step use the same valid-pixel set as the loss.
        batch["mask-common"] = common_data_mask

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

        self._patch_degenerate_samples(x_pre, x_post, y, common_data_mask, batch_size)

        raw_output = self(
            x_pre,
            x_post,
            sat_pass=batch.get("sat_pass_value"),
            beam=batch.get("beam_value"),
            pre_season=batch.get("pre_season"),
            post_season=batch.get("post_season"),
            time_delta=batch.get("time_delta_bin"),
            processing_year=batch.get("processing_year"),
        )

        # Deep supervision: raw_output is a list during training, single Tensor otherwise
        if isinstance(raw_output, list):
            all_outputs = raw_output  # [p_c4, p_c3, p_c2, p_c1, final]
            logits = all_outputs[-1]  # final prediction for metrics
        else:
            all_outputs = [raw_output]
            logits = raw_output

        y_float = y.float()
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
                device=logits.device, dtype=logits.dtype,
            )
            return _ForwardLossOutput(x_pre, x_post, y_float, dummy_one_hot, logits_safe, zero_loss, zero_loss, zero_loss, zero_loss, batch_size)

        logits_no_nan = self._sanitize_logits(logits)

        # Préparation du one-hot (+ variante avec ignore_index pour la loss)
        one_hot, one_hot_for_loss = self._prepare_one_hot_targets(y, common_data_mask, num_classes)

        # Vérifier qu'il reste des pixels valides
        if common_data_mask.sum() == 0:
            zero = torch.tensor(0.0, device=logits_no_nan.device, dtype=logits_no_nan.dtype, requires_grad=True)
            return _ForwardLossOutput(x_pre, x_post, y_float, one_hot, logits_no_nan, zero, zero, zero, zero, batch_size)

        # Vérifier entrées de la loss
        if not torch.isfinite(one_hot).all():
            raise RuntimeError("One-hot targets contain non-finite values (NaN/Inf).")

        if self.deep_supervision and len(all_outputs) > 1 and self.training:
            main_loss, focal_loss_val, lovasz_loss_val, final_head_loss = self._compute_deep_supervision_loss(
                all_outputs, one_hot_for_loss, target_shape=logits_no_nan.shape[2:],
            )
        else:
            # Standard single-head loss (final_head_loss == main_loss)
            main_loss, focal_loss_val, lovasz_loss_val = self._compute_single_head_loss(
                logits_no_nan, one_hot_for_loss,
            )
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

        return _ForwardLossOutput(
            x_pre, x_post, y_float, one_hot, logits_no_nan,
            main_loss, focal_loss_val, lovasz_loss_val, final_head_loss, batch_size,
        )

    @staticmethod
    def _sanitize_logits(logits: Tensor) -> Tensor:
        """Replace non-finite values in logits with numerically-safe ones.

        NaN is replaced with ``NAN_LOGIT_REPLACEMENT`` (0.0) — a neutral,
        low-confidence score — rather than an extreme value.  An earlier
        version of this code used ``nan=1e15``: since softmax subtracts the
        per-pixel max before exponentiating, a 1e15 logit would deterministically
        assign ~100% probability to that class for the affected pixel,
        injecting a large, meaningless gradient into the loss.  ``posinf``/
        ``neginf`` are clamped to ``1.0``/``0.0`` as before.
        """
        return torch.nan_to_num(logits, nan=NAN_LOGIT_REPLACEMENT, posinf=1.0, neginf=0.0)

    @staticmethod
    def _buffer_valid_region(
            buf: int, tr: int, tc: int, tile_h: int, tile_w: int, oh: int, ow: int,
    ) -> tuple[int, int, int, int]:
        """Valid (non-buffer) region of one tile, in tile-local pixel coordinates.

        Returns ``(row_start, row_end, col_start, col_end)`` bounding the
        pixels of a ``tile_h x tile_w`` tile — cropped from an expanded
        ``(cell + 2*buf)`` image at offset ``(tr, tc)`` — that fall inside
        the true, un-buffered ``oh x ow`` cell. Either span collapses
        (``*_end <= *_start``) when the tile lies entirely within the
        neighbour-context buffer ring, i.e. it owns zero true-cell pixels.

        The non-tiled case is the special case ``tr = tc = 0`` and
        ``tile_h, tile_w = oh + 2*buf, ow + 2*buf`` (the whole buffered
        image treated as a single "tile").
        """
        row_start = max(0, buf - tr)
        row_end = min(tile_h, buf + oh - tr)
        col_start = max(0, buf - tc)
        col_end = min(tile_w, buf + ow - tc)
        return row_start, row_end, col_start, col_end

    @staticmethod
    def _mask_buffer_zone(
            batch: dict[str, Any],
            common_data_mask: Tensor,
            batch_size: int,
    ) -> Tensor:
        """Zero out the neighbour-context buffer zone (``train_overlap_buffer``).

        When spatial context is loaded from adjacent cells the image is
        ``(h + 2b) × (w + 2b)`` but the ground-truth label mask only covers
        the central ``(h × w)`` region.  The buffer zone is zeroed in the
        returned mask so the loss is never computed there.  The dataset
        stores ``buffer_size``, ``cell_orig_height``, ``cell_orig_width``
        whenever a non-zero buffer was applied (train or predict).
        """
        if "buffer_size" not in batch:
            return common_data_mask

        buf_raw = batch["buffer_size"]
        buf = ChangeDetectionChangeFormer._scalar_at(buf_raw, 0)
        if buf <= 0:
            return common_data_mask

        common_data_mask = common_data_mask.clone()

        if "tile_row_start" in batch:
            # --- Tiled + buffered ---
            # Each tile is a crop of the expanded (cell + 2*buf) image.  The
            # valid zone (central cell, excluding neighbour context) spans
            # rows [buf, buf + orig_h) × cols [buf, buf + orig_w) in the
            # expanded image.  We compute the intersection of this valid
            # zone with each tile's coverage area and mask everything
            # outside.  Without this per-tile logic, masking buf pixels
            # from ALL 4 edges of EVERY tile wrongly discarded up to 63 % of
            # valid pixels on interior tiles that don't touch the buffer
            # boundary at all.
            tile_h, tile_w = common_data_mask.shape[2], common_data_mask.shape[3]
            orig_h_batch = batch["cell_orig_height"]
            orig_w_batch = batch["cell_orig_width"]
            tile_row_start = batch["tile_row_start"]
            tile_col_start = batch["tile_col_start"]

            for i in range(batch_size):
                tr = ChangeDetectionChangeFormer._scalar_at(tile_row_start, i)
                tc = ChangeDetectionChangeFormer._scalar_at(tile_col_start, i)
                oh = ChangeDetectionChangeFormer._scalar_at(orig_h_batch, i)
                ow = ChangeDetectionChangeFormer._scalar_at(orig_w_batch, i)

                vr_start, vr_end, vc_start, vc_end = ChangeDetectionChangeFormer._buffer_valid_region(
                    buf, tr, tc, tile_h, tile_w, oh, ow,
                )

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
            # BUGFIX: ``row_start``/``row_end`` (and column counterparts) are
            # clamped so a buffer >= half the image size correctly zeroes
            # the *entire* mask.  The previous code only masked when
            # ``buf < img_h`` and otherwise did nothing — but with
            # ``buf >= img_h`` the unconditional slice
            # ``common_data_mask[:, :, img_h - buf:, :]`` would have used a
            # *negative* start index (Python counts from the end), silently
            # zeroing the wrong rows instead of the whole mask.
            img_h, img_w = common_data_mask.shape[2], common_data_mask.shape[3]
            row_start = min(buf, img_h)
            row_end = max(img_h - buf, row_start)
            col_start = min(buf, img_w)
            col_end = max(img_w - buf, col_start)
            common_data_mask[:, :, :row_start, :] = 0.0
            common_data_mask[:, :, row_end:, :] = 0.0
            common_data_mask[:, :, :, :col_start] = 0.0
            common_data_mask[:, :, :, col_end:] = 0.0

        return common_data_mask

    @staticmethod
    def _patch_degenerate_samples(
            x_pre: Tensor,
            x_post: Tensor,
            y: Tensor,
            common_data_mask: Tensor,
            batch_size: int,
    ) -> None:
        """Replace near-empty samples with low-amplitude noise, in place.

        Samples with too few valid pixels can make LayerNorm compute a
        variance close to zero, which explodes gradients.  Such samples are
        overwritten with uniform noise, excluded from the loss (mask set to
        0), and their target zeroed out.
        """
        valid_ratio = common_data_mask.flatten(1).mean(dim=1)  # [B]
        bad_mask = valid_ratio < MIN_VALID_RATIO_FOR_PATCH  # [B] booléen
        if not bad_mask.any():
            return

        n_bad = bad_mask.sum().item()
        logger.warning(
            "Patching %d/%d samples with <%.0f%% valid pixels (ratios: %s)",
            n_bad, batch_size, MIN_VALID_RATIO_FOR_PATCH * 100,
            [f"{r:.3f}" for r, b in zip(valid_ratio.tolist(), bad_mask.tolist()) if b],
        )
        # Remplir les samples quasi-vides avec du bruit uniforme [0, 0.01]
        # pour que LayerNorm ait une variance > 0
        noise = torch.rand_like(x_pre[0:1]) * 0.01
        for idx in bad_mask.nonzero(as_tuple=True)[0]:
            x_pre[idx] = noise[0]
            x_post[idx] = noise[0]
            common_data_mask[idx] = 0.0  # exclure ces samples de la loss
            y[idx] = 0

    @staticmethod
    def _prepare_one_hot_targets(
            y: Tensor,
            common_data_mask: Tensor,
            num_classes: int,
    ) -> tuple[Tensor, Tensor]:
        """Build the one-hot target and its ignore-index-aware loss variant.

        Returns ``(one_hot, one_hot_for_loss)``.  Invalid pixels in
        ``one_hot_for_loss`` are filled with ``IGNORE_MASK_INDEX`` so that
        FocalLoss/LovaszLoss (which both natively support
        ``ignore_index=255``) skip them — avoiding the previous
        mask-multiplication approach which biased Lovász sorting (invalid
        pixels got error=1) and added phantom class-0 contributions to
        FocalLoss.
        """
        y_one_hot = y.squeeze(1) if y.dim() == 4 else y  # noqa: PLR2004
        y_one_hot = y_one_hot.clamp(min=0, max=num_classes - 1)  # clamp aussi les 255 → num_classes-1
        one_hot = F.one_hot(y_one_hot.long(), num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).contiguous().float()

        invalid_pixels = common_data_mask < MASK_VALID_THRESHOLD  # [B, 1, H, W], True=invalid
        one_hot_for_loss = one_hot.clone()
        one_hot_for_loss.masked_fill_(invalid_pixels.expand_as(one_hot), IGNORE_MASK_INDEX)
        return one_hot, one_hot_for_loss

    def _compute_single_head_loss(
            self,
            logits: Tensor,
            one_hot_for_loss: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute ``(main_loss, focal_loss, lovasz_loss)`` for one prediction head."""
        w_ml, w_sl = self.loss_ratio
        lovasz_loss_val = self.secondary_loss(logits.contiguous(), one_hot_for_loss)
        focal_loss_val = self.main_loss(logits.contiguous(), one_hot_for_loss)
        main_loss = w_sl * lovasz_loss_val + w_ml * focal_loss_val
        return main_loss, focal_loss_val, lovasz_loss_val

    def _compute_deep_supervision_loss(
            self,
            all_outputs: list[Tensor],
            one_hot_for_loss: Tensor,
            target_shape: torch.Size,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute the weighted-average loss over every ChangeFormer decoder head.

        Returns ``(main_loss, focal_loss, lovasz_loss, final_head_loss)``
        where ``final_head_loss`` is the *unweighted* loss of the last
        (highest-resolution) head, kept comparable to ``val_loss``/
        ``test_loss`` which always use a single head.
        """
        target_h, target_w = target_shape
        ds_weights = self.deep_supervision_weights
        # Ensure we have a weight for each head
        if len(ds_weights) < len(all_outputs):
            ds_weights = ds_weights + [1.0] * (len(all_outputs) - len(ds_weights))
        weight_sum = sum(ds_weights[:len(all_outputs)])

        device, dtype = all_outputs[-1].device, all_outputs[-1].dtype
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_focal = torch.tensor(0.0, device=device, dtype=dtype)
        total_lovasz = torch.tensor(0.0, device=device, dtype=dtype)
        final_head_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for head_idx, head_output in enumerate(all_outputs):
            head_logits = self._sanitize_logits(head_output)
            # Resize intermediate heads to final resolution
            if head_logits.shape[2] != target_h or head_logits.shape[3] != target_w:
                head_logits = F.interpolate(
                    head_logits, size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                )

            head_loss, head_focal, head_lovasz = self._compute_single_head_loss(head_logits, one_hot_for_loss)

            total_loss = total_loss + ds_weights[head_idx] * head_loss
            total_focal = total_focal + ds_weights[head_idx] * head_focal
            total_lovasz = total_lovasz + ds_weights[head_idx] * head_lovasz

            # Save final head loss for fair train/val comparison
            if head_idx == len(all_outputs) - 1:
                final_head_loss = head_loss

        return total_loss / weight_sum, total_focal / weight_sum, total_lovasz / weight_sum, final_head_loss

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
            if len(available) >= 3:  # noqa: PLR2004
                rgb_indices = [available[0], available[len(available) // 2], available[-1]]
            else:
                rgb_indices = available[:3]

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
                    # Use a distinct value (255) for visualization of masked pixels
                    pred = pred.clone()
                    pred[invalid] = self.changed_num_classes  # = 2 → index du gris dans la colormap

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
        # ``mask-common`` décrit la validité des acquisitions SAR, mais ne
        # contient pas nécessairement l'eau. Apply both masks even when the
        # common mask is available so water can never be vectorized as burn.
        invalid_mask = torch.zeros_like(y_pred, dtype=torch.bool)
        if "mask-common" in batch:
            common_mask = batch["mask-common"]  # [B, 1, H, W] bool ou float
            invalid_mask |= common_mask.squeeze(1) < MASK_VALID_THRESHOLD
        if "water_mask" in batch:
            water_mask = batch["water_mask"]  # [B, 1, H, W]
            invalid_mask |= water_mask.squeeze(1) > 0  # eau = valeur > 0
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
        return manifest

    @staticmethod
    def _write_manifest_file(manifest: dict[str, Any], base_dir: Path) -> None:
        manifest_path = base_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info("Saved prediction manifest to %s", manifest_path)

    @staticmethod
    def _write_single_geotiff_prediction(  # noqa: PLR0913
            *,
            pred_np: np.ndarray,
            profile: dict[str, Any],
            base_dir: Path,
            predict_date: str,
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
        """
        event_date_dir = base_dir / event_id / predict_date
        tile_dir = event_date_dir / cell_id
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

        event_date_key = str(event_date_dir)
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
            self._write_single_geotiff_prediction(
                pred_np=info["predictions"],  # [H, W] uint16
                profile=info["profile"],
                base_dir=base_dir,
                predict_date=predict_date,
                cell_id=str(info["cell_id"]),
                pair_id=info.get("pair_id"),
                event_id=str(info.get("event_id", "unknown_event")),
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

        self._write_manifest_file(manifest, base_dir)
        # Merge across cells / groups (same logic as per-tile path)
        merge_predictions(group_tile_paths, event_all_tile_paths)
        logger.info("All blended predictions saved to %s", base_dir)

    def on_predict_end(self) -> None:
        """Appelé après que tous les predict_step soient terminés.

        Structure de sortie :
            output_dir / predictions / EVENT_ID / PREDICTION_DATE / cell_id / image.tif
            output_dir / predictions / EVENT_ID / PREDICTION_DATE / merged.tif
        """
        predictions = self.trainer.predict_loop.predictions
        if not predictions:
            logger.warning("No predictions to save.")
            return

        # --- Base output directory ---
        predict_date = datetime.now().strftime("%Y%m%d_%H%M")
        if self.predict_output_dir is not None:
            base_dir = Path(self.predict_output_dir)
            if base_dir.name != "predictions":
                base_dir = base_dir / "predictions"
        else:
            base_dir = Path(self.trainer.default_root_dir) / "predictions"

        base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving predictions to %s", base_dir)

        # --- Try overlap-based tile reassembly (cosine blending) ---
        dm = getattr(self.trainer, "datamodule", None)
        assembled, used_blending = reassemble_overlapping_tiles(
            predictions,
            getattr(dm, "tile_size", None),
            getattr(dm, "tile_stride", None),
            no_data_value=NO_DATA,
            positive_class_index=self.probability_zone_class_index,
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

        self._write_manifest_file(manifest, base_dir)
        merge_predictions(group_tile_paths, event_all_tile_paths)
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

        buf = self._scalar_at(buf_raw, index)
        if buf <= 0:
            return pred_np, transform, tile_h, tile_w

        oh = self._scalar_at(cell_h_raw, index)
        ow = self._scalar_at(cell_w_raw, index)
        tile_row_starts = batch_result.get("tile_row_start")
        tile_col_starts = batch_result.get("tile_col_start")
        tr = self._scalar_at(tile_row_starts, index) if tile_row_starts is not None else 0
        tc = self._scalar_at(tile_col_starts, index) if tile_col_starts is not None else 0

        row_start, row_end, col_start, col_end = self._buffer_valid_region(buf, tr, tc, tile_h, tile_w, oh, ow)
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
            orig_h = self._scalar_at(orig_heights, i)
            orig_w = self._scalar_at(orig_widths, i)

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

            self._write_single_geotiff_prediction(
                pred_np=pred_np,
                profile=profile_i,
                base_dir=base_dir,
                predict_date=predict_date,
                cell_id=cell_id,
                pair_id=extract_scalar(batch_pair_ids, i, default=None),
                event_id=extract_scalar(batch_event_ids, i, default="unknown_event"),
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

