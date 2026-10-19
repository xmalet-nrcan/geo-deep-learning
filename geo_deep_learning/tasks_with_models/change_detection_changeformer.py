"""Change Detection with ChangeFormer model for RCM SAR data.

The Lightning task lives here; the self-contained pieces it relies on are split out:

* :mod:`~geo_deep_learning.tasks_with_models.change_detection_predict` — ``predict_step``
  (TTA), GeoTIFF / manifest writing and cross-cell merging (:class:`PredictOutputMixin`).
* :mod:`~geo_deep_learning.tools.losses.change_detection_loss` — target preparation,
  logit sanitising, burned false-negative penalty, boundary-band BCE.
* :mod:`~geo_deep_learning.utils.tiling` — tile-ownership and neighbour-buffer geometry.
* :mod:`~geo_deep_learning.tools.schedulers.builders` — warm-up / OneCycle scheduler builders.
* :mod:`~geo_deep_learning.tools.change_detection_figures` — validation / test figures.
"""

import logging
import warnings
from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import kornia as krn
import torch
import torch.nn.functional as F
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torchmetrics import F1Score, JaccardIndex, Precision, Recall
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper

from geo_deep_learning.models.change_detection.change_detection_model import ChangeDetectionModel
from geo_deep_learning.tasks_with_models.change_detection_predict import (
    BENCHMARK_LATEST_MANIFEST,  # noqa: F401  (re-exported)
    BENCHMARK_RUNS_DIRNAME,  # noqa: F401  (re-exported)
    DEFAULT_PROBABILITY_ZONE_THRESHOLDS,
    PREDICT_LAYOUT_BENCHMARK,
    PREDICT_LAYOUT_PRODUCTION,
    PREDICT_OUTPUT_LAYOUTS,
    PROBABILITY_NODATA,  # noqa: F401  (re-exported)
    PredictOutputMixin,
)
from geo_deep_learning.tools.change_detection_figures import (
    MIN_BURNED_RATIO_FOR_VISUALIZATION,  # noqa: F401  (re-exported)
    log_change_detection_figures,
)
from geo_deep_learning.tools.losses.change_detection_loss import (
    IGNORE_MASK_INDEX,
    MASK_VALID_THRESHOLD,
    boundary_bce_loss,
    burned_false_negative_penalty,
    patch_degenerate_samples,
    prepare_one_hot_targets,
    sanitize_logits,
)
from geo_deep_learning.tools.schedulers.builders import (
    build_onecycle_scheduler,
    scheduler_class_path,
    with_warmup,
)
from geo_deep_learning.utils.geotiff_merge import (
    DEFAULT_MERGED_ALL_MIN_PAIRS,
    DEFAULT_MERGED_ALL_PROB_THRESHOLD,
)
from geo_deep_learning.utils.models import load_weights_from_checkpoint
from geo_deep_learning.utils.tiling import mask_buffer_zone, tile_ownership_mask

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)

# Deep supervision weights for ChangeFormer's 5 output heads (c4→c1→final).
# Intermediate heads get decreasing weight; the final head gets the most.
DEEP_SUPERVISION_WEIGHTS = [0.1, 0.1, 0.15, 0.2, 1.0]

# Key under which ``water_mask`` goes through the kornia geometric augmentations
# (kornia infers the data type from the key prefix, see ``on_after_batch_transfer``).
WATER_AUG_KEY = "mask-water"


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


class ChangeDetectionChangeFormer(PredictOutputMixin, LightningModule):
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
            weights_strict: bool | None = None,
            in_channels: int | None = None,
            threshold: float = 0.5,
            predict_output_dir: str | None = None,  # For Outputs
            predict_output_layout: Literal["production", "benchmark"] = "production",
            predict_run_name: str | None = None,
            predict_write_merged_all: bool | None = None,
            merged_all_min_pairs: int = DEFAULT_MERGED_ALL_MIN_PAIRS,
            merged_all_prob_threshold: float = DEFAULT_MERGED_ALL_PROB_THRESHOLD,
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
            backbone_kwargs: dict[str, Any] | None = None,
            boundary_loss_weight: float = 0.0,
            boundary_width: int = 2,
            lr_warmup_iters: int = 0,
            lr_warmup_start_factor: float = 0.1,
            **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Initialize the model.

        Args:
            weights_strict: Load ``weights_from_checkpoint_path`` with
                ``strict=True`` (any missing / unexpected key or shape mismatch
                raises instead of logging a warning and leaving layers randomly
                initialised). ``None`` (default) ⇒ ``True`` in the ``benchmark``
                predict layout (best weights of the *same* architecture),
                ``False`` otherwise (transfer between model variants, e.g.
                ChangeFormerV6 → V7). Incompatible with ``load_parts``.
            in_channels: Number of data channels per image. Optional — when
                omitted (recommended) it is inferred from the DataModule's
                ``band_names`` (``len(band_names) + 1`` for BITMASK_CROPPED,
                ``+3`` more when ``separate_metadata=False``).
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
            backbone_kwargs: Extra keyword arguments forwarded to the backbone
                constructor selected by ``change_detection_model`` (they override
                the shared defaults ``decoder_softmax=False, embed_dim=256``).
                E.g. for ``change_detection_model: segformer``::

                    backbone_kwargs:
                      encoder: mit_b2
                      encoder_weights: imagenet   # or null
                      embed_dim: 256
                      fusion: concat_diff         # concat | diff | signed_diff
            boundary_loss_weight: Weight of an extra BCE restricted to a band
                around the ground-truth burn boundaries (see
                :meth:`_boundary_bce_loss`). Pushes the model to place its
                frontiers precisely. ``0`` (default) disables it. Added to the
                train, val and test losses alike (so ``val_loss`` stays
                comparable to the training objective).
            boundary_width: Half-width, in pixels, of that boundary band
                (``2`` -> a 5-pixel-wide band, i.e. ±200 m at 100 m/pixel).
            lr_warmup_iters: Linear learning-rate warm-up length, in scheduler
                steps (i.e. epochs when ``scheduler_config.interval`` is
                ``"epoch"``, batches when it is ``"step"``). When > 0, the
                configured ``scheduler`` is wrapped in
                ``SequentialLR([LinearLR, scheduler])`` and only starts after
                the warm-up. Built in code because jsonargparse cannot
                instantiate the nested schedulers of a ``SequentialLR`` given
                in YAML (they need the optimizer). Ignored for ``OneCycleLR``
                (it has its own warm-up, ``pct_start``). ``0`` = disabled.
            lr_warmup_start_factor: LR multiplier at the first warm-up step
                (``0.1`` -> starts at 10 % of the optimizer ``lr``).
            predict_output_dir: Root directory for predict outputs. In the
                ``production`` layout a ``predictions`` sub-folder is appended
                when missing; in the ``benchmark`` layout it is used as-is.
            predict_output_layout: Directory layout of predict outputs:

                - ``"production"`` (default, consumed by scanfire's ingestion)::

                      <dir>/predictions/<event_id>/<YYYYMMDD_HHMM>/<cell_id>/<output_name>.tif
                      <dir>/predictions/<event_id>/<YYYYMMDD_HHMM>/<merged per pre/post pair>.tif
                      <dir>/predictions/<event_id>/<YYYYMMDD_HHMM>/merged_all.tif
                      <dir>/predictions/manifest.json

                - ``"benchmark"`` (multi-model comparison, see
                  ``docs/dev-plans/2026-10-08_model_benchmark_predict.md``)::

                      <dir>/<event_id>/<output_name>.tif            (one per cell)
                      <dir>/<event_id>/<output_name>_prob.tif
                      <dir>/<event_id>/<merged per pre/post pair>.tif
                      <dir>/_runs/<YYYYMMDD_HHMM>/manifest.json
                      <dir>/manifest_latest.json

                  ``<dir>`` is typically ``<benchmark_output_root>/<model_name>``.
                  Re-running the same model overwrites its rasters (idempotent).
            predict_run_name: Optional run label (e.g. benchmark model name)
                written to the manifest as ``run_name``.
            predict_write_merged_all: Write the global
                ``merged_all_beam-<beam>_pass-<pass>.tif`` (+ ``*_prob.tif``) per
                event and beam/satellite pass. ``None`` (default) ⇒ ``True`` in
                the ``production`` layout, ``False`` in the ``benchmark`` layout
                (it mixes different pre/post pairs and would be meaningless for
                comparison).
            merged_all_min_pairs: Temporal confirmation for ``merged_all``: a
                pixel is burned only if at least this many pre/post pairs detect
                it (capped at the number of pairs with valid data there). See
                :func:`~geo_deep_learning.utils.geotiff_merge.temporal_confirmation_merge`.
            merged_all_prob_threshold: Probability above which a pair counts as
                a detection, also applied to the persistence probability
                (mean probability from the first detection onward).
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
        self.backbone_kwargs = dict(backbone_kwargs or {})
        if boundary_loss_weight < 0:
            msg = "boundary_loss_weight must be >= 0"
            raise ValueError(msg)
        if boundary_width < 1:
            msg = "boundary_width must be >= 1"
            raise ValueError(msg)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.boundary_width = int(boundary_width)
        if lr_warmup_iters < 0:
            msg = "lr_warmup_iters must be >= 0"
            raise ValueError(msg)
        if not 0.0 < lr_warmup_start_factor <= 1.0:
            msg = "lr_warmup_start_factor must be in (0, 1]"
            raise ValueError(msg)
        self.lr_warmup_iters = int(lr_warmup_iters)
        self.lr_warmup_start_factor = float(lr_warmup_start_factor)

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
        if predict_output_layout not in PREDICT_OUTPUT_LAYOUTS:
            msg = (
                f"predict_output_layout must be one of {PREDICT_OUTPUT_LAYOUTS}, "
                f"got {predict_output_layout!r}"
            )
            raise ValueError(msg)
        self.predict_output_layout = predict_output_layout
        self.predict_run_name = predict_run_name
        self.predict_write_merged_all = (
            (predict_output_layout == PREDICT_LAYOUT_PRODUCTION)
            if predict_write_merged_all is None
            else bool(predict_write_merged_all)
        )
        if merged_all_min_pairs < 1:
            msg = f"merged_all_min_pairs must be >= 1, got {merged_all_min_pairs}"
            raise ValueError(msg)
        self.merged_all_min_pairs = int(merged_all_min_pairs)
        self.merged_all_prob_threshold = float(merged_all_prob_threshold)
        self.weights_strict = (
            (predict_output_layout == PREDICT_LAYOUT_BENCHMARK)
            if weights_strict is None
            else bool(weights_strict)
        )
        if self.weights_strict and self.hparams.get("load_parts") is not None:
            msg = (
                "weights_strict=True is incompatible with load_parts "
                "(partial loading is always non-strict)"
            )
            raise ValueError(msg)

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
        self._fold_water_into_common_mask(batch)
        return batch

    @staticmethod
    def _fold_water_into_common_mask(batch: dict[str, Any]) -> None:
        """Mark water pixels invalid in ``batch["mask-common"]``, in place.

        ``mask-common`` is the single validity mask read by the loss, the
        metrics (and their tile-ownership restriction), the figures and the
        predict re-masking. Folding the static ``water_mask`` (``> 0`` = water)
        into it here — once, for every stage — guarantees water is excluded
        everywhere ``mask-common`` is used, instead of only at predict time.
        ``mask`` (the fire ground truth used for the loss) is left untouched.
        Geometric augmentations then flip the folded mask consistently.
        """
        if "mask-common" in batch and "water_mask" in batch:
            land = batch["water_mask"] <= 0
            batch["mask-common"] = batch["mask-common"] * land.to(batch["mask-common"].dtype)

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
            in_channels=self._resolve_in_channels(),
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
            **self.backbone_kwargs,
        )

        if self.weights_from_checkpoint_path:
            map_location = self.device
            load_parts = self.hparams.get("load_parts")
            logger.info(
                "Loading weights from checkpoint: %s (strict=%s)",
                self.weights_from_checkpoint_path,
                self.weights_strict,
            )
            load_weights_from_checkpoint(
                self.model,
                self.weights_from_checkpoint_path,
                load_parts=load_parts,
                map_location=map_location,
                strict=self.weights_strict,
            )

    def _resolve_in_channels(self) -> int:
        """Return ``in_channels``, inferring it from the DataModule when possible.

        ``configure_model`` runs after ``datamodule.setup()``, so the
        DataModule's ``num_input_channels`` (``len(band_names) + 1`` in FiLM
        mode) is available. It takes precedence over the YAML value because
        it reflects the actual tensor shape; a mismatch only logs a warning.
        The resolved value is written back into ``hparams`` so checkpoints
        are self-contained (``load_from_checkpoint`` / predict without a
        DataModule keep working).
        """
        trainer = getattr(self, "_trainer", None)  # ``self.trainer`` raises if detached
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        inferred = getattr(datamodule, "num_input_channels", None)

        if inferred is None:
            if self.in_channels is None:
                msg = (
                    "in_channels could not be inferred: no DataModule exposing "
                    "`num_input_channels` is attached. Set `model.init_args.in_channels` "
                    "explicitly (len(band_names) + 1 in FiLM mode)."
                )
                raise ValueError(msg)
            return self.in_channels

        if self.in_channels is not None and self.in_channels != inferred:
            logger.warning(
                "in_channels=%d from the config does not match the DataModule "
                "(band_names → %d channels). Using %d.",
                self.in_channels, inferred, inferred,
            )
        elif self.in_channels is None:
            logger.info("in_channels inferred from DataModule band_names: %d", inferred)

        self.in_channels = inferred
        self.hparams["in_channels"] = inferred
        if getattr(self, "_hparams_initial", None) is not None:
            self._hparams_initial["in_channels"] = inferred
        return inferred

    def configure_optimizers(self) -> list[list[dict[str, Any]]]:
        """Configure optimizers."""
        optimizer = self.optimizer(self.parameters())
        if scheduler_class_path(self.hparams.get("scheduler")) == "torch.optim.lr_scheduler.OneCycleLR":
            if self.lr_warmup_iters > 0:
                logger.warning(
                    "lr_warmup_iters=%d ignored: OneCycleLR already warms up (pct_start).",
                    self.lr_warmup_iters,
                )
            scheduler = build_onecycle_scheduler(
                optimizer, self.trainer, self.hparams.get("scheduler", {}).get("init_args", {}),
            )
        else:
            scheduler = with_warmup(
                optimizer, self.scheduler(optimizer), self.lr_warmup_iters, self.lr_warmup_start_factor,
            )

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
        if "water_mask" in batch:
            # kornia only augments dict keys that start with a DataKey name
            # (``image*``, ``mask*``…); a bare ``water_mask`` key would be passed
            # through un-flipped. Hence the ``mask-`` prefix.
            keys_to_aug[WATER_AUG_KEY] = batch["water_mask"].to(torch.float32)
        if "mask" in batch:
            keys_to_aug["mask"] = batch["mask"]

        transformed = self._geo_aug(keys_to_aug)
        for key in transformed:
            target_key = "water_mask" if key == WATER_AUG_KEY else key
            batch[target_key] = transformed[key].to(device, non_blocking=True)

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

    def _apply_tile_ownership(self, batch: dict[str, Any], common_mask: Tensor) -> Tensor:
        """Restrict ``common_mask`` to owned pixels so overlapping tiles are counted once."""
        dm = getattr(self.trainer, "datamodule", None)
        own = tile_ownership_mask(
            batch,
            common_mask.shape,
            getattr(dm, "tile_size", None),
            getattr(dm, "tile_stride", None),
        )
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

        common_data_mask = mask_buffer_zone(batch, batch["mask-common"], batch_size)
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

        patch_degenerate_samples(x_pre, x_post, y, common_data_mask, batch_size)

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

        logits_no_nan = sanitize_logits(logits)

        # Préparation du one-hot (+ variante avec ignore_index pour la loss)
        one_hot, one_hot_for_loss = prepare_one_hot_targets(y, common_data_mask, num_classes)

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
        burn_penalty = burned_false_negative_penalty(
            logits_no_nan, one_hot, common_data_mask, self.burned_class_weight,
        )
        main_loss = main_loss + burn_penalty
        final_head_loss = final_head_loss + burn_penalty

        # Boundary-band BCE (sharper burn frontiers), on the final head only
        if self.boundary_loss_weight > 0:
            boundary_loss = boundary_bce_loss(
                logits_no_nan, one_hot, common_data_mask, self.boundary_width,
            )
            weighted_boundary = self.boundary_loss_weight * boundary_loss
            main_loss = main_loss + weighted_boundary
            final_head_loss = final_head_loss + weighted_boundary
            if self.training and getattr(self, "_trainer", None) is not None:
                self.log(
                    "boundary_loss", boundary_loss,
                    on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
                )

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
            head_logits = sanitize_logits(head_output)
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
        """Log input-diff / ground-truth / prediction figures (see :func:`log_change_detection_figures`)."""
        return log_change_detection_figures(
            trainer,
            batch,
            outputs,
            max_samples,
            num_classes=self.num_classes,
            changed_num_classes=self.changed_num_classes,
            class_colors=self.class_colors,
            use_metadata_film=self.use_metadata_film,
            artifact_prefix=artifact_prefix,
            epoch_suffix=epoch_suffix,
        )
