"""Change Detection with ChangeFormer model."""

import json
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
import torch
import torch.nn.functional as F
from kornia.augmentation import AugmentationSequential
import kornia as krn
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from rasterio.merge import merge as rio_merge
from rasterio.transform import Affine
from torch import Tensor
from torchmetrics import JaccardIndex, F1Score
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper

from geo_deep_learning.datasets.rcm_change_detection_dataset import NO_DATA
from geo_deep_learning.models.change_detection.change_detection_model import ChangeDetectionModel
from geo_deep_learning.tools.visualization import visualize_prediction
from geo_deep_learning.utils.models import load_weights_from_checkpoint

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)


def _is_integer_dtype(dtype_name: str) -> bool:
    return "int" in dtype_name.lower() or "uint" in dtype_name.lower()


class ChangeDetectionChangeFormer(LightningModule):
    """Change Detection with ChangeFormer V6 model."""

    def __init__(
        self,
        change_detection_model: str,
        *,
        image_size: tuple[int, int],
        num_classes: int,
        max_samples: int,
        main_loss: Callable,
        secondary_loss: Callable,
        loss_ratio: tuple[float, float] = (1.0, 1.0),
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
        predict_output_dir: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if burned_class_weight < 1.0:
            msg = "burned_class_weight must be >= 1.0"
            raise ValueError(msg)

        self.change_detection_model = change_detection_model
        self.in_channels = in_channels
        self.burned_class_weight = float(burned_class_weight)
        self.num_classes = num_classes
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
        self.predict_output_dir = predict_output_dir
        # SAR-aware augmentation parameters (disabled by default; enable after baseline converges).
        self.sar_speckle_p = float(kwargs.get("sar_speckle_p", 0.14))
        self.sar_speckle_std = float(kwargs.get("sar_speckle_std", 0.08))
        self.sar_jitter_p = float(kwargs.get("sar_jitter_p", 0.14))
        self.sar_jitter_max = float(kwargs.get("sar_jitter_max", 0.08))

        self._effective_num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.labels = class_labels or [str(i) for i in range(self._effective_num_classes)]
        self._total_samples_visualized = 0

        # --- Metrics (train / val / test) ---
        for prefix in ("train", "val", "test"):
            iou, f1, precision, recall = self._create_metrics()
            setattr(self, f"{prefix}_iou", iou)
            setattr(self, f"{prefix}_f1", f1)
            setattr(self, f"{prefix}_precision", precision)
            setattr(self, f"{prefix}_recall", recall)

        # Classwise IoU for val & test
        for prefix in ("val", "test"):
            mean_iou = MeanIoU(
                num_classes=self._effective_num_classes,
                per_class=True,
                input_format="index",
                include_background=True,
            )
            setattr(
                self,
                f"{prefix}_iou_classwise",
                ClasswiseWrapper(mean_iou, labels=self.labels),
            )

    # ------------------------------------------------------------------
    # Metric factory
    # ------------------------------------------------------------------
    def _create_metrics(self):
        nc = self._effective_num_classes
        if nc == 2:
            iou = BinaryJaccardIndex(threshold=self.threshold)
            f1 = F1Score(task="binary", num_classes=nc)
            precision = BinaryPrecision(threshold=self.threshold)
            recall = BinaryRecall(threshold=self.threshold)
        else:
            iou = JaccardIndex(task="multiclass", num_classes=nc)
            f1 = F1Score(task="multiclass", num_classes=nc)
            from torchmetrics import Precision, Recall
            precision = Precision(task="multiclass", num_classes=nc)
            recall = Recall(task="multiclass", num_classes=nc)
        return iou, f1, precision, recall

    # ------------------------------------------------------------------
    # Augmentations
    # ------------------------------------------------------------------
    @staticmethod
    def _geo_aug() -> AugmentationSequential:
        return AugmentationSequential(
            krn.augmentation.RandomHorizontalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomVerticalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomRotation90(times=(1, 3), p=0.5, align_corners=True, keepdim=True),
            data_keys=None,
        )

    @staticmethod
    def _intensity_aug() -> AugmentationSequential:
        return AugmentationSequential(
            krn.augmentation.RandomGaussianNoise(mean=0.0,
                                                 std=0.05,
                                                 p=0.3, keepdim=True),
            # Keep mild blur/erase to avoid distorting SAR texture statistics too much.
            krn.augmentation.RandomGaussianBlur(kernel_size=(3, 3),
                                                sigma=(0.1, 0.8),
                                                p=0.1,
                                                keepdim=True),
            krn.augmentation.RandomErasing(scale=(0.01, 0.03),
                                           ratio=(0.5, 2.0),
                                           p=0.05, keepdim=True),
            data_keys=None,
        )

    def _apply_sar_aware_aug(self, image: Tensor) -> Tensor:
        """Apply lightweight SAR-specific augmentations (speckle + radiometric jitter)."""
        out = image

        if torch.rand(1, device=out.device).item() < self.sar_speckle_p:
            # Multiplicative speckle factor around 1.0.
            speckle = torch.randn_like(out) * self.sar_speckle_std + 1.0
            out = out * speckle.clamp_min(0.0)

        if torch.rand(1, device=out.device).item() < self.sar_jitter_p:
            # Per-sample gain jitter to mimic mild radiometric calibration drift.
            b = out.shape[0]
            gain = 1.0 + (torch.rand((b, 1, 1, 1), device=out.device) * 2 - 1) * self.sar_jitter_max
            out = out * gain

        return out

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def on_before_batch_transfer(self, batch: dict[str, Any], dataloader_idx: int) -> dict[str, Any]:
        pad = AugmentationSequential(
            krn.augmentation.PadTo(size=self.image_size, pad_mode="constant", pad_value=0, keepdim=False),
            data_keys=None,
        )
        keys_to_pad = {"image_pre": batch["image_pre"], "image": batch["image"]}
        for k in ("mask", "mask-common", "water_mask"):
            if k in batch:
                keys_to_pad[k] = batch[k] if k == "mask" else batch[k].to(torch.float32)
        batch.update(pad(keys_to_pad))
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training:
            return batch
        device = batch["image"].device

        # Geometric (images + masks)
        keys = {"image_pre": batch["image_pre"], "image": batch["image"]}
        for k in ("mask-common", "mask"):
            if k in batch:
                keys[k] = batch[k].to(torch.float32) if k == "mask-common" else batch[k]
        transformed = self._geo_aug()(keys)
        for k, v in transformed.items():
            batch[k] = v.to(device, non_blocking=True)

        # Intensity (images only)
        aug = self._intensity_aug()
        for k in ("image_pre", "image"):
            batch[k] = aug({k: batch[k]})[k]
            # batch[k] = self._apply_sar_aware_aug(batch[k])
        return batch

    # ------------------------------------------------------------------
    # Model / optimizers
    # ------------------------------------------------------------------
    def configure_model(self) -> None:
        self.model = ChangeDetectionModel(
            change_detection_model=self.change_detection_model,
            in_channels=self.in_channels,
            out_channels=self._effective_num_classes,
        )
        for m in self.model.modules():
            if isinstance(m, torch.nn.LayerNorm):
                m.eps = 1e-5

        if self.weights_from_checkpoint_path:
            logger.info("Loading weights from checkpoint: %s", self.weights_from_checkpoint_path)
            load_weights_from_checkpoint(
                self.model,
                self.weights_from_checkpoint_path,
                load_parts=self.hparams.get("load_parts"),
                map_location=self.device,
            )

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        sched_cfg = self.hparams.get("scheduler", {})
        class_path = sched_cfg.get("class_path", "")
        init_args = sched_cfg.get("init_args", {})

        if class_path == "torch.optim.lr_scheduler.OneCycleLR":
            max_lr = init_args["max_lr"]
            extra = {
                k: init_args[k]
                for k in ("pct_start", "anneal_strategy", "div_factor", "final_div_factor", "three_phase", "cycle_momentum")
                if k in init_args
            }
            stepping = self.trainer.estimated_stepping_batches
            if stepping > 0:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=stepping, **extra)
            elif hasattr(self.trainer.datamodule, "epoch_size"):
                dm = self.trainer.datamodule
                spe = math.ceil(dm.epoch_size / (dm.batch_size * self.trainer.accumulate_grad_batches))
                buf = int(spe * self.trainer.accumulate_grad_batches)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=max_lr, steps_per_epoch=spe + buf, epochs=self.trainer.max_epochs, **extra,
                )
            else:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=max_lr, total_steps=init_args["total_steps"], **extra,
                )
        else:
            scheduler = self.scheduler(optimizer)

        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def forward(self, image_pre: Tensor, image_post: Tensor) -> Tensor:
        return self.model(image_pre, image_post)[-1]

    # ------------------------------------------------------------------
    # Shared step logic
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_valid_pixels(logits: Tensor, targets: Tensor, common_mask: Tensor) -> tuple[Tensor, Tensor]:
        valid = common_mask.squeeze(1) > 0.5
        preds = torch.argmax(logits, dim=1)
        return preds[valid], targets[valid]

    def _forward_and_get_loss(self, batch: dict[str, Any]):
        x_pre, x_post = batch["image_pre"], batch["image"]
        y = batch["mask"]
        common_mask = batch["mask-common"].to(torch.float32)
        common_mask = torch.nan_to_num(common_mask, nan=0.0, posinf=1.0, neginf=0.0)
        B = x_post.shape[0]

        if not (torch.isfinite(x_pre).all() and torch.isfinite(x_post).all()):
            raise RuntimeError("Input images contain NaN/Inf")

        # Patch near-empty samples to avoid LayerNorm NaN
        valid_ratio = common_mask.flatten(1).mean(dim=1)
        bad = valid_ratio < 0.05
        if bad.any():
            logger.warning("Patching %d/%d near-empty samples", bad.sum().item(), B)
            noise = torch.rand_like(x_pre[0:1]) * 0.01
            for idx in bad.nonzero(as_tuple=True)[0]:
                x_pre[idx] = noise[0]
                x_post[idx] = noise[0]
                common_mask[idx] = 0.0
                y[idx] = 0

        logits = self(x_pre, x_post)

        # Handle non-finite logits
        if not torch.isfinite(logits).all():
            logger.warning("Non-finite logits — returning zero loss")
            zero = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
            dummy_targets = torch.zeros((B, *x_post.shape[2:]), device=logits.device, dtype=torch.long)
            return x_pre, x_post, y.float(), dummy_targets, torch.zeros_like(logits), zero, zero, zero, B

        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        # Class-index targets for metrics and index-based losses.
        y_sq = y.squeeze(1) if y.dim() == 4 else y
        y_clamped = y_sq.clamp(0, self._effective_num_classes - 1).long()

        # --- Compute losses on VALID pixels only ---
        # SMP FocalLoss/LovaszLoss in binary mode expect:
        #   y_pred: [B, 1, H, W] raw logits
        #   y_true: [B, H, W]    long targets (0 or 1)
        #
        # We extract only valid pixels, then reshape them as a fake [N, 1, 1, 1]
        # batch so each "sample" is a single pixel. This ensures NO invalid pixel
        # contributes to the loss at all.
        valid_mask_2d = common_mask.squeeze(1) > 0.5  # [B, H, W]

        if valid_mask_2d.sum() == 0:
            zero = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
            return x_pre, x_post, y.float(), y_clamped, logits, zero, zero, zero, B

        C = logits.shape[1]

        # For binary (C=2): use class-1 logit
        if C == 2:
            burn_logits = logits[:, 1, :, :]  # [B, H, W]
        else:
            burn_logits = logits[:, 0, :, :]  # fallback

        # Extract valid pixels → 1D tensors
        valid_burn_logits = burn_logits[valid_mask_2d]  # [N]
        valid_targets = y_clamped[valid_mask_2d]  # [N]

        # Reshape to [N, 1, 1, 1] for SMP binary losses
        N = valid_burn_logits.shape[0]
        loss_logits = valid_burn_logits.reshape(N, 1, 1, 1)  # [N, 1, 1, 1]
        loss_targets = valid_targets.reshape(N, 1, 1)  # [N, 1, 1]

        w_ml, w_sl = self.loss_ratio
        dice_loss = self.main_loss(loss_logits, loss_targets)
        ce_loss = self.secondary_loss(loss_logits, loss_targets)
        fn_penalty = self._burned_false_negative_penalty(logits, y_clamped, common_mask)
        total_loss = w_ml * dice_loss + w_sl * ce_loss + fn_penalty

        if not torch.isfinite(total_loss):
            logger.warning(
                "Non-finite total_loss detected — ce=%.6f, dice=%.6f, fn_penalty=%.6f. Returning zero loss.",
                ce_loss.item(), dice_loss.item(), fn_penalty.item(),
            )
            zero = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
            return x_pre, x_post, y.float(), y_clamped, logits, zero, zero, zero, B

        return x_pre, x_post, y.float(), y_clamped, logits, total_loss, dice_loss, ce_loss, B

    def _update_metrics(self, prefix: str, logits: Tensor, targets: Tensor, common_mask: Tensor):
        valid_preds, valid_targets = self._extract_valid_pixels(logits, targets, common_mask)
        if valid_preds.numel() == 0:
            return
        for name in ("iou", "f1", "precision", "recall"):
            getattr(self, f"{prefix}_{name}").update(valid_preds, valid_targets)
        if hasattr(self, f"{prefix}_iou_classwise"):
            getattr(self, f"{prefix}_iou_classwise").update(valid_preds, valid_targets)

    def _log_and_reset_metrics(self, prefix: str):
        for name in ("iou", "f1", "precision", "recall"):
            metric = getattr(self, f"{prefix}_{name}")
            self.log(f"{prefix}_{name}", metric.compute(), prog_bar=True, sync_dist=True)
            metric.reset()

        if hasattr(self, f"{prefix}_iou_classwise"):
            cw = getattr(self, f"{prefix}_iou_classwise")
            for class_name, value in cw.compute().items():
                self.log(f"{prefix}_iou_{class_name}", value, prog_bar=False, sync_dist=True)
            cw.reset()

    # ------------------------------------------------------------------
    # Train / Val / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        _, _, _, targets, logits, total_loss, dice_loss, ce_loss, bs = self._forward_and_get_loss(batch)
        self.log("train_loss", total_loss, batch_size=bs, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("main_loss", dice_loss, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("ce_loss", ce_loss, on_epoch=True, sync_dist=True, batch_size=bs)
        with torch.no_grad():
            self._update_metrics("train", logits, targets, batch["mask-common"])
        return total_loss

    def on_train_epoch_end(self):
        self._log_and_reset_metrics("train")
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor | None:
        if not batch.get("has_mask", torch.tensor([True])).any():
            return None
        _, _, _, targets, logits, loss, _, _, bs = self._forward_and_get_loss(batch)
        self.log("val_loss", loss, batch_size=bs, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        with torch.no_grad():
            self._update_metrics("val", logits, targets, batch["mask-common"])
        return logits

    def on_validation_epoch_end(self):
        # Alias before reset.
        self.log("val_recall_burn", self.val_recall.compute(), prog_bar=True, sync_dist=True)
        self._log_and_reset_metrics("val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        if not batch.get("has_mask", torch.tensor([True])).any():
            return None
        x_pre, x_post, y, targets, logits, loss, _, _, bs = self._forward_and_get_loss(batch)
        self.log("test_loss", loss, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True)
        with torch.no_grad():
            self._update_metrics("test", logits, targets, batch["mask-common"])

        # Visualizations
        remaining = self.max_samples - self._total_samples_visualized
        if remaining > 0:
            self._total_samples_visualized += self._log_visualizations(
                trainer=self.trainer, batch=batch, outputs=logits,
                max_samples=min(remaining, len(x_post)), artifact_prefix="test", epoch_suffix=False,
            )

    def on_test_epoch_end(self):
        self._log_and_reset_metrics("test")

    # ------------------------------------------------------------------
    # Burned FN penalty
    # ------------------------------------------------------------------
    def _burned_false_negative_penalty(self, logits: Tensor, targets: Tensor, valid_mask: Tensor) -> Tensor:
        if self.burned_class_weight <= 1.0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        # Force float32 to avoid NaN from BCE under AMP/float16.
        burned_logits = (logits[:, 1:2] if logits.shape[1] > 1 else logits).float()
        burned_targets = (targets == 1).unsqueeze(1).float()
        vm = valid_mask.unsqueeze(1) if valid_mask.dim() == 3 else valid_mask
        vm = vm.float()

        penalty = F.binary_cross_entropy_with_logits(
            burned_logits, burned_targets,
            pos_weight=torch.tensor(self.burned_class_weight, device=logits.device, dtype=torch.float32),
            reduction="none",
        ) * vm
        return penalty.sum() / vm.sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def _log_visualizations(
        self, trainer: Trainer, batch: dict[str, Any], outputs: Tensor,
        max_samples: int, artifact_prefix: str = "val", *, epoch_suffix: bool = True,
    ) -> int:
        if batch is None or outputs is None:
            return 0
        try:
            images = batch["image"]
            pre_images = batch["image_pre"]
            names = batch["pre_post_name"]
            masks = batch["mask"].squeeze(1).long()
            has_mask = batch.get("has_mask", torch.tensor([True] * len(images)))
            common_mask = batch.get("mask-common")

            # Pick 3 data bands for RGB vis (skip band 0=COMMON_MASK, last 2=SAT_PASS/BEAM)
            nb = images.shape[1]
            avail = list(range(1, max(nb - 2, 2)))
            rgb = [avail[0], avail[len(avail) // 2], avail[-1]] if len(avail) >= 3 else avail[:3]

            logged = 0
            for i in range(min(max_samples, len(images))):
                diff = torch.abs(images[i] - pre_images[i])
                vis = diff[rgb]
                pred = torch.argmax(outputs[i], dim=0)
                if common_mask is not None:
                    pred = pred.clone()
                    pred[common_mask[i].squeeze(0) < 0.5] = self._effective_num_classes

                mask_i = masks[i] if (has_mask[i] if isinstance(has_mask, (list, Tensor)) else has_mask) else None
                fig = visualize_prediction(
                    image=vis, mask=mask_i, prediction=pred,
                    sample_name=names[i][:80], num_classes=self.num_classes, class_colors=self.class_colors,
                )

                safe = Path(names[i][:60].replace("|", "_").replace("/", "_")).stem
                tag = f"{artifact_prefix}/{safe}/idx_{i}"
                if epoch_suffix:
                    tag += f"_epoch_{trainer.current_epoch}"
                tag += ".png"

                if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "log_figure"):
                    trainer.logger.experiment.log_figure(
                        figure=fig, artifact_file=tag, run_id=getattr(trainer.logger, "run_id", None),
                    )
                elif isinstance(trainer.logger, TensorBoardLogger):
                    trainer.logger.experiment.add_figure(
                        tag=tag, figure=fig, global_step=trainer.current_epoch if epoch_suffix else 0,
                    )
                plt.close(fig)
                logged += 1
            return logged
        except Exception:
            logger.exception("Error in visualization logging")
            return 0

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        x_pre, x_post = batch["image_pre"], batch["image"]
        with torch.no_grad():
            logits = self(x_pre, x_post)

        probs = torch.softmax(logits, dim=1)
        y_pred = torch.argmax(probs, dim=1)

        # Mask invalid pixels
        mask_key = "mask-common" if "mask-common" in batch else ("water_mask" if "water_mask" in batch else None)
        if mask_key:
            invalid = batch[mask_key].squeeze(1) == 0 if mask_key == "mask-common" else batch[mask_key].squeeze(1) > 0
            y_pred = y_pred.masked_fill(invalid, NO_DATA)

        result = {
            "predictions": y_pred, "probabilities": probs, "logits": logits,
            "pre_post_name": batch["pre_post_name"], "cell_id": batch["cell_id"],
            "profile": batch["profile"],
            "original_height": batch["original_height"], "original_width": batch["original_width"],
        }
        for k in ("pair_id", "event_id", "db_nbac_fire_id", "group_date_pre", "group_date_post", "group_id_pre", "group_id_post"):
            if k in batch:
                result[k] = batch[k]
        if batch.get("has_mask", torch.tensor(False)).any():
            result["mask"] = batch["mask"]
        return result

    def on_predict_end(self) -> None:
        predictions = self.trainer.predict_loop.predictions
        if not predictions:
            logger.warning("No predictions to save.")
            return

        predict_date = datetime.now().strftime("%Y%m%d_%H%M")
        base_dir = Path(self.predict_output_dir or self.trainer.default_root_dir)
        if base_dir.name != "predictions":
            base_dir = base_dir / "predictions"

        group_tiles: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
        event_tiles: dict[str, list[Path]] = defaultdict(list)
        manifest = {
            "prediction_date": predict_date,
            "model_name": self.change_detection_model,
            "checkpoint": str(self.weights_from_checkpoint_path or ""),
            "base_dir": str(base_dir),
            "predictions": [],
        }

        for batch_result in predictions:
            self._save_batch_tiles(batch_result, base_dir, predict_date, group_tiles, event_tiles, manifest)

        # Write manifest once
        manifest_path = base_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info("Saved manifest to %s", manifest_path)

        self._merge_predictions(group_tiles, event_tiles)
        logger.info("All predictions saved to %s", base_dir)

    def _save_batch_tiles(self, batch_result, base_dir, predict_date, group_tiles, event_tiles, manifest):
        y_pred = batch_result["predictions"]
        profiles = batch_result["profile"]
        B = y_pred.shape[0]

        for i in range(B):
            cell_id = batch_result["cell_id"][i]
            name = batch_result["pre_post_name"][i].replace("\n", "").replace("|", "_").replace("/", "_")
            pair_id = self._extract_scalar(batch_result.get("pair_id"), i, default=None)
            event_id = self._extract_scalar(batch_result.get("event_id") or batch_result.get("db_nbac_fire_id"), i, default="unknown_event")
            gid_pre = self._extract_scalar(batch_result.get("group_id_pre"), i, default="all")
            gid_post = self._extract_scalar(batch_result.get("group_id_post"), i, default="all")
            gdate_pre = self._extract_scalar(batch_result.get("group_date_pre"), i, default="all")
            gdate_post = self._extract_scalar(batch_result.get("group_date_post"), i, default="all")

            oh = batch_result["original_height"]
            ow = batch_result["original_width"]
            orig_h = oh[i].item() if isinstance(oh, Tensor) else int(oh[i])
            orig_w = ow[i].item() if isinstance(ow, Tensor) else int(ow[i])
            pred_np = y_pred[i, :orig_h, :orig_w].cpu().numpy().astype(np.uint16)

            t_list = [profiles["transform"][k][i].item() for k in range(6)]
            crs_val = profiles["crs"][i] if isinstance(profiles["crs"], (list, tuple)) else profiles["crs"]
            profile_i = {
                "driver": "GTiff", "dtype": "uint16", "count": 1, "nodata": 32767,
                "height": orig_h, "width": orig_w, "crs": crs_val, "transform": Affine(*t_list),
                "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256, "predictor": 2,
            }

            event_dir = base_dir / event_id / predict_date
            tile_dir = event_dir / cell_id
            tile_dir.mkdir(parents=True, exist_ok=True)
            out_path = tile_dir / f"{pair_id}-{name}.tif"

            with rio.open(str(out_path), "w", **profile_i) as dst:
                dst.write(pred_np[np.newaxis])

            edk = str(event_dir)
            group_tiles[(edk, str(gid_pre), str(gid_post))].append(out_path)
            event_tiles[edk].append(out_path)

            logger.info("Saved %s (%dx%d)", out_path, orig_w, orig_h)
            manifest["predictions"].append({
                "pair_id": pair_id, "event_id": event_id, "cell_id": cell_id,
                "group_id_pre": gid_pre, "group_id_post": gid_post,
                "group_date_pre": gdate_pre, "group_date_post": gdate_post,
                "tif_path": str(out_path),
            })

    @staticmethod
    def _extract_scalar(field, index: int, default: str = "unknown") -> str:
        if field is None:
            return default
        if isinstance(field, Tensor):
            return str(field[index].item())
        if isinstance(field, (list, tuple)):
            return str(field[index])
        return str(field)

    @staticmethod
    def _merge_tiles(tile_paths: list[Path], out_path: Path, label: str) -> None:
        if len(tile_paths) < 2:
            logger.info("Skip merge %s (only %d tile)", label, len(tile_paths))
            return
        datasets = []
        try:
            datasets = [rio.open(str(p)) for p in tile_paths]
            mosaic, transform = rio_merge(datasets)
            profile = datasets[0].profile.copy()
            dtype_name = str(profile.get("dtype", ""))
            profile.update(
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform,
                compress="lzw",
                tiled=True,
                blockxsize=256,
                blockysize=256,
                predictor=2 if _is_integer_dtype(dtype_name) else 3,
            )
            with rio.open(str(out_path), "w", **profile) as dst:
                dst.write(mosaic)
            logger.info("Merged %d tiles → %s (%dx%d)", len(tile_paths), out_path, mosaic.shape[2], mosaic.shape[1])
        except Exception:
            logger.exception("Failed merge: %s", label)
        finally:
            for ds in datasets:
                try:
                    ds.close()
                except Exception:
                    pass

    @classmethod
    def _merge_predictions(cls, group_tiles, event_tiles) -> None:
        for (edk, gpre, gpost), paths in group_tiles.items():
            cls._merge_tiles(paths, Path(edk) / f"merged_group_{gpre}_{gpost}.tif", f"group {gpre}/{gpost}")
        for edk, paths in event_tiles.items():
            cls._merge_tiles(paths, Path(edk) / "merged_all.tif", f"global {edk}")