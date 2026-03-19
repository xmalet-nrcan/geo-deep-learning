"""Segmentation SegFormer model."""

import logging
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
import kornia as krn
import torch
from datetime import datetime
from rasterio.transform import Affine
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import TensorBoardLogger
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
            self.train_iou = BinaryJaccardIndex(threshold=self.threshold)
            self.val_iou = BinaryJaccardIndex(threshold=self.threshold)
            self.test_iou = BinaryJaccardIndex(threshold=self.threshold)
        else:
            self.train_iou = JaccardIndex(task=task_type, num_classes=num_classes)
            self.val_iou = JaccardIndex(task=task_type, num_classes=num_classes)
            self.test_iou = JaccardIndex(task=task_type, num_classes=num_classes)

        self.train_f1 = F1Score(task=task_type, num_classes=num_classes)
        self.val_f1 = F1Score(task=task_type, num_classes=num_classes)
        self.test_f1 = F1Score(task=task_type, num_classes=num_classes)

        self.predict_output_dir = predict_output_dir

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
        aug = AugmentationSequential(
            krn.augmentation.PadTo(size=self.image_size, pad_mode='constant', pad_value=0, keepdim=False),
            data_keys=None,
        )

        keys_to_pad = {"image_pre": batch["image_pre"],
                       "image": batch["image"]}

        # En predict, mask et mask-common sont toujours présents dans votre dataset
        # car __getitem__ les retourne toujours
        for mask_names in ['mask','mask-common','water_mask']:
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

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training:
            return batch
        aug = self._apply_aug()
        device = batch["image"].device

        keys_to_aug = {
            "image_pre": batch["image_pre"],
            "image": batch["image"],
        }
        if "mask-common" in batch:
            keys_to_aug["mask-common"] = batch["mask-common"].to(torch.float32)
        if "mask" in batch:
            keys_to_aug["mask"] = batch["mask"]

        transformed = aug(keys_to_aug)
        for key in transformed:
            batch[key] = transformed[key].to(device, non_blocking=True)

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
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log("main_loss", main_loss, on_epoch=True, sync_dist=True)
        self.log("ce_loss", ce_loss, on_epoch=True, sync_dist=True)

        # --- Calcul des métriques différé (pour éviter de casser autograd) ---
        with torch.no_grad():
            # On accumule les prédictions pour calculer les métriques à la fin
            self.train_iou.update(logits, one_hot)
            self.train_f1.update(logits, one_hot)

        return loss

    def on_train_epoch_end(self):
        self.log("train_iou", self.train_iou.compute(), prog_bar=True, sync_dist=True)
        self.log("train_f1", self.train_f1.compute(), prog_bar=True, sync_dist=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)

        self.train_iou.reset()
        self.train_f1.reset()

    def validation_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run validation step."""
        has_mask = batch.get("has_mask", torch.tensor([True]))
        if not has_mask.any():
            return None  # skip ce batch
        x_pre, x_post, y, one_hot, logits, loss, main_loss, ce_loss, batch_size = self._forward_and_get_loss(batch)

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
        with torch.no_grad():
            y_pred = torch.argmax(logits, dim=1)
            y_true = torch.argmax(one_hot, dim=1)
            self.val_iou_classwise.update(y_pred, y_true)

        self.val_iou(logits, one_hot)
        self.val_f1(logits, one_hot)

        self.log("val_iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return logits

    def on_validation_epoch_end(self):
        classwise_iou = self.val_iou_classwise.compute()

        for class_name, value in classwise_iou.items():
            self.log(
                f"val_iou_{class_name}",
                value,
                prog_bar=False,
                sync_dist=True,
            )

        self.val_iou_classwise.reset()

    def test_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run test step."""

        has_mask = batch.get("has_mask", torch.tensor([True]))
        if not has_mask.any():
            return None

        x_pre, x_post, y, one_hot, logits, loss, main_loss, ce_loss, batch_size = self._forward_and_get_loss(batch)
        # Convert logits to class predictions
        y_pred = torch.argmax(logits, dim=1)
        y_true = torch.argmax(one_hot, dim=1)

        # --- Update metrics ---
        with torch.no_grad():
            self.test_iou_classwise.update(y_pred, y_true)
            self.test_iou.update(logits, one_hot)
            self.test_f1.update(logits, one_hot)

        # --- Log test loss (epoch-aggregated) ---
        self.log(
            "test_loss",
            loss,
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

        # --- Reset metrics ---
        self.test_iou_classwise.reset()
        self.test_iou.reset()
        self.test_f1.reset()

    def _forward_and_get_loss(self, batch: dict[str, Any]) -> tuple[
        Any, Any, Any, Tensor, Any, float | Any, Any, Any, Any
    ]:
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

        if logger.isEnabledFor(logging.DEBUG):
            with torch.no_grad():
                y_cpu = y.detach().cpu()
                logger.debug(
                    "mask min: %s, max: %s, unique (échantillon): %s",
                    y_cpu.min().item(),
                    y_cpu.max().item(),
                    torch.unique(y_cpu)[:20],
                )

        logits = self(x_pre, x_post)  # [B, C, H, W]
        y_float = y.float()
        logits_no_nan = torch.nan_to_num(logits, nan=1e15)

        # Vérifier les logits avant masquage
        if not torch.isfinite(logits).all():
            with torch.no_grad():

                logger.debug(
                    "[DEBUG] %s - %s logits non-finis: min=%s, max=%s, mean=%s",
                    batch['image_pre_name'],
                    batch['image_name_post'],
                    torch.min(logits_no_nan).item() if torch.isfinite(logits_no_nan).any() else "NaN",
                    torch.max(logits_no_nan).item() if torch.isfinite(logits_no_nan).any() else "NaN",
                    torch.nanmean(logits).item() if torch.isfinite(logits).any() else "NaN",
                )
            raise RuntimeError("Logits contain non-finite values (NaN/Inf) before masking.")


        num_classes = self.num_classes + 1 if self.num_classes == 1 else self.num_classes

        # Préparation du one-hot
        y_one_hot = y.squeeze(1) if y.dim() == 4 else y
        y_one_hot = y_one_hot.clamp(min=0, max=num_classes - 1)
        one_hot = torch.nn.functional.one_hot(y_one_hot.long(), num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).contiguous().float()

        # Optionnel: appliquer aussi le masque sur le one_hot (pour ignorer les no-data)
        # if common_data_mask.shape[-2:] == one_hot.shape[-2:]:
        #     one_hot = one_hot * common_data_mask

        # Vérifier qu'il reste des pixels valides
        valid_sum = common_data_mask.sum()
        if valid_sum == 0:
            # Eviter NaN si la loss divise par le nombre de pixels
            main_loss = torch.tensor(0.0, device=logits_no_nan.device, dtype=logits_no_nan.dtype)
            ce_loss = torch.tensor(0.0, device=logits_no_nan.device, dtype=logits_no_nan.dtype)
            loss = main_loss
            return x_pre, x_post, y_float, one_hot, logits_no_nan, loss, main_loss, ce_loss, batch_size

        w_ml, w_sl = self.loss_ratio

        # Vérifier entrées de la loss
        if not torch.isfinite(one_hot).all():
            raise RuntimeError("One-hot targets contain non-finite values (NaN/Inf).")

        # --- Losses ---
        ce_loss = self.secondary_loss(logits_no_nan.contiguous(), one_hot)
        loss = self.main_loss(logits_no_nan.contiguous(), one_hot)
        main_loss = w_sl * ce_loss + w_ml * loss

        # Dernière vérification
        if not torch.isfinite(main_loss):
            raise RuntimeError(
                f"Computed loss is NaN/Inf. "
                f"ce_loss={ce_loss.detach().cpu().item()}, "
                f"loss={loss.detach().cpu().item()}"
            )

        return x_pre, x_post, y_float, one_hot, logits_no_nan, main_loss, loss, ce_loss, batch_size

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
            image_batch = batch["image"]
            pre_image_batch = batch["image_pre"]
            c_batch_size = len(image_batch)
            logger.info("Batch size: %d", c_batch_size)
            batch_image_name = batch["pre_post_name"]
            num_samples = min(max_samples, c_batch_size)
            has_mask_flags = batch.get("has_mask", torch.tensor([True] * len(image_batch)))
            mask_batch = batch["mask"].squeeze(1).long()
            for i in range(num_samples):
                image = image_batch[i]
                pre_image = pre_image_batch[i]
                image_name = batch_image_name[i].replace('\n','')
                image_diff = torch.abs(image - pre_image)
                # mean = mean_batch[i]
                # std = std_batch[i]
                # image = denormalization(image, mean=mean, std=std)
                mask_i = mask_batch[i]
                has_real_mask = has_mask_flags[i] if isinstance(has_mask_flags,
                                                                (list, torch.Tensor)) else has_mask_flags
                fig = visualize_prediction(
                    image=image_diff[[2,3,4], :, :],
                    mask=mask_i if has_real_mask else None,  # None si pas de vrai masque
                    prediction=torch.argmax(outputs[i], dim=0),
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

    def predict_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        """Run prediction step (inference only, no loss/metrics)."""
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

        # Inclure le masque de vérité-terrain si disponible (pas toujours le cas en predict)
        if batch.get("has_mask", torch.tensor(False)).any():
            result["mask"] = batch["mask"]

        return result

    def on_predict_end(self) -> None:
        """Appelé après que tous les predict_step soient terminés."""
        predictions = self.trainer.predict_loop.predictions
        if not predictions:
            logger.warning("No predictions to save.")
            return

        # --- Build the output directory ---
        predict_date = datetime.now().strftime("%Y-%m-%d_%H%M")
        if self.predict_output_dir is not None:
            output_dir = Path(self.predict_output_dir)
            if output_dir.name != "predictions" :
                output_dir = output_dir / "predictions"

        else:
            output_dir = Path(self.trainer.default_root_dir) / "predictions"

        output_dir = output_dir / predict_date
        logger.info(f"Saving predictions to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        for batch_result in predictions:
            batch_cell_id = batch_result['cell_id']
            y_pred = batch_result["predictions"]  # [B, H_padded, W_padded]
            names = batch_result["pre_post_name"]
            batch_profiles = batch_result["profile"]
            orig_heights = batch_result["original_height"]  # Tensor [B] ou list
            orig_widths = batch_result["original_width"]  # Tensor [B] ou list
            batch_size = y_pred.shape[0]

            for i in range(batch_size):
                cell_id = batch_cell_id[i]
                sample_name = names[i].replace('\n', '').replace('|', '_').replace('/', '_')

                # --- Récupérer les dimensions originales ---
                orig_h = orig_heights[i].item() if isinstance(orig_heights, torch.Tensor) else int(orig_heights[i])
                orig_w = orig_widths[i].item() if isinstance(orig_widths, torch.Tensor) else int(orig_widths[i])

                # --- Découper le padding (crop au coin supérieur-gauche) ---
                pred_np = y_pred[i, :orig_h, :orig_w].cpu().numpy().astype(np.uint8)

                # --- Reconstruire le profil rasterio ---
                crs_val = batch_profiles["crs"][i] if isinstance(batch_profiles["crs"], (list, tuple)) else batch_profiles["crs"]

                transform_raw = batch_profiles["transform"]
                print(transform_raw)
                if isinstance(transform_raw, torch.Tensor):
                    t_list = transform_raw[i].tolist()
                elif isinstance(transform_raw, list) and len(transform_raw) > 0 and isinstance(transform_raw[0],
                                                                                               (list, torch.Tensor)):
                    t_list = transform_raw[i] if isinstance(transform_raw[i], list) else transform_raw[i].tolist()
                else:
                    t_list = transform_raw
                t_list = [transform_raw[k][i].item() for k in range(6)]

                print(t_list)

                profile_i = {
                    "driver": "GTiff",
                    "dtype": "uint8",
                    "count": 1,
                    "height": orig_h,  # ← dimensions ORIGINALES, pas paddées
                    "width": orig_w,  # ← dimensions ORIGINALES, pas paddées
                    "crs": crs_val,
                    "transform": Affine(*t_list),
                }
                (output_dir / cell_id ).mkdir(parents=True, exist_ok=True)
                out_path = output_dir / cell_id / f"{sample_name}.tif"
                logger.info(f"Saving predictions to {out_path}")
                with rio.open(str(out_path), "w", **profile_i) as dst:
                    dst.write(pred_np[np.newaxis, :, :])  # (1, orig_h, orig_w)

                logger.info("Saved prediction to %s (%dx%d)", out_path, orig_w, orig_h)

        logger.info("All predictions saved to %s", output_dir)



