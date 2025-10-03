"""SetCriterion for Mask2Former."""

import torch
import torch.nn.functional as fn
from torch import Tensor, nn

from .loss_utils import (
    dice_loss_with_num_masks,
    get_src_permutation_idx,
    sigmoid_focal_loss,
)
from .matcher import HungarianMatcher


class SetCriterion(nn.Module):
    """
    Mask2Former loss computation.

    This class computes the loss for Mask2Former, which consists of:
        - Classification loss (cross entropy on matched queries)
        - Mask loss (focal/BCE loss on matched masks)
        - Dice loss (dice loss on matched masks)

    The process:
        1. Use HungarianMatcher for optimal prediction-target assignment
        2. Compute losses only on matched prediction-target pairs
        3. Aggregate losses with specified weights
        4. Include auxiliary losses from intermediate decoder layers

    Args:
        num_classes: Number of object categories (excluding background)
        matcher: HungarianMatcher module for optimal assignment
        weight_dict: Dict with loss names as keys and their weights as values.
                    Expected keys: "loss_ce", "loss_mask", "loss_dice"
        eos_coef: Relative classification weight applied to the no-object category
        losses: List of losses to compute. Can include: "labels", "masks"
        num_points: Number of points to sample for mask loss computation (efficiency)
        oversample_ratio: Oversampling ratio for point sampling
        importance_sample_ratio: Ratio of points sampled via importance sampling

    Example:
        >>> num_classes = 4  # Your semantic classes
        >>> matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0)
        >>> weight_dict = {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}
        >>> criterion = SetCriterion(
        ...     num_classes=num_classes,
        ...     matcher=matcher,
        ...     weight_dict=weight_dict,
        ...     eos_coef=0.1,
        ...     losses=["labels", "masks"],
        ... )

    """

    def __init__(  # noqa: PLR0913
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float],
        eos_coef: float = 0.1,
        losses: list[str] | None = None,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
    ) -> None:
        """Initialize SetCriterion."""
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses or ["labels", "masks"]
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        # Weight for the no-object class
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """
        Compute the losses.

        Args:
            outputs: Dict with keys:
                - "pred_logits": Tensor of shape [B, Q, C+1] (class predictions)
                - "pred_masks": Tensor of shape [B, Q, H, W] (mask predictions)
                - "aux_outputs": Optional list of intermediate prediction dicts
            targets: List of dicts (one per batch element) with keys:
                - "labels": Tensor of shape [N] (ground truth class indices)
                - "masks": Tensor of shape [N, H, W] (ground truth binary masks)

        Returns:
            Dict of losses with keys like "loss_ce", "loss_mask", "loss_dice"

        """
        # Compute losses for main outputs (exclude auxiliary outputs)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target masks for normalization
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = (
            torch.clamp(num_masks / torch.distributed.get_world_size(), min=1).item()
            if torch.distributed.is_initialized()
            else torch.clamp(num_masks, min=1).item()
        )
        # Compute all requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs_without_aux, targets, indices, num_masks),
            )
        # Compute auxiliary losses (if present)
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_masks,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def get_loss(
        self,
        loss: str,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_masks: float,
    ) -> dict[str, Tensor]:
        """
        Dispatch to the appropriate loss function.

        Args:
            loss: Loss name ("labels" or "masks")
            outputs: Model outputs
            targets: Ground truth targets
            indices: Matched indices from Hungarian matcher
            num_masks: Number of masks for normalization

        Returns:
            Dict with computed losses

        """
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            msg = f"Loss {loss} not supported"
            raise ValueError(msg)
        return loss_map[loss](outputs, targets, indices, num_masks)

    def loss_labels(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_masks: float,  # noqa: ARG002
    ) -> dict[str, Tensor]:
        """
        Classification loss (cross entropy).

        Computes cross entropy loss on the classification logits.

        Args:
            outputs: Dict with "pred_logits" key
            targets: List of target dicts with "labels" key
            indices: Matched indices from matcher
            num_masks: Not used for classification loss

        Returns:
            Dict with "loss_ce" key

        """
        if "pred_logits" not in outputs:
            msg = "pred_logits not in outputs"
            raise KeyError(msg)
        src_logits = outputs["pred_logits"]  # [B, Q, C+1]
        # Get indices of matched predictions and targets
        idx = get_src_permutation_idx(indices)
        # Gather target classes for matched predictions
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices, strict=False)],
        )
        # Create full target tensor (most queries are "no object" = num_classes)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        # Compute cross entropy loss
        loss_ce = fn.cross_entropy(
            src_logits.transpose(1, 2),  # [B, C+1, Q]
            target_classes,
            self.empty_weight,
        )
        return {"loss_ce": loss_ce}

    def loss_masks(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_masks: float,
    ) -> dict[str, Tensor]:
        """
        Mask losses (focal/BCE + dice).

        Computes mask losses using point sampling for memory efficiency.

        Args:
            outputs: Dict with "pred_masks" key
            targets: List of target dicts with "masks" key
            indices: Matched indices from matcher
            num_masks: Number of masks for normalization

        Returns:
            Dict with "loss_mask" and "loss_dice" keys

        """
        if "pred_masks" not in outputs:
            msg = "pred_masks not in outputs"
            raise KeyError(msg)

        src_idx = get_src_permutation_idx(indices)
        # tgt_idx = get_tgt_permutation_idx(indices)
        # Get matched predictions and targets
        src_masks = outputs["pred_masks"]  # [B, Q, H, W]
        src_masks = src_masks[src_idx]  # [num_matched, H, W]
        # Concatenate all target masks
        target_masks = torch.cat(
            [t["masks"][i] for t, (_, i) in zip(targets, indices, strict=False)],
        )  # [num_matched, H, W]
        target_masks = target_masks.to(src_masks)
        # Upsample predictions to target size if needed
        if src_masks.shape[-2:] != target_masks.shape[-2:]:
            src_masks = fn.interpolate(
                src_masks.unsqueeze(1),
                size=target_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        # Flatten spatial dimensions
        src_masks = src_masks.flatten(1)  # [num_matched, H*W]
        target_masks = target_masks.flatten(1)  # [num_matched, H*W]
        # Point sampling for memory efficiency
        with torch.no_grad():
            # Sample points
            point_coords = self._get_point_coords_from_masks(
                target_masks,
                num_masks,
            )  # [num_matched, num_points, 2]
        # Sample from masks at point coordinates
        src_masks_sampled = self._point_sample(
            src_masks.unsqueeze(1),  # [num_matched, 1, H*W]
            point_coords,
            align_corners=False,
        ).squeeze(1)  # [num_matched, num_points]

        target_masks_sampled = self._point_sample(
            target_masks.unsqueeze(1),  # [num_matched, 1, H*W]
            point_coords,
            align_corners=False,
        ).squeeze(1)  # [num_matched, num_points]

        # Compute focal loss
        with torch.autocast(device_type="cuda", enabled=False):
            src_masks_sampled = src_masks_sampled.float()
            target_masks_sampled = target_masks_sampled.float()

            loss_mask = (
                sigmoid_focal_loss(src_masks_sampled, target_masks_sampled).sum()
                / num_masks
            )

        # Compute dice loss (on full masks, not sampled)
        loss_dice = dice_loss_with_num_masks(src_masks, target_masks, num_masks)
        return {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
        }

    def _get_point_coords_from_masks(
        self,
        masks: Tensor,
        num_masks: float,  # noqa: ARG002
    ) -> Tensor:
        """
        Get point coordinates for sampling using importance + random sampling.

        Args:
            masks: Masks of shape [N, H*W]
            num_masks: Number of masks for normalization

        Returns:
            Point coordinates of shape [N, num_points, 2] in range [0, H*W-1]

        """
        n, hw = masks.shape

        # Determine number of points to sample
        num_points_total = self.num_points
        # Split into importance sampling and random sampling
        num_points_importance = int(num_points_total * self.importance_sample_ratio)
        num_points_random = num_points_total - num_points_importance
        # Importance sampling: sample points with uncertainty (near decision boundary)
        # For binary masks, uncertainty is where mask values are close to 0.5
        with torch.no_grad():
            # Calculate uncertainty (for binary masks, use absolute distance from 0.5)
            uncertainty = torch.abs(masks - 0.5)  # [N, H*W]
            # Convert to probability (higher uncertainty = higher sampling probability)
            uncertainty = 1 - uncertainty  # Now higher values = more uncertainty
            # Sample points based on uncertainty
            importance_indices = torch.multinomial(
                uncertainty + 1e-6,  # Add small value to avoid zeros
                num_points_importance,
                replacement=True,
            )  # [N, num_points_importance]
        # Random sampling: uniformly sample remaining points
        random_indices = torch.randint(
            0,
            hw,
            (n, num_points_random),
            device=masks.device,
        )  # [N, num_points_random]
        # Combine importance and random samples
        point_indices = torch.cat(
            [importance_indices, random_indices],
            dim=1,
        )  # [N, num_points]
        # Convert linear indices to 2D coordinates
        # For flat masks, we can use indices directly
        # Create dummy 2D coords (just use linear indices twice for compatibility)
        return torch.stack(
            [point_indices, point_indices],
            dim=-1,
        ).float()  # [N, num_points, 2]

    def _point_sample(
        self,
        input_tensor: Tensor,
        point_coords: Tensor,
        *,
        align_corners: bool = False,  # noqa: ARG002
    ) -> Tensor:
        """
        Sample features at given point indices (for flattened masks).

        Args:
            input_tensor: Tensor of shape [N, C, H*W]
            point_coords: Indices of shape [N, num_points, 2]
            align_corners: Not used for flat masks

        Returns:
            Sampled features of shape [N, C, num_points]

        """
        # For flattened masks, just gather along the spatial dimension
        point_indices = point_coords[..., 0].long()  # [N, num_points]
        # Gather features at the point indices
        return torch.gather(
            input_tensor,
            dim=-1,
            index=point_indices.unsqueeze(1).expand(-1, input_tensor.shape[1], -1),
        )  # [N, C, num_points]

    def __repr__(self) -> str:
        """Representation of the criterion."""
        head = "Criterion " + self.__class__.__name__
        body = [
            f"num_classes: {self.num_classes}",
            f"eos_coef: {self.eos_coef}",
            f"losses: {self.losses}",
            f"weight_dict: {self.weight_dict}",
            f"num_points: {self.num_points}",
        ]
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)


def build_criterion(  # noqa: PLR0913
    num_classes: int,
    weight_dict: dict[str, float] | None = None,
    eos_coef: float = 0.1,
    matcher_cost_class: float = 2.0,
    matcher_cost_mask: float = 5.0,
    matcher_cost_dice: float = 5.0,
    num_points: int = 12544,
) -> SetCriterion:
    """
    Build a SetCriterion with default settings.

    Convenience function for creating a criterion.

    Args:
        num_classes: Number of object categories (excluding background)
        weight_dict: Loss weights. If None, uses defaults.
        eos_coef: Weight for no-object class
        matcher_cost_class: Classification cost weight for matcher
        matcher_cost_mask: Mask cost weight for matcher
        matcher_cost_dice: Dice cost weight for matcher
        num_points: Number of points to sample for losses

    Returns:
        Configured SetCriterion instance

    Example:
        >>> criterion = build_criterion(
        ...     num_classes=4,
        ...     weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0},
        ... )

    """
    if weight_dict is None:
        weight_dict = {
            "loss_ce": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0,
        }
    # Build matcher
    matcher = HungarianMatcher(
        cost_class=matcher_cost_class,
        cost_mask=matcher_cost_mask,
        cost_dice=matcher_cost_dice,
        num_points=num_points,
    )
    # Build criterion
    return SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=["labels", "masks"],
        num_points=num_points,
    )


# if __name__ == "__main__":
#     print("Testing SetCriterion...")
#     # Create criterion
#     print("\n1. Building criterion...")
#     num_classes = 4
#     criterion = build_criterion(
#         num_classes=num_classes,
#         weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0},
#         eos_coef=0.1,
#         num_points=1000,  # Smaller for testing
#     )
#     print(criterion)
#     # Create dummy outputs (like from Mask2Former decoder)
#     print("\n2. Creating dummy outputs...")
#     batch_size, num_queries = 2, 100
#     h, w = 128, 128
#     outputs = {
#         "pred_logits": torch.randn(batch_size, num_queries, num_classes + 1),
#         "pred_masks": torch.randn(batch_size, num_queries, h, w),
#         "aux_outputs": [
#             {
#                 "pred_logits": torch.randn(batch_size, num_queries, num_classes + 1),
#                 "pred_masks": torch.randn(batch_size, num_queries, h, w),
#             }
#             for _ in range(2)  # 2 auxiliary outputs
#         ],
#     }
#     # Create dummy targets (semantic segmentation format)
#     print("3. Creating dummy targets...")
#     targets = [
#         {
#             "labels": torch.tensor([0, 1, 2]),  # 3 classes present
#             "masks": torch.randint(0, 2, (3, h, w)).float(),
#         },
#         {
#             "labels": torch.tensor([0, 3]),  # 2 classes present
#             "masks": torch.randint(0, 2, (2, h, w)).float(),
#         },
#     ]
#     # Compute losses
#     print("\n4. Computing losses...")
#     losses = criterion(outputs, targets)
#     print("\n5. Loss values:")
#     for k, v in losses.items():
#         print(f"   {k}: {v.item():.4f}")
#     # Test loss aggregation
#     print("\n6. Computing weighted total loss...")
#     total_loss = sum(
#         losses[k] * criterion.weight_dict[k]
#         for k in ["loss_ce", "loss_mask", "loss_dice"]
#         if k in losses
#     )
#     print(f"   Total loss: {total_loss.item():.4f}")
#     # Test auxiliary loss aggregation
#     aux_loss_keys = [
#         k for k in losses.keys() if "_" in k and k.split("_")[0] in ["loss"]
#     ]
#     if aux_loss_keys:
#         print(f"\n7. Auxiliary losses found: {len(aux_loss_keys)} losses")
#         aux_total = sum(losses[k] for k in aux_loss_keys)
#         print(f"   Auxiliary total: {aux_total.item():.4f}")
#     print("\n✅ All tests passed!")
