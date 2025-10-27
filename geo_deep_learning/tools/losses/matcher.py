"""Hungarian Matcher for Mask2Former."""

import logging

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from .loss_utils import batch_dice_loss_jit, batch_sigmoid_ce_loss_jit, point_sample

logger = logging.getLogger(__name__)


class HungarianMatcher(nn.Module):
    """Hungarian Matcher for optimal assignment between predictions and targets."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_points: int = 12544,
    ) -> None:
        """Initialize the Hungarian Matcher."""
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            msg = "All costs cannot be 0"
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Representation of the Hungarian Matcher."""
        head = "Matcher " + self.__class__.__name__
        body = [
            f"cost_class: {self.cost_class}",
            f"cost_mask: {self.cost_mask}",
            f"cost_dice: {self.cost_dice}",
        ]
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def _compute_mask_costs_with_sampling(
        self,
        out_mask: Tensor,
        tgt_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute mask costs using point sampling.

        Samples random points from the masks for memory-efficient computation.

        Args:
            out_mask: Predicted masks [B*Q, H, W]
            tgt_mask: Target masks [total_targets, H, W]

        Returns:
            Tuple of (focal_loss_cost, dice_loss_cost)

        """
        out_mask = out_mask[:, None]
        tgt_mask = tgt_mask[:, None]
        # Sample points
        point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)

        # Sample from predicted masks
        out_mask_sampled = point_sample(
            out_mask,
            point_coords.repeat(out_mask.shape[0], 1, 1),
            align_corners=False,
        ).squeeze(1)

        # Sample from target masks
        tgt_mask_sampled = point_sample(
            tgt_mask,
            point_coords.repeat(tgt_mask.shape[0], 1, 1),
            align_corners=False,
        ).squeeze(1)

        with torch.autocast(device_type="cuda", enabled=False):
            out_mask_sampled = out_mask_sampled.float()
            tgt_mask_sampled = tgt_mask_sampled.float()
            cost_mask = batch_sigmoid_ce_loss_jit(
                out_mask_sampled,
                tgt_mask_sampled,
            )
            # Compute dice loss
            cost_dice = batch_dice_loss_jit(out_mask_sampled, tgt_mask_sampled)
        return cost_mask, cost_dice

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> list[tuple[Tensor, Tensor]]:
        """
        Perform the matching.

        Args:
            outputs: Dict with keys:
                - "pred_logits": Tensor of shape [B, Q, C+1] (class predictions)
                - "pred_masks": Tensor of shape [B, Q, H, W] (mask predictions)
            targets: List of dicts (one per batch element) with keys:
                - "labels": Tensor of shape [N] (ground truth class indices)
                - "masks": Tensor of shape [N, H, W] (ground truth binary masks)

        Returns:
            List of (src_idx, tgt_idx) tuples, one per batch element.
            src_idx: Indices of matched predictions
            tgt_idx: Indices of matched targets

        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        # Flatten to compute cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*Q, C+1]
        out_mask = outputs["pred_masks"].flatten(0, 1)  # [B*Q, H, W]
        # Concatenate target labels and masks
        tgt_ids = torch.cat([v["labels"] for v in targets])  # [total_targets]
        tgt_mask = torch.cat([v["masks"] for v in targets])  # [total_targets, H, W]
        # Compute classification cost: -prob[target class]
        # Use negative because we want to maximize probability (minimize negative prob)
        cost_class = -out_prob[:, tgt_ids]  # [B*Q, total_targets]
        cost_mask, cost_dice = self._compute_mask_costs_with_sampling(
            out_mask,
            tgt_mask,
        )
        # Final cost matrix: weighted sum of all costs
        C = (
            self.cost_mask * cost_mask
            + self.cost_class * cost_class
            + self.cost_dice * cost_dice
        )  # [B*Q, total_targets]

        if torch.isnan(C).any() or torch.isinf(C).any():
            logger.warning("NaN/Inf detected in cost matrix!")
            logger.warning(
                "cost_class: min=%.2f, max=%.2f, nan=%d",
                cost_class.min().item(),
                cost_class.max().item(),
                torch.isnan(cost_class).sum().item(),
            )
            logger.warning(
                "cost_mask: min=%.2f, max=%.2f, nan=%d",
                cost_mask.min().item(),
                cost_mask.max().item(),
                torch.isnan(cost_mask).sum().item(),
            )
            logger.warning(
                "cost_dice: min=%.2f, max=%.2f, nan=%d",
                cost_dice.min().item(),
                cost_dice.max().item(),
                torch.isnan(cost_dice).sum().item(),
            )
            logger.warning(
                "logits: min=%.2f, max=%.2f",
                outputs["pred_logits"].min().item(),
                outputs["pred_logits"].max().item(),
            )
        C = torch.nan_to_num(C, nan=1e8, posinf=1e8, neginf=-1e8)
        C = C.view(batch_size, num_queries, -1).cpu()
        # Compute optimal assignment for each batch element
        sizes = [len(v["labels"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        # Convert to torch tensors
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def build_matcher(
    cost_class: float = 2.0,
    cost_mask: float = 5.0,
    cost_dice: float = 5.0,
    num_points: int | None = 12544,
) -> HungarianMatcher:
    """
    Build a HungarianMatcher with specified costs.

    Convenience function for creating a matcher.

    Args:
        cost_class: Weight for classification cost
        cost_mask: Weight for mask cost
        cost_dice: Weight for dice cost
        num_points: Number of points to sample (None for full masks)

    Returns:
        Configured HungarianMatcher instance

    """
    return HungarianMatcher(
        cost_class=cost_class,
        cost_mask=cost_mask,
        cost_dice=cost_dice,
        num_points=num_points,
    )
