"""Hungarian Matcher for Mask2Former."""

import torch
import torch.nn.functional as fn
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from .loss_utils import sigmoid_focal_loss


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for optimal assignment between predictions and targets.

    This module computes a cost matrix between predictions and ground truth,
    then uses the Hungarian algorithm to find the optimal one-to-one matching.

    The cost matrix is computed as a weighted sum of:
        - Classification cost (cross entropy)
        - Mask cost (focal loss or BCE)
        - Dice cost (dice loss)

    Args:
        cost_class: Weight for classification cost (default: 2.0)
        cost_mask: Weight for mask cost (default: 5.0)
        cost_dice: Weight for dice cost (default: 5.0)
        num_points: Number of points to sample for mask cost computation.
                   If None, uses full masks. (default: 12544)

    Example:
        >>> matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0)
        >>> outputs = {
        ...     "pred_logits": torch.randn(2, 100, 5),  # [B, Q, C+1]
        ...     "pred_masks": torch.randn(2, 100, 128, 128),  # [B, Q, H, W]
        ... }
        >>> targets = [
        ...     {"labels": torch.tensor([0, 1, 2]), "masks": torch.randn(3, 128, 128)},
        ...     {"labels": torch.tensor([0, 3]), "masks": torch.randn(2, 128, 128)},
        ... ]
        >>> indices = matcher(outputs, targets)
        >>> # indices is a list of (src_idx, tgt_idx) tuples, one per batch

    """

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
        # Compute mask costs
        if self.num_points is not None:
            # Point-based sampling for memory efficiency
            cost_mask, cost_dice = self._compute_mask_costs_with_sampling(
                out_mask,
                tgt_mask,
            )
        else:
            # Full mask comparison
            cost_mask, cost_dice = self._compute_mask_costs_full(
                out_mask,
                tgt_mask,
            )
        # Final cost matrix: weighted sum of all costs
        C = (
            self.cost_mask * cost_mask
            + self.cost_class * cost_class
            + self.cost_dice * cost_dice
        )  # [B*Q, total_targets]

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
        # Sample points
        point_coords = self._get_point_coords(tgt_mask)  # [1, num_points, 2]

        # Sample from predicted masks
        out_mask_sampled = self._point_sample(
            out_mask.unsqueeze(1),  # [B*Q, 1, H, W]
            point_coords.repeat(out_mask.shape[0], 1, 1),  # [B*Q, num_points, 2]
            align_corners=False,
        ).squeeze(1)  # [B*Q, num_points]

        # Sample from target masks
        tgt_mask_sampled = self._point_sample(
            tgt_mask.unsqueeze(1),  # [total_targets, 1, H, W]
            point_coords.repeat(
                tgt_mask.shape[0],
                1,
                1,
            ),  # [total_targets, num_points, 2]
            align_corners=False,
        ).squeeze(1)  # [total_targets, num_points]

        with torch.autocast(device_type="cuda", enabled=False):
            out_mask_sampled = out_mask_sampled.float()
            tgt_mask_sampled = tgt_mask_sampled.float()
            num_queries = out_mask_sampled.shape[0]
            num_targets = tgt_mask_sampled.shape[0]
            # Compute focal loss cost pairwise
            # Expand dims to broadcast: [B*Q, total_targets, num_points]
            out_mask_expanded = out_mask_sampled.unsqueeze(1).expand(
                -1,
                num_targets,
                -1,
            )  # [B*Q, total_targets, num_points]
            tgt_mask_expanded = tgt_mask_sampled.unsqueeze(0).expand(
                num_queries,
                -1,
                -1,
            )  # [B*Q, total_targets, num_points]
            # Compute focal loss
            cost_mask = sigmoid_focal_loss(
                out_mask_expanded,
                tgt_mask_expanded,
            ).mean(-1)  # Mean over points -> [B*Q, total_targets]
            # Compute dice loss
            out_mask_sampled_sigmoid = out_mask_sampled.sigmoid()
            numerator = 2 * (
                out_mask_sampled_sigmoid.unsqueeze(1) * tgt_mask_sampled.unsqueeze(0)
            ).sum(-1)  # [B*Q, total_targets]
            denominator = out_mask_sampled_sigmoid.unsqueeze(1).sum(
                -1,
            ) + tgt_mask_sampled.unsqueeze(0).sum(-1)  # [B*Q, total_targets]
            cost_dice = 1 - (numerator + 1) / (denominator + 1)  # [B*Q, total_targets]
        return cost_mask, cost_dice

    def _compute_mask_costs_full(
        self,
        out_mask: Tensor,
        tgt_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute mask costs using full masks.

        Args:
            out_mask: Predicted masks [B*Q, H, W]
            tgt_mask: Target masks [total_targets, H, W]

        Returns:
            Tuple of (focal_loss_cost, dice_loss_cost)

        """
        with torch.autocast(device_type="cuda", enabled=False):
            out_mask = out_mask.float()
            tgt_mask = tgt_mask.float()
            # Flatten spatial dimensions
            out_mask_flat = out_mask.flatten(1)  # [B*Q, H*W]
            tgt_mask_flat = tgt_mask.flatten(1)  # [total_targets, H*W]
            num_queries = out_mask_flat.shape[0]
            num_targets = tgt_mask_flat.shape[0]
            # Expand for pairwise computation: [B*Q, total_targets, H*W]
            out_mask_expanded = out_mask_flat.unsqueeze(1).expand(-1, num_targets, -1)
            tgt_mask_expanded = tgt_mask_flat.unsqueeze(0).expand(num_queries, -1, -1)
            # Compute focal loss pairwise: [B*Q, total_targets]
            cost_mask = sigmoid_focal_loss(
                out_mask_expanded,
                tgt_mask_expanded,
            ).mean(-1)  # Mean over spatial dimension
            # Compute dice loss pairwise: [B*Q, total_targets]
            out_mask_sigmoid = out_mask_flat.sigmoid()
            numerator = 2 * (
                out_mask_sigmoid.unsqueeze(1) * tgt_mask_flat.unsqueeze(0),
            ).sum(-1)
            denominator = out_mask_sigmoid.unsqueeze(1).sum(
                -1,
            ) + tgt_mask_flat.unsqueeze(0).sum(-1)
            cost_dice = 1 - (numerator + 1) / (denominator + 1)
        return cost_mask, cost_dice

    def _get_point_coords(self, masks: Tensor) -> Tensor:
        """
        Sample random point coordinates from masks.

        Args:
            masks: Tensor of shape [N, H, W]

        Returns:
            Point coordinates of shape [1, num_points, 2] in range [-1, 1]

        """
        _, h, w = masks.shape
        # Sample random coordinates
        # Use uniform sampling over the mask spatial dimensions
        y = torch.rand(1, self.num_points, device=masks.device)
        x = torch.rand(1, self.num_points, device=masks.device)
        # Convert to [-1, 1] range for grid_sample
        y = y * 2 - 1
        x = x * 2 - 1
        # Stack to [1, num_points, 2] format (x, y order for grid_sample)
        return torch.stack([x, y], dim=-1)

    def _point_sample(
        self,
        input_tensor: Tensor,
        point_coords: Tensor,
        *,
        align_corners: bool = False,
    ) -> Tensor:
        """
        Sample features at given point coordinates using bilinear interpolation.

        Args:
            input_tensor: Tensor of shape [N, C, H, W]
            point_coords: Tensor of shape [N, P, 2] with values in [-1, 1]
            align_corners: Whether to align corners in grid_sample

        Returns:
            Sampled features of shape [N, C, P]

        """
        # Add singleton dimension for grid_sample: [N, P, 2] -> [N, P, 1, 2]
        point_coords = point_coords.unsqueeze(2)

        # Sample using grid_sample
        output = fn.grid_sample(
            input_tensor,
            point_coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        )

        # Remove singleton dimension: [N, C, P, 1] -> [N, C, P]
        return output.squeeze(-1)

    def __repr__(self) -> str:
        """Representation of the Hungarian Matcher."""
        head = "Matcher " + self.__class__.__name__
        body = [
            f"cost_class: {self.cost_class}",
            f"cost_mask: {self.cost_mask}",
            f"cost_dice: {self.cost_dice}",
            f"num_points: {self.num_points}",
        ]
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)


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


# if __name__ == "__main__":
#     print("Testing HungarianMatcher...")

#     # Test with point sampling
#     print("\n1. Testing with point sampling (num_points=12544)...")
#     matcher = HungarianMatcher(
#         cost_class=2.0,
#         cost_mask=5.0,
#         cost_dice=5.0,
#         num_points=12544,
#     )
#     print(matcher)

#     # Create dummy outputs
#     batch_size, num_queries, num_classes = 2, 100, 5
#     outputs = {
#         "pred_logits": torch.randn(batch_size, num_queries, num_classes),
#         "pred_masks": torch.randn(batch_size, num_queries, 128, 128),
#     }

#     # Create dummy targets (semantic segmentation format)
#     targets = [
#         {
#             "labels": torch.tensor([0, 1, 2]),  # 3 classes present in image 1
#             "masks": torch.randint(0, 2, (3, 128, 128)).float(),
#         },
#         {
#             "labels": torch.tensor([0, 3]),  # 2 classes present in image 2
#             "masks": torch.randint(0, 2, (2, 128, 128)).float(),
#         },
#     ]

#     # Perform matching
#     indices = matcher(outputs, targets)

#     print(f"\n   Batch size: {batch_size}")
#     print(f"   Num queries: {num_queries}")
#     print(f"   Targets per batch: {[len(t['labels']) for t in targets]}")
#     print(f"\n   Matched indices for batch 0:")
#     print(f"      Prediction indices: {indices[0][0].tolist()}")
#     print(f"      Target indices: {indices[0][1].tolist()}")
#     print(f"   Matched indices for batch 1:")
#     print(f"      Prediction indices: {indices[1][0].tolist()}")
#     print(f"      Target indices: {indices[1][1].tolist()}")
#     # Test with full masks
#     print("\n2. Testing with full masks (num_points=None)...")
#     matcher_full = HungarianMatcher(
#         cost_class=2.0,
#         cost_mask=5.0,
#         cost_dice=5.0,
#         num_points=None,
#     )

#     # Use smaller masks for full comparison
#     outputs_small = {
#         "pred_logits": torch.randn(batch_size, num_queries, num_classes),
#         "pred_masks": torch.randn(batch_size, num_queries, 64, 64),
#     }
#     targets_small = [
#         {
#             "labels": torch.tensor([0, 1]),
#             "masks": torch.randint(0, 2, (2, 64, 64)).float(),
#         },
#         {
#             "labels": torch.tensor([2]),
#             "masks": torch.randint(0, 2, (1, 64, 64)).float(),
#         },
#     ]

#     indices_full = matcher_full(outputs_small, targets_small)
#     print(f"   Matched indices for batch 0: {indices_full[0][0].tolist()}")
#     print(f"   Matched indices for batch 1: {indices_full[1][0].tolist()}")

#     # Test builder function
#     print("\n3. Testing build_matcher convenience function...")
#     matcher_built = build_matcher(cost_class=1.0, cost_mask=3.0, cost_dice=3.0)
#     print(f"   Created matcher with custom costs")

#     print("\n✅ All tests passed!")
