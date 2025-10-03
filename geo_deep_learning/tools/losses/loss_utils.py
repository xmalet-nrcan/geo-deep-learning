"""Mask2Former loss utilities."""

import torch
import torch.nn.functional as fn
from torch import Tensor


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """
    Sigmoid focal loss for Mask2Former (per-pixel, no reduction).

    This is the original Mask2Former implementation which differs slightly
    from the smp FocalLoss in reduction behavior for matching cost computation

    Args:
        inputs: Predictions (logits) of shape [N, H, W] or [N, H*W]
        targets: Ground truth labels of shape [N, H, W] or [N, H*W] (0 or 1)
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Exponent of the modulating factor (1 - p_t)^gamma

    Returns:
        Focal loss value per pixel (no reduction)

    """
    prob = inputs.sigmoid()
    ce_loss = fn.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss


def dice_loss_for_matching(
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Compute the DICE loss for matching cost computation.

    This version is specifically for computing matching costs in the
    Hungarian matcher, with a different signature than smp's DiceLoss.

    Args:
        inputs: Predictions (logits) of shape [N, H, W] or [N, H*W]
        targets: Ground truth of shape [N, H, W] or [N, H*W] (0 or 1)

    Returns:
        Dice loss per prediction [N]

    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return 1 - (numerator + 1) / (denominator + 1)


def dice_loss_with_num_masks(
    inputs: Tensor,
    targets: Tensor,
    num_masks: float,
) -> Tensor:
    """
    Dice loss normalized by number of masks (for criterion loss computation).

    Args:
        inputs: Predictions (logits) of shape [N, H, W]
        targets: Ground truth of shape [N, H, W] (0 or 1)
        num_masks: Number of masks for normalization

    Returns:
        Normalized dice loss (scalar)

    """
    loss = dice_loss_for_matching(inputs, targets)
    return loss.sum() / num_masks


def batch_sigmoid_ce_loss_mean(
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Binary cross entropy loss with sigmoid, mean reduction.

    Args:
        inputs: Predictions (logits) of shape [B, Q, H, W]
        targets: Ground truth of shape [B, Q, H, W] (0 or 1)

    Returns:
        Mean BCE loss (scalar)

    """
    loss = fn.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean()


def get_src_permutation_idx(
    indices: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """
    Get source permutation indices from matching.

    Used to permute predictions following the Hungarian matching indices.

    Args:
        indices: List of (src_idx, tgt_idx) tuples from matcher, one per batch element

    Returns:
        Tuple of (batch_idx, src_idx) tensors for advanced indexing

    Example:
        If indices = [(tensor([2, 5]), tensor([0, 1])),
                      (tensor([1, 3]), tensor([0, 1]))],
        Returns: (tensor([0, 0, 1, 1]), tensor([2, 5, 1, 3]))

    """
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)],
    )
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def get_tgt_permutation_idx(
    indices: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """
    Get target permutation indices from matching.

    Used to permute targets following the Hungarian matching indices.

    Args:
        indices: List of (src_idx, tgt_idx) tuples from matcher, one per batch element

    Returns:
        Tuple of (batch_idx, tgt_idx) tensors for advanced indexing

    Example:
        If indices = [(tensor([2, 5]), tensor([0, 1])),
                      (tensor([1, 3]), tensor([0, 1]))],
        Returns: (tensor([0, 0, 1, 1]), tensor([0, 1, 0, 1]))

    """
    batch_idx = torch.cat(
        [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)],
    )
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


# if __name__ == "__main__":
#     print("Testing Mask2Former-specific loss utilities...")

#     # Test sigmoid focal loss
#     print("\n1. Testing sigmoid_focal_loss...")
#     inputs = torch.randn(10, 100, 100)  # [N, H, W]
#     targets = torch.randint(0, 2, (10, 100, 100)).float()
#     loss = sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0)
#     print(f"   Shape: {loss.shape}, Mean: {loss.mean().item():.4f}")

#     # Test dice loss for matching
#     print("\n2. Testing dice_loss_for_matching...")
#     inputs = torch.randn(10, 64, 64)
#     targets = torch.randint(0, 2, (10, 64, 64)).float()
#     loss = dice_loss_for_matching(inputs, targets)
#     print(f"   Shape: {loss.shape}, Values: {loss[:3].tolist()}")

#     # Test dice loss with num_masks
#     print("\n3. Testing dice_loss_with_num_masks...")
#     loss = dice_loss_with_num_masks(inputs, targets, num_masks=10.0)
#     print(f"   Normalized loss: {loss.item():.4f}")

#     # Test permutation indices
#     print("\n4. Testing get_src_permutation_idx...")
#     indices = [
#         (torch.tensor([2, 5, 7]), torch.tensor([0, 1, 2])),
#         (torch.tensor([1, 3]), torch.tensor([0, 1])),
#     ]
#     batch_idx, src_idx = get_src_permutation_idx(indices)
#     print(f"   Batch idx: {batch_idx.tolist()}")
#     print(f"   Source idx: {src_idx.tolist()}")

#     print("\n5. Testing get_tgt_permutation_idx...")
#     batch_idx, tgt_idx = get_tgt_permutation_idx(indices)
#     print(f"   Batch idx: {batch_idx.tolist()}")
#     print(f"   Target idx: {tgt_idx.tolist()}")

#     print("\n✅ All tests passed!")
