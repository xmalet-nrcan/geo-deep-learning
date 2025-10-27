"""Mask2Former loss utilities."""

from collections.abc import Callable

import torch
import torch.nn.functional as fn
from torch import Tensor


def sigmoid_ce_loss(
    inputs: Tensor,
    targets: Tensor,
    num_masks: float,
) -> Tensor:
    """Binary cross entropy loss."""
    loss = fn.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)


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
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss_with_num_masks)


def batch_dice_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute the DICE loss, similar to generalized IOU for masks."""
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)


def batch_sigmoid_ce_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute BCE loss in batched pairwise fashion."""
    hw = inputs.shape[1]
    pos = fn.binary_cross_entropy_with_logits(
        inputs,
        torch.ones_like(inputs),
        reduction="none",
    )
    neg = fn.binary_cross_entropy_with_logits(
        inputs,
        torch.zeros_like(inputs),
        reduction="none",
    )
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm",
        neg,
        (1 - targets),
    )
    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)


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


# Adapted from detectron2/point_features.py
def point_sample(input_tensor: Tensor, point_coords: Tensor, **kwargs: any) -> Tensor:
    """
    Sample points from input feature map.

    Args:
        input_tensor (Tensor): Shape (N, C, H, W)
        point_coords (Tensor): Shape (N, P, 2) with values in [0, 1] x [0, 1]
        kwargs: Additional arguments for grid_sample

    Returns:
        output (Tensor): Shape (N, C, P)

    """
    add_dim = False
    points_dim = 3
    if point_coords.dim() == points_dim:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = fn.grid_sample(input_tensor, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


# Adapted from mask2former/point_features.py
def calculate_uncertainty(logits: Tensor) -> Tensor:
    """Calculate uncertainty from logits."""
    if logits.shape[1] != 1:
        msg = "logits must have 1 channel"
        raise ValueError(msg)
    return -(torch.abs(logits.clone()))


# Adapted from detectron2/point_features.py
def get_uncertain_point_coords_with_randomness(
    coarse_logits: Tensor,
    uncertainty_func: Callable,
    num_points: int,
    oversample_ratio: int,
    importance_sample_ratio: float,
) -> Tensor:
    """
    Sample points based on uncertainty with oversampling.

    Args:
        coarse_logits (Tensor): Shape (N, C, H, W) or (N, 1, H, W)
        uncertainty_func: Function that computes uncertainty from logits
        num_points (int): Number of points to sample
        oversample_ratio (int): Oversample by this ratio then select top-k
        importance_sample_ratio (float): Ratio of importance vs random sampling

    Returns:
        point_coords (Tensor): Shape (N, P, 2) in [0, 1] x [0, 1] space

    """
    if oversample_ratio < 1:
        msg = "oversample_ratio must be >= 1"
        raise ValueError(msg)
    if importance_sample_ratio > 1 or importance_sample_ratio < 0:
        msg = "importance_sample_ratio must be between 0 and 1"
        raise ValueError(msg)

    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)

    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)

    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)

    point_uncertainties = uncertainty_func(point_logits)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        num_boxes,
        dtype=torch.long,
        device=coarse_logits.device,
    )
    idx += shift[:, None]

    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes,
        num_uncertain_points,
        2,
    )

    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(
                    num_boxes,
                    num_random_points,
                    2,
                    device=coarse_logits.device,
                ),
            ],
            dim=1,
        )

    return point_coords
