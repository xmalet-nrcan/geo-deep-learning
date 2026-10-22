"""Loss-side helpers for the change-detection task.

Pure functions (no ``LightningModule`` dependency): target preparation,
logit sanitising, degenerate-sample patching and the two auxiliary losses
(burned false-negative penalty, boundary-band BCE) added on top of the
focal/Lovász main loss.
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Label value skipped by losses and metrics (``ignore_index``).
IGNORE_MASK_INDEX = 255

# Threshold above which a ``mask-common`` / validity-mask pixel is considered
# valid (values are floats after augmentation/interpolation, not strict 0/1).
MASK_VALID_THRESHOLD = 0.5

# Replacement value for NaN logits before they reach the loss functions.
# Using 0.0 (a neutral / low-confidence score) avoids injecting an
# artificially "certain" class prediction into the loss — see docstring on
# ``sanitize_logits`` for details on why an extreme value here is unsafe.
NAN_LOGIT_REPLACEMENT = 0.0

# Samples with fewer than this fraction of valid pixels are patched with
# low-amplitude noise and excluded from the loss (see
# ``patch_degenerate_samples``) to avoid near-zero LayerNorm variance.
MIN_VALID_RATIO_FOR_PATCH = 0.05


def sanitize_logits(logits: Tensor) -> Tensor:
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


def patch_degenerate_samples(
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
    logger.debug(
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


def prepare_one_hot_targets(
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


def burned_false_negative_penalty(
        logits: Tensor,
        one_hot: Tensor,
        valid_mask: Tensor,
        burned_class_weight: float,
) -> Tensor:
    """Extra BCE penalty on positive (burned) pixels to reduce false negatives."""
    if burned_class_weight <= 1.0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)

    if logits.shape[1] > 1:
        burned_logits = logits[:, 1:2, :, :]
        burned_targets = one_hot[:, 1:2, :, :]
    else:
        burned_logits = logits
        burned_targets = one_hot

    if valid_mask.dim() == 3:  # noqa: PLR2004
        valid_mask = valid_mask.unsqueeze(1)
    valid_mask = valid_mask.to(dtype=logits.dtype)

    pos_weight = torch.tensor(
        burned_class_weight,
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


def boundary_band(target: Tensor, valid_mask: Tensor, width: int) -> Tensor:
    """Band of ``±width`` pixels around the edges of a binary ``[B, 1, H, W]`` mask.

    Computed as morphological dilation minus erosion (max-pooling), then
    restricted to valid pixels so nodata borders never create fake edges
    inside the band.
    """
    kernel = 2 * width + 1
    dilated = F.max_pool2d(target, kernel, stride=1, padding=width)
    eroded = -F.max_pool2d(-target, kernel, stride=1, padding=width)
    return ((dilated - eroded) > 0).to(target.dtype) * valid_mask


def boundary_bce_loss(
        logits: Tensor,
        one_hot: Tensor,
        valid_mask: Tensor,
        width: int,
) -> Tensor:
    """BCE restricted to a band around the ground-truth change boundaries.

    Uses the change-vs-no-change logit ``logsumexp(l_1..l_C) - l_0``, whose
    sigmoid equals the softmax probability of "change" for any number of
    classes (``l_1 - l_0`` in the binary case), so the loss is consistent
    with the ``argmax`` predictions whatever the backbone's logit scale.
    Pixels near both sides of each edge are roughly balanced by
    construction, so no pos_weight is applied. Returns 0 when the batch
    contains no boundary (e.g. no burned pixel).
    """
    if valid_mask.dim() == 3:  # noqa: PLR2004
        valid_mask = valid_mask.unsqueeze(1)
    valid = (valid_mask > MASK_VALID_THRESHOLD).to(logits.dtype)

    if logits.shape[1] > 1:
        change_logit = torch.logsumexp(logits[:, 1:], dim=1, keepdim=True) - logits[:, :1]
        change_target = 1.0 - one_hot[:, :1]
    else:
        change_logit = logits
        change_target = one_hot[:, :1]
    change_target = change_target.to(logits.dtype)

    band = boundary_band(change_target * valid, valid, width)
    n_band = band.sum()
    if n_band == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(change_logit, change_target, reduction="none")
    return (bce * band).sum() / n_band
