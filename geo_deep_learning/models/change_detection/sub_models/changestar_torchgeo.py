"""ChangeStar (FarSeg) from torchgeo, adapted to the geo-deep-learning pipeline.

Reference:
    Zheng et al., "Change is Everywhere: Single-Temporal Supervised Object
    Change Detection in Remote Sensing Imagery", ICCV 2021.
    https://arxiv.org/abs/2108.07002

Implementation:
    ``torchgeo.models.ChangeStarFarSeg`` (FarSeg dense feature extractor +
    ``ChangeMixin``). We reuse torchgeo's sub-modules as-is and only adapt the
    I/O contract so the model is a drop-in replacement for the other backbones
    of :data:`~geo_deep_learning.models.change_detection.change_detection_model._MODEL_REGISTRY`.

Differences with calling ``torchgeo.models.ChangeStarFarSeg`` directly:

1. **Input**: ``forward(x1, x2)`` with ``[B, C, H, W]`` tensors instead of a
   single ``[B, T=2, C, H, W]`` tensor.
2. **Arbitrary input channels**: torchvision/torchgeo ResNets are built for a
   fixed number of channels (3 for ImageNet, 2 for Sentinel-1 weights...).
   ``conv1`` is rebuilt for ``input_nc`` channels; when pretrained weights
   are loaded, their mean over the input dimension is replicated (and
   rescaled) so the pretrained filters are kept rather than discarded.
3. **Output = logits, always**: torchgeo returns ``change_prob`` (sigmoid
   already applied) in eval mode and a dict in train mode. Here every mode
   returns a ``list[Tensor]`` of **logits** ``[B, output_nc, H, W]``, so
   ``outputs[-1]`` is the final prediction, as for ChangeFormer/ChangeStar2.
4. **1 logit → 2 classes**: ``ChangeMixin`` predicts a single change logit
   ``z``. With ``output_nc=2`` we return ``[-z/2, z/2]``, whose softmax is
   exactly ``[1 - sigmoid(z), sigmoid(z)]``: the softmax/one-hot loss and
   ``argmax`` metrics of the task stay mathematically equivalent to the
   original sigmoid formulation.
5. **No unused layers**: ChangeStar's segmentation head (only needed for
   single-temporal supervision) and the ResNet ImageNet classifier (``fc``,
   never called by FarSeg) are dropped, so every parameter receives a
   gradient (no ``find_unused_parameters`` needed with DDP).
6. **Any H, W**: inputs are zero-padded to a multiple of 32 (ResNet stride)
   and outputs cropped back, so ``ChangeMixin``'s fixed x4 upsampling always
   lands exactly on ``H x W``.

Full-resolution refinement (optional, ``full_res_refine=True``):
    FarSeg and ``ChangeMixin`` work at H/4 (400 m blocks at 100 m/pixel) and
    the prediction is bilinearly upsampled x4, so no detail finer than 4
    pixels can reach the output. :class:`FullResRefineHead` adds a U-Net-like
    skip: ``ChangeMixin``'s inner features (context) are upsampled to H x W
    and fused with a shallow branch computed on the full-resolution images
    ``(x_a, x_b, x_a - x_b)`` (precise edges). It predicts a *residual* added
    to the upsampled coarse logit; its last conv is zero-initialised, so the
    model starts exactly as plain ChangeStar and checkpoints without the head
    still load (``weights_strict: false``).

Temporal direction:
    ``ChangeMixin`` predicts ``c12`` from ``cat(t1, t2)`` and ``c21`` from
    ``cat(t2, t1)`` with shared weights. Training both against the same label
    (``bidirectional_training=True``, as in the paper) makes the prediction
    order-invariant, i.e. it can no longer tell a backscatter *decrease* from
    an *increase*. For burned-area mapping the direction is informative, hence
    ``bidirectional_training=False`` by default (only the ``inference_mode``
    direction is trained and predicted).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)

# Output stride of the ResNet backbone used by FarSeg.
_RESNET_STRIDE = 32

# Clamp used when converting the "mean" probability back to a logit.
_PROB_EPS = 1e-6

InferenceMode = Literal["t1t2", "t2t1", "mean"]
_INFERENCE_MODES: tuple[str, ...] = ("t1t2", "t2t1", "mean")


def _resolve_backbone_weights(backbone_weights: Any) -> Any:
    """Turn a YAML-friendly weights spec into a torchvision ``WeightsEnum``.

    Accepts ``None``, an already-built ``WeightsEnum``, or a string such as
    ``"ResNet50_Weights.SENTINEL1_ALL_MOCO"`` (torchgeo) or
    ``"ResNet50_Weights.IMAGENET1K_V2"`` (torchvision).
    """
    if backbone_weights is None or not isinstance(backbone_weights, str):
        return backbone_weights
    if backbone_weights.lower() in {"", "none", "null"}:
        return None

    errors: list[str] = []
    try:
        from torchgeo.models import get_weight as torchgeo_get_weight

        return torchgeo_get_weight(backbone_weights)
    except Exception as exc:  # noqa: BLE001 - fall back to torchvision
        errors.append(f"torchgeo: {exc}")
    try:
        from torchvision.models import get_weight as torchvision_get_weight

        return torchvision_get_weight(backbone_weights)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torchvision: {exc}")
    msg = f"Unknown backbone_weights={backbone_weights!r} ({'; '.join(errors)})"
    raise ValueError(msg)


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int, *, pretrained: bool) -> nn.Conv2d:
    """Rebuild ``conv`` for ``in_channels`` inputs, keeping pretrained filters.

    The pretrained kernel is averaged over its input channels and replicated,
    scaled by ``old_in / new_in`` so the activation magnitude is preserved
    (same strategy as timm's ``adapt_input_conv``).
    """
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    if pretrained:
        with torch.no_grad():
            mean_kernel = conv.weight.mean(dim=1, keepdim=True)  # [O, 1, k, k]
            scale = conv.in_channels / in_channels
            new_conv.weight.copy_(mean_kernel.repeat(1, in_channels, 1, 1) * scale)
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
    return new_conv


class FullResRefineHead(nn.Module):
    """Residual full-resolution refinement of ChangeStar's coarse change logit.

    Args:
        image_channels: Channels of each input image.
        context_channels: Channels of the low-resolution ``ChangeMixin`` features.
        channels: Hidden width of the head.
    """

    def __init__(self, image_channels: int, context_channels: int, channels: int = 16) -> None:
        super().__init__()
        self.high_res = nn.Sequential(
            nn.Conv2d(3 * image_channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + context_channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Conv2d(channels, 1, 1)
        # Zero init: the head starts as an identity on the coarse prediction.
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x_a: Tensor, x_b: Tensor, context: Tensor, coarse_logit: Tensor) -> Tensor:
        """Return the refined ``[B, 1, H, W]`` logit for the direction ``a -> b``."""
        size = x_a.shape[-2:]
        high_res = self.high_res(torch.cat([x_a, x_b, x_a - x_b], dim=1))
        context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)
        delta = self.residual(self.fuse(torch.cat([high_res, context], dim=1)))
        return coarse_logit + delta


class ChangeStarTorchGeo(nn.Module):
    """``torchgeo.models.ChangeStarFarSeg`` with the pipeline's I/O contract.

    Args:
        input_nc: Number of input channels per image (after the optional
            signed-difference concatenation done by ``ChangeDetectionModel``).
        output_nc: ``2`` (softmax no-change/change logits, the pipeline
            default) or ``1`` (raw change logit).
        backbone: FarSeg ResNet backbone: ``resnet18`` | ``resnet34`` |
            ``resnet50`` | ``resnet101``.
        backbone_weights: Pretrained weights, e.g.
            ``"ResNet50_Weights.SENTINEL1_ALL_MOCO"`` (SAR, torchgeo) or
            ``"ResNet50_Weights.IMAGENET1K_V2"``. ``None`` = random init.
            Requires torchgeo >= 0.9.
        inference_mode: Direction used for prediction: ``t1t2`` (pre→post,
            default), ``t2t1``, or ``mean`` (mean of both probabilities, as in
            torchgeo). ``mean`` should be paired with
            ``bidirectional_training=True``.
        bidirectional_training: In training mode, return both directions
            ``[c21, c12]`` (or ``[c12, c21]`` for ``t2t1``) so the task's
            deep-supervision loss trains both. Requires ``deep_supervision:
            true`` in the task, ideally with ``deep_supervision_weights:
            [1.0, 1.0]``.
        changemixin_inner_channels: ``ChangeMixin`` hidden width (torchgeo: 16).
        changemixin_num_convs: ``ChangeMixin`` conv blocks (torchgeo: 4).
        full_res_refine: Add :class:`FullResRefineHead` (full-resolution
            residual refinement of the change logit).
        refine_channels: Hidden width of the refinement head.
        decoder_softmax: Apply softmax to the outputs (keep ``False``: the
            task's losses expect logits).
        **kwargs: Ignored (e.g. the registry default ``embed_dim``).
    """

    def __init__(  # noqa: PLR0913
        self,
        input_nc: int = 3,
        output_nc: int = 2,
        backbone: str = "resnet50",
        backbone_weights: Any = None,
        inference_mode: InferenceMode = "t1t2",
        bidirectional_training: bool = False,
        changemixin_inner_channels: int = 16,
        changemixin_num_convs: int = 4,
        full_res_refine: bool = False,
        refine_channels: int = 16,
        decoder_softmax: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        super().__init__()
        try:
            from torchgeo.models import ChangeMixin, ChangeStarFarSeg
        except ImportError as exc:  # pragma: no cover - depends on the env
            msg = "change_detection_model='changestar_farseg' requires torchgeo (pip install torchgeo)"
            raise ImportError(msg) from exc

        if output_nc not in (1, 2):
            msg = (
                f"ChangeStar predicts binary change: output_nc must be 1 or 2, got {output_nc}. "
                "Use num_classes: 1 (or 2) in the task."
            )
            raise ValueError(msg)
        if inference_mode not in _INFERENCE_MODES:
            msg = f"inference_mode must be one of {_INFERENCE_MODES}, got {inference_mode!r}"
            raise ValueError(msg)
        if inference_mode == "mean" and not bidirectional_training:
            logger.warning(
                "ChangeStar: inference_mode='mean' with bidirectional_training=False — "
                "the t2t1 direction is never trained, its predictions are meaningless.",
            )

        self.output_nc = output_nc
        self.inference_mode = inference_mode
        self.bidirectional_training = bidirectional_training
        self.apply_softmax = decoder_softmax

        weights = _resolve_backbone_weights(backbone_weights)
        farseg_kwargs: dict[str, Any] = {"backbone": backbone, "classes": 1}
        if weights is not None:
            farseg_kwargs["backbone_weights"] = weights  # torchgeo >= 0.9
        reference = ChangeStarFarSeg(**farseg_kwargs)

        # FarSeg with its classifier replaced by Identity -> [N, 128, H/4, W/4].
        self.dense_feature_extractor = reference.dense_feature_extractor
        resnet = self.dense_feature_extractor.backbone
        resnet.conv1 = _adapt_first_conv(resnet.conv1, input_nc, pretrained=weights is not None)
        # FarSeg never calls the ImageNet classifier (fc); drop it so that every
        # parameter receives a gradient (also true in torchgeo's own model).
        resnet.fc = nn.Identity()

        # Same layout as torchgeo's ChangeStarFarSeg, configurable width/depth.
        self.changemixin = ChangeMixin(
            in_channels=128 * 2,
            inner_channels=changemixin_inner_channels,
            num_convs=changemixin_num_convs,
            scale_factor=4.0,
        )
        # reference.seg_classifier is intentionally dropped (see module docstring).

        self.refine_head: FullResRefineHead | None = (
            FullResRefineHead(input_nc, changemixin_inner_channels, refine_channels)
            if full_res_refine else None
        )

        logger.info(
            "ChangeStarFarSeg(torchgeo): backbone=%s, weights=%s, input_nc=%d, output_nc=%d, "
            "inference_mode=%s, bidirectional_training=%s, full_res_refine=%s",
            backbone, weights, input_nc, output_nc, inference_mode, bidirectional_training,
            full_res_refine,
        )

    def _to_output(self, change_logit: Tensor) -> Tensor:
        """``[B, 1, H, W]`` change logit -> ``[B, output_nc, H, W]`` logits."""
        if self.output_nc == 1:
            return change_logit
        half = change_logit / 2
        return torch.cat([-half, half], dim=1)

    def _mixin_with_refinement(self, bi_feature: Tensor, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        """Run ``ChangeMixin`` exposing its inner features, then refine both directions.

        Same computation as ``torchgeo.models.ChangeMixin.forward`` (and same
        weights layout), split before the classification conv to reuse the
        inner features as context for the refinement head.
        """
        batch_size = bi_feature.size(0)
        t1t2 = torch.cat([bi_feature[:, 0], bi_feature[:, 1]], dim=1)
        t2t1 = torch.cat([bi_feature[:, 1], bi_feature[:, 0]], dim=1)
        convs = self.changemixin.convs  # [conv blocks..., cls conv, upsample]
        inner = convs[:-2](torch.cat([t1t2, t2t1], dim=0))
        coarse = convs[-1](convs[-2](inner))
        inner12, inner21 = torch.split(inner, batch_size, dim=0)
        coarse12, coarse21 = torch.split(coarse, batch_size, dim=0)
        # Only refine the direction(s) actually returned by ``forward``
        # (the full-resolution branch is the costly part of the head).
        both = (self.training and self.bidirectional_training) or (
            self.inference_mode == "mean" and not self.training
        )
        need12 = both or self.inference_mode != "t2t1"
        need21 = both or self.inference_mode == "t2t1"
        c12 = self.refine_head(x1, x2, inner12, coarse12) if need12 else coarse12
        c21 = self.refine_head(x2, x1, inner21, coarse21) if need21 else coarse21
        return c12, c21

    def forward(self, x1: Tensor, x2: Tensor) -> list[Tensor]:
        """Forward pass.

        Args:
            x1: Pre-event image ``[B, C, H, W]``.
            x2: Post-event image ``[B, C, H, W]``.

        Returns:
            ``list[Tensor]`` of logits ``[B, output_nc, H, W]``; the last item is
            the prediction to use. Two items (other direction first) only in
            training mode with ``bidirectional_training=True``.
        """
        batch_size, _, height, width = x1.shape

        pad_h = (-height) % _RESNET_STRIDE
        pad_w = (-width) % _RESNET_STRIDE
        if pad_h or pad_w:
            x1 = F.pad(x1, (0, pad_w, 0, pad_h))
            x2 = F.pad(x2, (0, pad_w, 0, pad_h))

        # Siamese feature extraction, both dates in one pass (shared weights).
        features = self.dense_feature_extractor(torch.cat([x1, x2], dim=0))
        bi_feature = torch.stack([features[:batch_size], features[batch_size:]], dim=1)
        if self.refine_head is None:
            c12, c21 = self.changemixin(bi_feature)  # [B, 1, H', W'] each
        else:
            c12, c21 = self._mixin_with_refinement(bi_feature, x1, x2)
        c12 = c12[..., :height, :width]
        c21 = c21[..., :height, :width]

        if self.training and self.bidirectional_training:
            ordered = [c12, c21] if self.inference_mode == "t2t1" else [c21, c12]
        elif self.inference_mode == "t2t1":
            ordered = [c21]
        elif self.inference_mode == "mean" and not self.training:
            prob = (torch.sigmoid(c12) + torch.sigmoid(c21)) / 2
            ordered = [torch.logit(prob, eps=_PROB_EPS)]
        else:
            ordered = [c12]

        outputs = [self._to_output(o) for o in ordered]
        if self.apply_softmax:
            outputs = [torch.softmax(o, dim=1) if self.output_nc > 1 else torch.sigmoid(o) for o in outputs]
        return outputs
