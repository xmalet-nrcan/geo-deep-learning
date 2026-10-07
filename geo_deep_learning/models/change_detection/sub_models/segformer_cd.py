"""SegFormer-based bi-temporal change detection backbone.

Adapts the single-image SegFormer (``models/segmentation/segformer.py``,
Mix-Transformer encoder + all-MLP decoder) to the bi-temporal change
detection pipeline used by
:class:`~geo_deep_learning.tasks_with_models.change_detection_changeformer.ChangeDetectionChangeFormer`.

Architecture:
    1. Siamese Mix-Transformer (MiT-b0 … b5) encoder with shared weights,
       applied independently to the pre- and post-event images.
    2. Per-scale bi-temporal fusion of the 4 hierarchical features
       (strides 4, 8, 16, 32). Supported modes:
         - ``"concat_diff"`` (default): ``conv1x1([f1, f2, f1 - f2])``
         - ``"concat"``:               ``conv1x1([f1, f2])``
         - ``"diff"``:                 ``|f1 - f2|``
         - ``"signed_diff"``:          ``f1 - f2``
    3. SegFormer all-MLP decoder on the fused features.

Interface (identical to ChangeFormerV6/V7, HDANet, ChangeStar2, ...)::

    forward(x1, x2) -> list[Tensor]  # [p_c4, p_c3, p_c2, p_c1, final]

All 5 predictions are returned at the input resolution, so the deep
supervision, DFA and loss logic of the LightningModule work unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils import model_zoo

from geo_deep_learning.models.decoders.segformer_mlp import MLP
from geo_deep_learning.models.encoders.mix_transformer import mix_transformer_encoders
from geo_deep_learning.models.utils import patch_first_conv

logger = logging.getLogger(__name__)

RGB_CHANNELS = 3
FUSION_MODES = ("concat_diff", "concat", "diff", "signed_diff")


def build_mit_encoder(
    encoder: str,
    in_channels: int,
    encoder_weights: str | None = None,
) -> nn.Module:
    """Build a Mix-Transformer encoder for an arbitrary number of input channels.

    Unlike :func:`geo_deep_learning.models.encoders.mix_transformer.get_encoder`,
    pretrained (e.g. ``"imagenet"``) weights are also supported when
    ``in_channels != 3``: the RGB weights are loaded first, then the first
    patch-embedding convolution is expanded to ``in_channels`` (cyclic copy
    of the RGB kernels, rescaled by ``3 / in_channels``).

    The registry ``params`` dict is copied (not mutated in place).
    """
    try:
        cfg = mix_transformer_encoders[encoder]
    except KeyError as err:
        msg = (
            f"Unknown SegFormer encoder {encoder!r}. "
            f"Supported encoders: {sorted(mix_transformer_encoders)}"
        )
        raise ValueError(msg) from err

    encoder_cls = cfg["encoder"]
    load_pretrained = encoder_weights is not None
    build_channels = RGB_CHANNELS if load_pretrained else in_channels
    params = {**cfg["params"], "in_channels": build_channels, "depth": 5}
    model = encoder_cls(**params)

    if load_pretrained:
        try:
            settings = cfg["pretrained_settings"][encoder_weights]
        except KeyError as err:
            msg = (
                f"Unknown encoder_weights {encoder_weights!r} for {encoder!r}. "
                f"Available: {list(cfg['pretrained_settings'])}"
            )
            raise ValueError(msg) from err
        logger.info("Loading %s pretrained weights for %s", encoder_weights, encoder)
        model.load_state_dict(model_zoo.load_url(settings["url"], map_location="cpu"))
        if in_channels != RGB_CHANNELS:
            patch_first_conv(model=model, new_in_channels=in_channels, pretrained=True)

    return model


class BiTemporalFusion(nn.Module):
    """Fuse pre/post features of one scale into a single change feature map."""

    def __init__(self, channels: int, mode: str = "concat_diff") -> None:
        super().__init__()
        if mode not in FUSION_MODES:
            msg = f"Unknown fusion mode {mode!r}. Supported: {FUSION_MODES}"
            raise ValueError(msg)
        self.mode = mode
        self.proj: nn.Module | None = None
        n_inputs = {"concat_diff": 3, "concat": 2}.get(mode)
        if n_inputs is not None:
            self.proj = nn.Sequential(
                nn.Conv2d(channels * n_inputs, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, f1: Tensor, f2: Tensor) -> Tensor:
        """Fuse ``f1`` (pre) and ``f2`` (post), both ``[B, C, H, W]``."""
        if self.mode == "diff":
            return torch.abs(f1 - f2)
        if self.mode == "signed_diff":
            return f1 - f2
        if self.mode == "concat":
            return self.proj(torch.cat((f1, f2), dim=1))
        return self.proj(torch.cat((f1, f2, f1 - f2), dim=1))


class SegFormerChangeDecoder(nn.Module):
    """SegFormer all-MLP decoder with one auxiliary head per scale.

    Same structure as :class:`geo_deep_learning.models.decoders.segformer_mlp.Decoder`
    but always returns 4 auxiliary predictions (c4, c3, c2, c1) + the final
    one, ordered like the ChangeFormer decoder so the deep supervision
    weights and DFA gates line up.
    """

    def __init__(
        self,
        in_channels: list[int],
        embedding_dim: int = 256,
        num_classes: int = 2,
        dropout_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.linear_c4 = MLP(input_dim=c4, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1, embed_dim=embedding_dim)

        self.aux_heads = nn.ModuleDict({
            key: nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
            for key in ("c4", "c3", "c2", "c1")
        })

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    @staticmethod
    def _project(mlp: MLP, feat: Tensor, size: torch.Size) -> Tensor:
        """Linear-embed ``feat`` and resize it to ``size`` (the c1 resolution)."""
        n, _, h, w = feat.shape
        out = mlp(feat).permute(0, 2, 1).reshape(n, -1, h, w).contiguous()
        if out.shape[2:] != size:
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        return out

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        """Return ``[p_c4, p_c3, p_c2, p_c1, final]`` at stride 4."""
        c1, c2, c3, c4 = feats
        size = c1.shape[2:]
        _c4 = self._project(self.linear_c4, c4, size)
        _c3 = self._project(self.linear_c3, c3, size)
        _c2 = self._project(self.linear_c2, c2, size)
        _c1 = self._project(self.linear_c1, c1, size)

        aux = [
            self.aux_heads["c4"](_c4),
            self.aux_heads["c3"](_c3),
            self.aux_heads["c2"](_c2),
            self.aux_heads["c1"](_c1),
        ]
        fused = self.linear_fuse(torch.cat((_c4, _c3, _c2, _c1), dim=1))
        final = self.linear_pred(self.dropout(fused))
        return [*aux, final]


class SegFormerChangeDetection(nn.Module):
    """Siamese SegFormer for bi-temporal change detection.

    Args:
        input_nc: Number of input channels per date (after optional
            signed-difference concat done by ``ChangeDetectionModel``).
        output_nc: Number of output classes.
        encoder: Mix-Transformer variant (``"mit_b0"`` … ``"mit_b5"``).
        encoder_weights: Pretrained weights key (``"imagenet"``) or ``None``.
        embed_dim: Decoder embedding dimension (SegFormer uses 256 for
            b0/b1 and 768 for b2+).
        fusion: Bi-temporal fusion mode, one of :data:`FUSION_MODES`.
        decoder_dropout: Dropout ratio before the final classifier.
        freeze_encoder: If True, freeze all encoder parameters.
        decoder_softmax: If True, apply softmax to every output.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_nc: int = 3,
        output_nc: int = 2,
        encoder: str = "mit_b2",
        encoder_weights: str | None = None,
        embed_dim: int = 256,
        fusion: str = "concat_diff",
        decoder_dropout: float = 0.1,
        freeze_encoder: bool = False,
        decoder_softmax: bool = False,
        **kwargs: Any,  # noqa: ARG002 — accept extra kwargs for compatibility
    ) -> None:
        super().__init__()
        self.encoder_name = encoder
        self.encoder = build_mit_encoder(encoder, input_nc, encoder_weights)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        feat_channels = list(mix_transformer_encoders[encoder]["params"]["embed_dims"])
        self.fusions = nn.ModuleList(
            [BiTemporalFusion(ch, mode=fusion) for ch in feat_channels],
        )
        self.decoder = SegFormerChangeDecoder(
            in_channels=feat_channels,
            embedding_dim=embed_dim,
            num_classes=output_nc,
            dropout_ratio=decoder_dropout,
        )
        self.apply_softmax = decoder_softmax

    def forward(self, x1: Tensor, x2: Tensor) -> list[Tensor]:
        """Forward pass.

        Args:
            x1: Pre-event image  ``[B, C, H, W]``.
            x2: Post-event image ``[B, C, H, W]``.

        Returns:
            ``[p_c4, p_c3, p_c2, p_c1, final]``, each ``[B, output_nc, H, W]``.
        """
        target_size = x1.shape[2:]
        feats1 = self.encoder(x1)
        feats2 = self.encoder(x2)
        fused = [fuse(f1, f2) for fuse, f1, f2 in zip(self.fusions, feats1, feats2, strict=True)]

        outputs = [
            F.interpolate(o, size=target_size, mode="bilinear", align_corners=False)
            for o in self.decoder(fused)
        ]
        if self.apply_softmax:
            outputs = [torch.softmax(o, dim=1) for o in outputs]
        return outputs


def _variant(encoder: str) -> type[SegFormerChangeDetection]:
    """Create a ``SegFormerChangeDetection`` subclass bound to one MiT encoder."""

    class _SegFormerVariant(SegFormerChangeDetection):
        def __init__(self, input_nc: int = 3, output_nc: int = 2, **kwargs: Any) -> None:
            kwargs.setdefault("encoder", encoder)
            super().__init__(input_nc=input_nc, output_nc=output_nc, **kwargs)

    suffix = encoder.removeprefix("mit_").upper()
    _SegFormerVariant.__name__ = _SegFormerVariant.__qualname__ = f"SegFormerChangeDetection{suffix}"
    _SegFormerVariant.__doc__ = f"SegFormer change detection with a {encoder} encoder."
    return _SegFormerVariant


SegFormerChangeDetectionB0 = _variant("mit_b0")
SegFormerChangeDetectionB1 = _variant("mit_b1")
SegFormerChangeDetectionB2 = _variant("mit_b2")
SegFormerChangeDetectionB3 = _variant("mit_b3")
SegFormerChangeDetectionB4 = _variant("mit_b4")
SegFormerChangeDetectionB5 = _variant("mit_b5")


if __name__ == "__main__":
    model = SegFormerChangeDetectionB0(input_nc=10, output_nc=2)
    a = torch.randn(2, 10, 256, 256)
    b = torch.randn(2, 10, 256, 256)
    outs = model(a, b)
    print(len(outs), [tuple(o.shape) for o in outs])  # noqa: T201
