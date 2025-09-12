"""Mask2Former utils."""

import copy
import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as fn


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:  # noqa: N803
    """Get clones of a module."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return fn.relu
    if activation == "gelu":
        return fn.gelu
    if activation == "glu":
        return fn.glu
    err_msg = f"activation should be relu/gelu, not {activation}."
    raise RuntimeError(err_msg)


def c2_xavier_fill(module: nn.Module) -> None:
    """XavierFill in Caffe2."""
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def get_norm(norm: str | None, out_channels: int) -> nn.Module | None:
    """Get normalization layer."""
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            # "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": nn.SyncBatchNorm,
            # "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            # "naiveSyncBN": NaiveSyncBatchNorm,
            # expose stats_mode N as an option to caller, required for zero-len inputs
            # "naiveSyncBN_N": lambda channels: NaiveSyncBatchNorm(
            #     channels, stats_mode="N"
            # ),
            # "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


class SelfAttentionLayer(nn.Module):
    """Self-attention layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,  # noqa: FBT001,FBT002
    ) -> None:
        """Initialize SelfAttentionLayer."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Reset parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(
        self,
        tensor: torch.Tensor,
        pos: torch.Tensor | None,
    ) -> torch.Tensor:
        """Add positional embedding to tensor."""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)

    def forward_pre(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        return tgt + self.dropout(tgt2)

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,  # noqa: FBT001,FBT002
    ) -> None:
        """Initialize CrossAttentionLayer."""
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Reset parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(
        self,
        tensor: torch.Tensor,
        pos: torch.Tensor | None,
    ) -> torch.Tensor:
        """Add positional embedding to tensor."""
        return tensor if pos is None else tensor + pos

    def forward_post(  # noqa: PLR0913
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)

    def forward_pre(  # noqa: PLR0913
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        return tgt + self.dropout(tgt2)

    def forward(  # noqa: PLR0913
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            memory_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class FFNLayer(nn.Module):
    """Feedforward layer."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,  # noqa: FBT001,FBT002
    ) -> None:
        """Initialize FFNLayer."""
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Reset parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(
        self,
        tensor: torch.Tensor,
        pos: torch.Tensor | None,
    ) -> torch.Tensor:
        """Add positional embedding to tensor."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)

    def forward_pre(self, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        return tgt + self.dropout(tgt2)

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        """Initialize MLP."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            (
                nn.Linear(n, k)
                for n, k in zip([input_dim, *h], [*h, output_dim], strict=False)
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, layer in enumerate(self.layers):
            x = fn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Conv2d(torch.nn.Conv2d):
    """A wrapper around :class:`torch.nn.Conv2d`."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize Conv2d."""
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = fn.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """Position Embedding."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: float = 10000,
        normalize: bool = False,  # noqa: FBT001,FBT002
        scale: float | None = None,
    ) -> None:
        """Initialize Position Embedding."""
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            err_msg = "normalize should be True if scale is passed"
            raise ValueError(err_msg)
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)),
                device=x.device,
                dtype=torch.bool,
            )
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    def __repr__(self, _repr_indent: int = 4) -> str:
        """Return string representation of the module."""
        head = "Positional encoding " + self.__class__.__name__
        body = [
            f"num_pos_feats: {self.num_pos_feats}",
            f"temperature: {self.temperature}",
            f"normalize: {self.normalize}",
            f"scale: {self.scale}",
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
