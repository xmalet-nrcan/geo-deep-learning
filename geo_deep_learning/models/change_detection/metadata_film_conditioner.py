"""FiLM (Feature-wise Linear Modulation) conditioner for RCM SAR metadata.

Instead of feeding SAT_PASS and BEAM as constant spatial bands to the
Transformer encoder (wasting 2 of its channels), this module encodes them
as learned embeddings and produces per-channel scale (gamma) and shift (beta)
parameters to modulate the input features.

    conditioned = gamma * features + beta

This is lightweight (< 0.01% of ChangeFormer parameters) but allows the
model to adapt its behaviour to acquisition geometry -- critical for SAR
where ascending/descending pass and beam mode affect backscatter.

The conditioner is **extensible**: any number of categorical metadata fields
can be added (season, satellite ID, time-delta bin, etc.) via the
``metadata_fields`` dict.

Reference:
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer",
    AAAI 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MetadataFiLMConditioner(nn.Module):
    """Generate FiLM parameters from categorical acquisition metadata.

    Each metadata field is embedded independently, then all embeddings are
    concatenated and passed through a shared MLP to produce per-channel
    gamma (scale) and beta (shift).

    Args:
        in_channels: Number of image channels to modulate.
        metadata_fields: Dict mapping field name to number of categories.
            Example: {"sat_pass": 2, "beam": 4, "season": 4, "satellite_id": 3}
            If None, falls back to default {"sat_pass": num_sat_pass, "beam": num_beams}.
        num_sat_pass: (backward compat) Number of satellite pass categories.
        num_beams: (backward compat) Number of beam categories.
        embed_dim: Dimension of the embedding vector per field.
    """

    def __init__(
        self,
        in_channels: int,
        num_sat_pass: int = 2,
        num_beams: int = 4,
        embed_dim: int = 32,
        metadata_fields: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Build embedding layers for each metadata field
        if metadata_fields is not None:
            self._field_names = list(metadata_fields.keys())
            self.embeddings = nn.ModuleDict({
                name: nn.Embedding(num_classes, embed_dim)
                for name, num_classes in metadata_fields.items()
            })
        else:
            # Backward-compatible default: sat_pass + beam
            self._field_names = ["sat_pass", "beam"]
            self.embeddings = nn.ModuleDict({
                "sat_pass": nn.Embedding(num_sat_pass, embed_dim),
                "beam": nn.Embedding(num_beams, embed_dim),
            })

        total_embed_dim = embed_dim * len(self._field_names)

        # MLP: concat embeddings -> FiLM parameters (gamma, beta) for each channel
        self.film_generator = nn.Sequential(
            nn.Linear(total_embed_dim, total_embed_dim),
            nn.GELU(),
            nn.Linear(total_embed_dim, in_channels * 2),  # gamma and beta
        )

        # Initialize so gamma=1, beta=0 -> identity at start of training
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        with torch.no_grad():
            self.film_generator[-1].bias[:in_channels] = 1.0

    @property
    def field_names(self) -> list[str]:
        """Names of expected metadata fields in order."""
        return self._field_names

    def forward(
        self,
        features: Tensor,
        sat_pass: Tensor | None = None,
        beam: Tensor | None = None,
        **metadata_kwargs: Tensor,
    ) -> Tensor:
        """Apply FiLM conditioning to input features.

        Args:
            features: [B, C, H, W] image features.
            sat_pass: [B] integer tensor (0=Ascending, 1=Descending).
            beam: [B] integer tensor (0=A, 1=B, 2=C, 3=D).
            **metadata_kwargs: Additional metadata fields as [B] int tensors.
                Keys must match field names provided at init.
                Example: season=tensor([2, 0, 1, ...])

        Returns:
            Conditioned features [B, C, H, W].
        """
        # Collect all metadata tensors in field order
        all_metadata: dict[str, Tensor | None] = {}
        if sat_pass is not None:
            all_metadata["sat_pass"] = sat_pass
        if beam is not None:
            all_metadata["beam"] = beam
        all_metadata.update(metadata_kwargs)

        # Embed each field and concatenate
        embedded = []
        for name in self._field_names:
            if name not in all_metadata or all_metadata[name] is None:
                # Missing field -> zeros (neutral, no effect on conditioning)
                batch_size = features.shape[0]
                embedded.append(
                    torch.zeros(batch_size, self.embed_dim, device=features.device)
                )
            else:
                embedded.append(
                    self.embeddings[name](all_metadata[name].long())
                )

        combined = torch.cat(embedded, dim=1)  # [B, total_embed_dim]
        film_params = self.film_generator(combined)  # [B, 2*C]

        # Split into gamma (scale) and beta (shift)
        gamma, beta = film_params.split(self.in_channels, dim=1)
        # Reshape for broadcasting: [B, C] -> [B, C, 1, 1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * features + beta