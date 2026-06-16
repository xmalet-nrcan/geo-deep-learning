"""FiLM (Feature-wise Linear Modulation) conditioner for RCM SAR metadata.

Instead of feeding SAT_PASS and BEAM as constant spatial bands to the
Transformer encoder (wasting 2 of its channels), this module encodes them
as learned embeddings and produces per-channel scale (γ) and shift (β)
parameters to modulate the input features.

    conditioned = γ ⊙ features + β

This is lightweight (< 0.01% of ChangeFormer parameters) but allows the
model to adapt its behaviour to acquisition geometry — critical for SAR
where ascending/descending pass and beam mode affect backscatter.

Reference:
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer",
    AAAI 2018.
"""

import torch
import torch.nn as nn
from torch import Tensor


class MetadataFiLMConditioner(nn.Module):
    """Generate FiLM parameters from SAT_PASS and BEAM metadata.

    Args:
        num_sat_pass: Number of satellite pass categories (default 2: ASC/DESC).
        num_beams: Number of beam categories (default 4: A/B/C/D).
        embed_dim: Dimension of the embedding vectors.
        in_channels: Number of image channels to modulate (γ and β per channel).
    """

    def __init__(
        self,
        in_channels: int,
        num_sat_pass: int = 2,
        num_beams: int = 4,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        # Learned embeddings for categorical metadata
        self.sat_pass_embed = nn.Embedding(num_sat_pass, embed_dim)
        self.beam_embed = nn.Embedding(num_beams, embed_dim)

        # MLP: concat embeddings → FiLM parameters (γ, β) for each channel
        self.film_generator = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, in_channels * 2),  # γ and β
        )

        # Initialize so γ ≈ 1, β ≈ 0 → identity at start of training
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        # Set the gamma portion of bias to 1.0
        with torch.no_grad():
            self.film_generator[-1].bias[:in_channels] = 1.0

    def forward(
        self,
        features: Tensor,
        sat_pass: Tensor,
        beam: Tensor,
    ) -> Tensor:
        """Apply FiLM conditioning to input features.

        Args:
            features: [B, C, H, W] image features.
            sat_pass: [B] integer tensor (0=Ascending, 1=Descending).
            beam: [B] integer tensor (0=A, 1=B, 2=C, 3=D).

        Returns:
            Conditioned features [B, C, H, W].
        """
        # Embed metadata → [B, embed_dim] each
        sp_emb = self.sat_pass_embed(sat_pass.long())  # [B, embed_dim]
        bm_emb = self.beam_embed(beam.long())  # [B, embed_dim]

        # Concatenate and generate FiLM parameters
        combined = torch.cat([sp_emb, bm_emb], dim=1)  # [B, 2*embed_dim]
        film_params = self.film_generator(combined)  # [B, 2*C]

        # Split into γ (scale) and β (shift)
        gamma, beta = film_params.split(self.in_channels, dim=1)
        # Reshape for broadcasting: [B, C] → [B, C, 1, 1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * features + beta
