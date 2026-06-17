"""Difference Feature Attention (DFA) — Multi-scale attention on change features.

In ChangeFormer, the decoder computes feature differences at 4 scales (c1→c4).
These differences can contain noise (speckle-induced false changes) or
irrelevant changes (e.g., moisture variations vs actual fire).

The DFA module wraps around the decoder's difference features and applies
learned attention to suppress false change signals.  It operates on the
CONCATENATED difference features before the final fusion, re-weighting each
scale's contribution based on global context.

This is a DECODER-LEVEL enhancement — it wraps the ChangeFormer decoder output
without modifying the internal decoder code.

Two integration strategies (both non-invasive):
  A) Post-decoder: apply DFA on the list of decoder outputs before loss computation
  B) Wrapper: wrap the decoder forward to apply DFA inside ChangeDetectionModel

Usage in ChangeDetectionModel:
    self.dfa = DifferenceFeatureAttention(
        embedding_dim=256,
        num_scales=4,
    ) if use_dfa else None

    # In forward, after getting decoder outputs:
    outputs = self.change_detection_model(x1, x2)  # list of predictions
    if self.dfa is not None:
        outputs = self.dfa(outputs)  # re-weighted multi-scale outputs
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DifferenceFeatureAttention(nn.Module):
    """Attention module for multi-scale decoder difference features.

    Takes the list of decoder predictions (one per scale + final) and applies
    a learned gating mechanism to suppress unreliable predictions at each scale.

    This helps with:
    - Reducing false positives from speckle at fine scales
    - Emphasizing coarse-scale predictions for large fires
    - Adapting scale importance during training

    Args:
        num_classes: Number of output classes (must match decoder output channels).
        num_scales: Number of intermediate decoder heads (typically 4 for ChangeFormer).
        gate_hidden: Hidden dimension for the per-scale gating MLP.
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_scales: int = 4,
        gate_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.num_classes = num_classes

        # Per-scale gates: learn a scalar weight for each intermediate prediction
        # based on the prediction's own statistics (adaptive gating)
        self.scale_gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),   # [B, C, 1, 1]
                nn.Flatten(),               # [B, C]
                nn.Linear(num_classes, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_scales)
        ])

        # Initialize gates to output ~1.0 (identity at start)
        for gate in self.scale_gates:
            nn.init.constant_(gate[-2].bias, 2.0)  # sigmoid(2) ≈ 0.88

    def forward(self, outputs: list[Tensor]) -> list[Tensor]:
        """Apply learned attention gates to intermediate decoder outputs.

        Args:
            outputs: List of [B, C, H_i, W_i] tensors from the decoder.
                Convention: outputs[:-1] are intermediate heads, outputs[-1] is final.

        Returns:
            Gated outputs list (same shapes). The final head is NOT gated.
        """
        if len(outputs) <= 1:
            return outputs

        gated_outputs = []
        num_intermediate = min(self.num_scales, len(outputs) - 1)

        for i in range(num_intermediate):
            pred = outputs[i]
            gate_weight = self.scale_gates[i](pred)  # [B, 1]
            gate_weight = gate_weight.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
            gated_outputs.append(pred * gate_weight)

        # Pass through any remaining intermediate outputs ungated
        for i in range(num_intermediate, len(outputs) - 1):
            gated_outputs.append(outputs[i])

        # Final prediction is never gated (it's the primary output)
        gated_outputs.append(outputs[-1])

        return gated_outputs


class MultiScaleConsistencyLoss(nn.Module):
    """Auxiliary loss that encourages multi-scale prediction consistency.

    Penalizes disagreement between intermediate decoder heads and the final
    prediction, encouraging all scales to converge toward the same answer.
    This acts as self-distillation from the final head to intermediate heads.

    Usage:
        consistency_loss = MultiScaleConsistencyLoss()
        # In training_step, after getting outputs:
        aux_loss = consistency_loss(outputs)
        total_loss = main_loss + 0.1 * aux_loss
    """

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, outputs: list[Tensor]) -> Tensor:
        """Compute KL divergence between intermediate and final predictions.

        Args:
            outputs: List of decoder outputs [p_c4, p_c3, p_c2, p_c1, final].

        Returns:
            Scalar consistency loss.
        """
        if len(outputs) <= 1:
            return torch.tensor(0.0, device=outputs[0].device)

        final = outputs[-1]
        target_size = final.shape[2:]
        # Teacher: detached final prediction as soft targets
        teacher_probs = F.softmax(final.detach() / self.temperature, dim=1)

        total_loss = torch.tensor(0.0, device=final.device, dtype=final.dtype)
        count = 0

        for pred in outputs[:-1]:
            # Resize intermediate prediction to final resolution
            if pred.shape[2:] != target_size:
                pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)

            student_log_probs = F.log_softmax(pred / self.temperature, dim=1)
            kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            total_loss = total_loss + kl
            count += 1

        if count > 0:
            total_loss = total_loss / count * (self.temperature ** 2)

        return total_loss
