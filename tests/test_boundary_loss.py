"""Tests for the boundary-band BCE of the change-detection task."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("kornia")

import torch.nn.functional as F  # noqa: E402

from geo_deep_learning.tasks_with_models.change_detection_changeformer import (  # noqa: E402
    ChangeDetectionChangeFormer as Task,
)


def _square_target(h: int = 16, w: int = 16) -> torch.Tensor:
    """Binary [1, 1, H, W] mask with a burned square in the middle."""
    target = torch.zeros(1, 1, h, w)
    target[..., 5:11, 5:11] = 1.0
    return target


def _two_class(target: torch.Tensor) -> torch.Tensor:
    return torch.cat([1 - target, target], dim=1)


def test_band_covers_both_sides_of_the_edge() -> None:
    target = _square_target()
    band = Task._boundary_band(target, torch.ones_like(target), width=1)
    assert band[0, 0, 4, 8] == 1  # just outside the square
    assert band[0, 0, 5, 8] == 1  # just inside
    assert band[0, 0, 8, 8] == 0  # deep inside
    assert band[0, 0, 0, 0] == 0  # far outside


def test_band_ignores_invalid_pixels() -> None:
    target = _square_target()
    valid = torch.ones_like(target)
    valid[..., :, :8] = 0
    band = Task._boundary_band(target * valid, valid, width=1)
    assert band[..., :, :8].sum() == 0


def test_no_boundary_gives_zero_loss() -> None:
    target = torch.zeros(1, 1, 16, 16)
    logits = torch.randn(1, 2, 16, 16, requires_grad=True)
    loss = Task._boundary_bce_loss(logits, _two_class(target), torch.ones_like(target), 2)
    assert loss.item() == 0.0


def test_loss_matches_bce_on_band_and_uses_softmax_logit() -> None:
    target = _square_target()
    valid = torch.ones_like(target)
    logits = torch.randn(1, 2, 16, 16)
    loss = Task._boundary_bce_loss(logits, _two_class(target), valid, 1)

    band = Task._boundary_band(target, valid, 1).bool()
    z = logits[:, 1:] - logits[:, :1]  # sigmoid(z) == softmax(logits)[:, 1]
    expected = F.binary_cross_entropy_with_logits(z[band], target[band])
    torch.testing.assert_close(loss, expected)


def test_loss_rewards_correct_boundaries() -> None:
    target = _square_target()
    valid = torch.ones_like(target)
    good = _two_class(target) * 10 - 5  # confident, correct
    bad = -good  # confident, wrong
    assert Task._boundary_bce_loss(good, _two_class(target), valid, 2) < 0.01
    assert Task._boundary_bce_loss(bad, _two_class(target), valid, 2) > 5
