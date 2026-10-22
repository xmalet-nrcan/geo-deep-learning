"""Tests for the torchgeo ChangeStar(FarSeg) change-detection backbone."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchgeo")

from geo_deep_learning.models.change_detection.change_detection_model import (  # noqa: E402
    ChangeDetectionModel,
)
from geo_deep_learning.models.change_detection.sub_models.changestar_torchgeo import (  # noqa: E402
    ChangeStarTorchGeo,
    _adapt_first_conv,
)

BACKBONE = "resnet18"  # fastest FarSeg backbone for tests


def _pair(b: int = 2, c: int = 10, h: int = 64, w: int = 64) -> tuple:
    return torch.randn(b, c, h, w), torch.randn(b, c, h, w)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("size", [(64, 64), (70, 45)])  # 70x45: not a multiple of 32
def test_output_contract(training: bool, size: tuple[int, int]) -> None:
    model = ChangeStarTorchGeo(input_nc=10, output_nc=2, backbone=BACKBONE)
    model.train(training)
    outputs = model(*_pair(h=size[0], w=size[1]))
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert outputs[-1].shape == (2, 2, *size)


def test_two_class_softmax_matches_sigmoid() -> None:
    """softmax([-z/2, z/2])[1] must equal sigmoid(z) of the raw ChangeMixin logit."""
    model = ChangeStarTorchGeo(input_nc=4, output_nc=2, backbone=BACKBONE).eval()
    raw = ChangeStarTorchGeo(input_nc=4, output_nc=1, backbone=BACKBONE).eval()
    raw.load_state_dict(model.state_dict())
    x1, x2 = _pair(c=4)
    with torch.no_grad():
        prob = torch.softmax(model(x1, x2)[-1], dim=1)[:, 1:]
        ref = torch.sigmoid(raw(x1, x2)[-1])
    torch.testing.assert_close(prob, ref)


def test_matches_torchgeo_reference() -> None:
    """Same weights => same change probability as torchgeo's ChangeStarFarSeg (eval)."""
    from torchgeo.models import ChangeStarFarSeg

    reference = ChangeStarFarSeg(backbone=BACKBONE, classes=1).eval()
    model = ChangeStarTorchGeo(input_nc=3, output_nc=2, backbone=BACKBONE).eval()
    reference_state = {
        k: v for k, v in reference.dense_feature_extractor.state_dict().items()
        if not k.startswith("backbone.fc.")  # unused ImageNet classifier, dropped by the adapter
    }
    model.dense_feature_extractor.load_state_dict(reference_state)
    model.changemixin.load_state_dict(reference.changemixin.state_dict())

    x1, x2 = _pair(c=3)
    with torch.no_grad():
        expected = reference(torch.stack([x1, x2], dim=1))["change_prob"]
        got = torch.softmax(model(x1, x2)[-1], dim=1)[:, 1:]
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)


def test_bidirectional_training_returns_both_directions() -> None:
    model = ChangeStarTorchGeo(input_nc=10, backbone=BACKBONE, bidirectional_training=True)
    model.train()
    assert len(model(*_pair())) == 2
    model.eval()
    assert len(model(*_pair())) == 1


def test_mean_inference_mode_is_order_invariant() -> None:
    model = ChangeStarTorchGeo(
        input_nc=10, backbone=BACKBONE, inference_mode="mean", bidirectional_training=True,
    ).eval()
    x1, x2 = _pair()
    with torch.no_grad():
        torch.testing.assert_close(model(x1, x2)[-1], model(x2, x1)[-1], atol=1e-5, rtol=1e-4)


def test_all_params_get_grad() -> None:
    """Segmentation head is dropped: no unused parameter (DDP-safe)."""
    model = ChangeStarTorchGeo(input_nc=10, backbone=BACKBONE).train()
    sum(o.sum() for o in model(*_pair())).backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing


def test_adapt_first_conv_keeps_pretrained_response() -> None:
    """A constant input gives the same response before/after channel adaptation."""
    conv = torch.nn.Conv2d(3, 8, 7, 2, 3, bias=False)
    adapted = _adapt_first_conv(conv, 10, pretrained=True)
    assert adapted.in_channels == 10
    with torch.no_grad():
        torch.testing.assert_close(adapted(torch.ones(1, 10, 16, 16)), conv(torch.ones(1, 3, 16, 16)))


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="output_nc"):
        ChangeStarTorchGeo(input_nc=3, output_nc=3, backbone=BACKBONE)
    with pytest.raises(ValueError, match="inference_mode"):
        ChangeStarTorchGeo(input_nc=3, backbone=BACKBONE, inference_mode="both")
    with pytest.raises(ValueError, match="backbone_weights"):
        ChangeStarTorchGeo(input_nc=3, backbone=BACKBONE, backbone_weights="NotAWeight.FOO")


def test_change_detection_model_with_extras() -> None:
    """Registry key + FiLM/CBAM/channel-dropout/signed-difference, as in the RCM config."""
    model = ChangeDetectionModel(
        change_detection_model="changestar_farseg",
        in_channels=10,
        out_channels=2,
        use_metadata_film=True,
        use_cbam=True,
        use_channel_dropout=True,
        use_signed_difference=True,
        signed_difference_normalize=True,
        backbone=BACKBONE,
    )
    model.train()
    outputs = model(
        *_pair(),
        sat_pass=torch.tensor([0, 1]),
        beam=torch.tensor([0, 3]),
    )
    assert len(outputs) == 1
    assert outputs[-1].shape == (2, 2, 64, 64)


# ---------------------------------------------------------------------------
# Full-resolution refinement head
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("training", [True, False])
def test_refine_output_contract(training: bool) -> None:
    model = ChangeStarTorchGeo(input_nc=10, backbone=BACKBONE, full_res_refine=True)
    model.train(training)
    outputs = model(*_pair(h=70, w=45))
    assert len(outputs) == 1
    assert outputs[-1].shape == (2, 2, 70, 45)


def test_refine_starts_as_plain_changestar() -> None:
    """Zero-initialised residual: identical predictions with/without the head at init."""
    plain = ChangeStarTorchGeo(input_nc=10, backbone=BACKBONE).eval()
    refined = ChangeStarTorchGeo(input_nc=10, backbone=BACKBONE, full_res_refine=True).eval()
    missing, unexpected = refined.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith("refine_head.") for k in missing)
    x1, x2 = _pair()
    with torch.no_grad():
        torch.testing.assert_close(refined(x1, x2)[-1], plain(x1, x2)[-1])


def test_refine_head_adds_full_resolution_detail() -> None:
    """Once trained (non-zero residual), the head changes the output at the pixel level."""
    model = ChangeStarTorchGeo(input_nc=4, backbone=BACKBONE, full_res_refine=True).eval()
    torch.nn.init.normal_(model.refine_head.residual.weight)
    x1, x2 = _pair(c=4)
    with torch.no_grad():
        logits = model(x1, x2)[-1][:, 1]
    # Bilinear x4 upsampling alone is smooth; the refined map has pixel-scale variation.
    pixel_jumps = (logits[..., 1:] - logits[..., :-1]).abs().mean()
    assert pixel_jumps > 1e-3


def test_refine_all_params_get_grad() -> None:
    model = ChangeStarTorchGeo(input_nc=10, backbone=BACKBONE, full_res_refine=True).train()
    sum(o.sum() for o in model(*_pair())).backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing


def test_refine_mean_mode_is_order_invariant() -> None:
    model = ChangeStarTorchGeo(
        input_nc=10, backbone=BACKBONE, inference_mode="mean",
        bidirectional_training=True, full_res_refine=True,
    ).eval()
    torch.nn.init.normal_(model.refine_head.residual.weight)
    x1, x2 = _pair()
    with torch.no_grad():
        torch.testing.assert_close(model(x1, x2)[-1], model(x2, x1)[-1], atol=1e-5, rtol=1e-4)
