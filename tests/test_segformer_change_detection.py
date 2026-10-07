"""Tests for the SegFormer change-detection backbone."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from geo_deep_learning.models.change_detection.change_detection_model import (  # noqa: E402
    ChangeDetectionModel,
)
from geo_deep_learning.models.change_detection.sub_models.segformer_cd import (  # noqa: E402
    FUSION_MODES,
    SegFormerChangeDetection,
    SegFormerChangeDetectionB0,
)


@pytest.mark.parametrize("fusion", FUSION_MODES)
def test_segformer_cd_outputs(fusion: str) -> None:
    model = SegFormerChangeDetection(input_nc=10, output_nc=2, encoder="mit_b0", fusion=fusion)
    x1, x2 = torch.randn(2, 10, 64, 64), torch.randn(2, 10, 64, 64)
    outputs = model(x1, x2)
    assert isinstance(outputs, list)
    assert len(outputs) == 5  # [p_c4, p_c3, p_c2, p_c1, final] like ChangeFormer
    for out in outputs:
        assert out.shape == (2, 2, 64, 64)


def test_segformer_cd_all_params_get_grad() -> None:
    model = SegFormerChangeDetectionB0(input_nc=4, output_nc=2)
    model.train()
    outputs = model(torch.randn(2, 4, 64, 64), torch.randn(2, 4, 64, 64))
    sum(o.sum() for o in outputs).backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing


def test_change_detection_model_segformer_with_extras() -> None:
    model = ChangeDetectionModel(
        change_detection_model="segformer",
        in_channels=9,
        out_channels=2,
        use_metadata_film=True,
        use_cbam=True,
        use_dfa=True,
        use_signed_difference=True,
        encoder="mit_b0",
        fusion="concat",
    )
    sat_pass = torch.tensor([0, 1])
    beam = torch.tensor([0, 3])
    outputs = model(torch.randn(2, 9, 64, 64), torch.randn(2, 9, 64, 64), sat_pass=sat_pass, beam=beam)
    assert len(outputs) == 5
    assert outputs[-1].shape == (2, 2, 64, 64)


def test_unknown_encoder_raises() -> None:
    with pytest.raises(ValueError, match="Unknown SegFormer encoder"):
        SegFormerChangeDetection(input_nc=3, output_nc=2, encoder="mit_b9")
