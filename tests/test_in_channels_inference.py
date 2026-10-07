"""Tests for automatic ``in_channels`` inference from ``band_names``."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("rasterio")
pytest.importorskip("lightning")
pytest.importorskip("kornia")

from geo_deep_learning.datasets.rcm_change_detection_dataset import (  # noqa: E402
    BandName,
    RCMChangeDetectionDataset,
)
from geo_deep_learning.datasets.rcm_change_detection_dataset_merge_pre_post import (  # noqa: E402
    RCMChangeDetectionDatasetMergePrePost,
)
from geo_deep_learning.tasks_with_models.change_detection_changeformer import (  # noqa: E402
    ChangeDetectionChangeFormer,
)

NINE_BANDS = ["LOCALINCANGLE", "PDN", "PSN", "PVN", "S0", "RR", "RL", "NDSV", "RFDI"]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"band_names": NINE_BANDS}, 10),
        ({"band_names": ["LOCALINCANGLE", "RR", "RL"]}, 4),
        ({"bands": [2, 9, 10]}, 4),
        ({}, len(BandName)),
        ({"band_names": NINE_BANDS, "separate_metadata": False}, 13),
    ],
)
def test_dataset_num_input_channels(kwargs: dict, expected: int) -> None:
    assert RCMChangeDetectionDataset.num_input_channels(**kwargs) == expected


def test_merge_pre_post_doubles_channels() -> None:
    n = RCMChangeDetectionDatasetMergePrePost.num_input_channels(
        band_names=["S0", "RR"], separate_metadata=False,
    )
    assert n == 2 * (2 + 1 + 3)


def _fake_module(in_channels: int | None, num_input_channels: int | None) -> SimpleNamespace:
    datamodule = (
        SimpleNamespace(num_input_channels=num_input_channels)
        if num_input_channels is not None else None
    )
    trainer = SimpleNamespace(datamodule=datamodule)
    return SimpleNamespace(
        _trainer=trainer,
        in_channels=in_channels,
        hparams={"in_channels": in_channels},
        _hparams_initial={"in_channels": in_channels},
    )


def test_resolve_in_channels_inferred() -> None:
    module = _fake_module(in_channels=None, num_input_channels=10)
    assert ChangeDetectionChangeFormer._resolve_in_channels(module) == 10
    assert module.in_channels == 10
    assert module.hparams["in_channels"] == 10  # persisted in checkpoints
    assert module._hparams_initial["in_channels"] == 10


def test_resolve_in_channels_datamodule_wins_on_mismatch() -> None:
    module = _fake_module(in_channels=8, num_input_channels=10)
    assert ChangeDetectionChangeFormer._resolve_in_channels(module) == 10


def test_resolve_in_channels_without_datamodule_uses_config() -> None:
    module = _fake_module(in_channels=10, num_input_channels=None)
    assert ChangeDetectionChangeFormer._resolve_in_channels(module) == 10


def test_resolve_in_channels_without_any_source_raises() -> None:
    module = _fake_module(in_channels=None, num_input_channels=None)
    with pytest.raises(ValueError, match="in_channels could not be inferred"):
        ChangeDetectionChangeFormer._resolve_in_channels(module)
