"""
Tests for predict output layouts of ``ChangeDetectionChangeFormer`` (dev plan P4).

``production`` (default, consumed by scanfire) must stay byte-for-byte identical
in structure; ``benchmark`` writes ``<dir>/<event_id>/<output_name>.tif`` for
multi-model comparison (see docs/dev-plans/2026-10-08_model_benchmark_predict.md).

No GPU / checkpoint needed: ``on_predict_end`` is fed with synthetic
``predict_step``-like batch dicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("rasterio")
pytest.importorskip("lightning")
pytest.importorskip("kornia")

import numpy as np
import rasterio as rio
import torch
from rasterio.transform import Affine

from geo_deep_learning.tasks_with_models.change_detection_changeformer import (
    BENCHMARK_LATEST_MANIFEST,
    BENCHMARK_RUNS_DIRNAME,
    ChangeDetectionChangeFormer,
)
from geo_deep_learning.utils.geotiff_merge import merge_predictions

EVENT_ID = 42
TILE = 4  # pixels per cell side in the synthetic data
PIXEL = 10.0  # metres per pixel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _loss(*_args: object, **_kwargs: object) -> torch.Tensor:
    return torch.tensor(0.0)


def _make_module(
    tmp_path: Path,
    predictions: list[dict],
    *,
    datamodule: object | None = None,
    **kwargs: object,
) -> ChangeDetectionChangeFormer:
    module = ChangeDetectionChangeFormer(
        "changestar2",
        image_size=(TILE, TILE),
        num_classes=1,
        max_samples=1,
        main_loss=_loss,
        secondary_loss=_loss,
        **kwargs,
    )
    module._trainer = SimpleNamespace(  # noqa: SLF001 — minimal trainer stub
        predict_loop=SimpleNamespace(predictions=predictions),
        datamodule=datamodule or SimpleNamespace(tile_size=None, tile_stride=None),
        default_root_dir=str(tmp_path / "default_root"),
    )
    return module


def _output_name(cell_id: str, gid_post: int = 3, date_post: str = "20250710") -> str:
    """Build a name with the convention of scanfire's ``build_output_name``."""
    return (
        f"event-{EVENT_ID}_start-20250701_end_20250725"
        f"_pre-g1-20250620_post-g{gid_post}-{date_post}"
        f"_cell-{cell_id}_beam-A_pass-ASC.tif"
    )


def _merged_name(gid_post: int = 3, date_post: str = "20250710") -> str:
    """Build a name like ``geotiff_merge.merged_group_filename`` (no cell)."""
    return (
        f"event-{EVENT_ID}_start-20250701_end_20250725"
        f"_pre-g1-20250620_post-g{gid_post}-{date_post}_beam-A_pass-ASC.tif"
    )


def _per_tile_batch(
    cells: list[tuple[str, int]], *, gid_post: int = 3, date_post: str = "2025-07-10",
) -> dict:
    """
    Batch for the per-tile (non-blended) path.

    ``cells`` = ``[(cell_id, column_index), ...]``; each cell is a TILExTILE
    raster placed side by side (column_index x TILE pixels to the east).
    """
    n = len(cells)
    preds = torch.zeros((n, TILE, TILE), dtype=torch.long)
    for i in range(n):
        preds[i, : i + 1, :] = 1  # distinct content per sample
    prob = torch.full((n, TILE, TILE), 0.25, dtype=torch.float32)
    transforms = [
        Affine(PIXEL, 0, col * TILE * PIXEL, 0, -PIXEL, 1000.0) for _, col in cells
    ]
    compact = date_post.replace("-", "")
    return {
        "predictions": preds,
        "probability": prob,
        "pre_post_name": [f"{cell}|pair" for cell, _ in cells],
        "cell_id": [cell for cell, _ in cells],
        "profile": {
            "crs": ["EPSG:3979"] * n,
            # default_collate transposes a list of 6-coefficient lists
            "transform": [
                torch.tensor([t[k] for t in transforms], dtype=torch.float64)
                for k in range(6)
            ],
        },
        "original_height": torch.tensor([TILE] * n),
        "original_width": torch.tensor([TILE] * n),
        "pair_id": torch.arange(1, n + 1),
        "event_id": torch.tensor([EVENT_ID] * n),
        "event_start_date": ["2025-07-01"] * n,
        "event_end_date": ["2025-07-25"] * n,
        "beam": ["A"] * n,
        "sat_pass": ["ASC"] * n,
        "group_id_pre": torch.tensor([1] * n),
        "group_id_post": torch.tensor([gid_post] * n),
        "group_date_pre": ["2025-06-20"] * n,
        "group_date_post": [date_post] * n,
        "output_name": [_output_name(cell, gid_post, compact) for cell, _ in cells],
    }


def _blended_batch() -> dict:
    """One 6x6 source cell split in four overlapping 4x4 tiles (stride 2)."""
    origins = [(0, 0), (0, 2), (2, 0), (2, 2)]
    n = len(origins)
    probs = torch.zeros((n, 2, TILE, TILE), dtype=torch.float32)
    probs[:, 0] = 0.3
    probs[:, 1] = 0.7  # burn everywhere
    src = Affine(PIXEL, 0, 0.0, 0, -PIXEL, 1000.0)
    tile_tf = [src * Affine.translation(c, r) for r, c in origins]
    return {
        "predictions": torch.ones((n, TILE, TILE), dtype=torch.long),
        "probability": probs[:, 1].clone(),
        "probabilities": probs,
        "mask_common": torch.ones((n, 1, TILE, TILE)),
        "pre_post_name": [f"C9|pair|tile_{r}_{c}" for r, c in origins],
        "cell_id": ["C9"] * n,
        "profile": {
            "crs": ["EPSG:3979"] * n,
            "transform": [
                torch.tensor([t[k] for t in tile_tf], dtype=torch.float64)
                for k in range(6)
            ],
        },
        "original_height": torch.tensor([TILE] * n),
        "original_width": torch.tensor([TILE] * n),
        "tile_row_start": torch.tensor([r for r, _ in origins]),
        "tile_col_start": torch.tensor([c for _, c in origins]),
        "source_height": torch.tensor([6] * n),
        "source_width": torch.tensor([6] * n),
        "pair_id": torch.tensor([7] * n),
        "event_id": torch.tensor([EVENT_ID] * n),
        "event_start_date": ["2025-07-01"] * n,
        "event_end_date": ["2025-07-25"] * n,
        "beam": ["A"] * n,
        "sat_pass": ["ASC"] * n,
        "group_id_pre": torch.tensor([1] * n),
        "group_id_post": torch.tensor([3] * n),
        "group_date_pre": ["2025-06-20"] * n,
        "group_date_post": ["2025-07-10"] * n,
        "output_name": [_output_name("C9")] * n,
    }


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# Configuration / helpers
# ---------------------------------------------------------------------------


def test_invalid_layout_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="predict_output_layout"):
        _make_module(tmp_path, [], predict_output_layout="flat")


@pytest.mark.parametrize(
    ("layout", "explicit", "expected"),
    [
        ("production", None, True),
        ("benchmark", None, False),
        ("benchmark", True, True),
        ("production", False, False),
    ],
)
def test_merged_all_default(
    tmp_path: Path, layout: str, explicit: bool | None, expected: bool,
) -> None:
    module = _make_module(
        tmp_path, [], predict_output_layout=layout, predict_write_merged_all=explicit,
    )
    assert module.predict_write_merged_all is expected


@pytest.mark.parametrize(
    ("layout", "explicit", "expected"),
    [
        ("production", None, False),
        ("benchmark", None, True),
        ("benchmark", False, False),
        ("production", True, True),
    ],
)
def test_weights_strict_default(
    tmp_path: Path, layout: str, explicit: bool | None, expected: bool,
) -> None:
    module = _make_module(
        tmp_path, [], predict_output_layout=layout, weights_strict=explicit,
    )
    assert module.weights_strict is expected


def test_weights_strict_incompatible_with_load_parts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="load_parts"):
        _make_module(tmp_path, [], weights_strict=True, load_parts=["encoder"])
    # non-strict (production default) partial loading stays allowed
    _make_module(tmp_path, [], load_parts=["encoder"])


@pytest.mark.parametrize(("layout", "expected"), [("production", False), ("benchmark", True)])
def test_configure_model_forwards_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str, expected: bool,
) -> None:
    import geo_deep_learning.tasks_with_models.change_detection_changeformer as cdc  # noqa: PLC0415

    calls: list[dict] = []
    monkeypatch.setattr(cdc, "ChangeDetectionModel", lambda **_kw: torch.nn.Linear(2, 2))
    monkeypatch.setattr(
        cdc,
        "load_weights_from_checkpoint",
        lambda _model, path, **kw: calls.append({"path": path, **kw}),
    )
    module = _make_module(
        tmp_path,
        [],
        in_channels=10,
        predict_output_layout=layout,
        weights_from_checkpoint_path="best.ckpt",
    )
    module.configure_model()
    assert len(calls) == 1
    assert calls[0]["path"] == "best.ckpt"
    assert calls[0]["strict"] is expected


def test_load_weights_strict_raises_on_mismatch(tmp_path: Path) -> None:
    from geo_deep_learning.utils.models import load_weights_from_checkpoint  # noqa: PLC0415

    src = torch.nn.Linear(2, 2)
    state = {f"model.{k}": v for k, v in src.state_dict().items()}
    ckpt = tmp_path / "best.ckpt"
    torch.save({"state_dict": {**state, "model.extra.weight": torch.zeros(1)}}, ckpt)

    with pytest.raises(RuntimeError, match="extra.weight"):
        load_weights_from_checkpoint(torch.nn.Linear(2, 2), str(ckpt), strict=True)
    # non-strict: loads compatible weights, ignores the unexpected key
    dst = torch.nn.Linear(2, 2)
    result = load_weights_from_checkpoint(dst, str(ckpt), strict=False)
    assert result.unexpected_keys == ["extra.weight"]
    assert torch.equal(dst.weight, src.weight)


@pytest.mark.parametrize(
    ("layout", "output_dir", "run_name", "expected"),
    [
        ("production", "out", None, "out/predictions"),
        ("production", "out/predictions", None, "out/predictions"),
        ("production", None, None, "default_root/predictions"),
        ("benchmark", "out/model_a", None, "out/model_a"),
        ("benchmark", "out/predictions", None, "out/predictions"),
        ("benchmark", None, "model_a", "default_root/benchmark/model_a"),
        ("benchmark", None, None, "default_root/benchmark/changestar2"),
    ],
)
def test_resolve_predict_base_dir(
    tmp_path: Path,
    layout: str,
    output_dir: str | None,
    run_name: str | None,
    expected: str,
) -> None:
    module = _make_module(
        tmp_path,
        [],
        predict_output_layout=layout,
        predict_output_dir=str(tmp_path / output_dir) if output_dir else None,
        predict_run_name=run_name,
    )
    assert module._resolve_predict_base_dir() == tmp_path / expected  # noqa: SLF001


def test_prediction_dirs(tmp_path: Path) -> None:
    prod = _make_module(tmp_path, [])
    bench = _make_module(tmp_path, [], predict_output_layout="benchmark")
    base = tmp_path / "b"
    assert prod._prediction_dirs(base, "42", "20261008_1200", "C1") == (  # noqa: SLF001
        base / "42" / "20261008_1200" / "C1",
        base / "42" / "20261008_1200",
    )
    assert bench._prediction_dirs(base, "42", "20261008_1200", "C1") == (
        base / "42",
        base / "42",
    )


# ---------------------------------------------------------------------------
# on_predict_end — benchmark layout
# ---------------------------------------------------------------------------


def test_benchmark_layout_per_tile(tmp_path: Path) -> None:
    out = tmp_path / "bench" / "model_a"
    batch = _per_tile_batch([("C1", 0), ("C2", 1)])
    module = _make_module(
        tmp_path,
        [batch],
        predict_output_dir=str(out),
        predict_output_layout="benchmark",
        predict_run_name="model_a",
    )
    module.on_predict_end()

    files = _files(out)
    event_files = {f for f in files if f.startswith(f"{EVENT_ID}/")}
    assert event_files == {
        f"{EVENT_ID}/{_output_name('C1')}",
        f"{EVENT_ID}/{_output_name('C1').replace('.tif', '_prob.tif')}",
        f"{EVENT_ID}/{_output_name('C2')}",
        f"{EVENT_ID}/{_output_name('C2').replace('.tif', '_prob.tif')}",
        f"{EVENT_ID}/{_merged_name()}",
    }
    # no production artefacts
    assert not (out / "predictions").exists()
    assert not any("merged_all" in f for f in files)
    assert not any(
        p.is_dir() for p in (out / str(EVENT_ID)).iterdir()
    )  # no date / cell sub-folders

    # manifests
    runs = list((out / BENCHMARK_RUNS_DIRNAME).iterdir())
    assert len(runs) == 1
    run_manifest = json.loads((runs[0] / "manifest.json").read_text())
    latest = json.loads((out / BENCHMARK_LATEST_MANIFEST).read_text())
    assert run_manifest == latest
    assert latest["output_layout"] == "benchmark"
    assert latest["run_name"] == "model_a"
    assert latest["base_dir"] == str(out)
    assert {Path(e["tif_path"]).parent for e in latest["predictions"]} == {
        out / str(EVENT_ID),
    }
    assert all(Path(e["probability_tif_path"]).exists() for e in latest["predictions"])
    assert not (out / "manifest.json").exists()

    # raster content preserved
    with rio.open(out / str(EVENT_ID) / _output_name("C2")) as src:
        np.testing.assert_array_equal(
            src.read(1), batch["predictions"][1].numpy().astype(np.uint16),
        )
    # merged pair raster spans both cells
    with rio.open(out / str(EVENT_ID) / _merged_name()) as src:
        assert (src.height, src.width) == (TILE, 2 * TILE)


def test_benchmark_two_pairs_same_cells_no_collision(tmp_path: Path) -> None:
    out = tmp_path / "bench" / "model_a"
    batches = [
        _per_tile_batch([("C1", 0), ("C2", 1)], gid_post=3, date_post="2025-07-10"),
        _per_tile_batch([("C1", 0), ("C2", 1)], gid_post=4, date_post="2025-07-20"),
    ]
    module = _make_module(
        tmp_path,
        batches,
        predict_output_dir=str(out),
        predict_output_layout="benchmark",
    )
    module.on_predict_end()

    tifs = {f for f in _files(out / str(EVENT_ID)) if not f.endswith("_prob.tif")}
    assert tifs == {
        _output_name("C1", 3, "20250710"),
        _output_name("C2", 3, "20250710"),
        _output_name("C1", 4, "20250720"),
        _output_name("C2", 4, "20250720"),
        _merged_name(3, "20250710"),
        _merged_name(4, "20250720"),
    }


def test_benchmark_rerun_is_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "bench" / "model_a"
    module = _make_module(
        tmp_path,
        [_per_tile_batch([("C1", 0)])],
        predict_output_dir=str(out),
        predict_output_layout="benchmark",
    )
    module.on_predict_end()
    first = {f for f in _files(out) if not f.startswith(BENCHMARK_RUNS_DIRNAME)}
    module.on_predict_end()
    second = {f for f in _files(out) if not f.startswith(BENCHMARK_RUNS_DIRNAME)}
    assert first == second


def test_benchmark_layout_overlap_blended(tmp_path: Path) -> None:
    out = tmp_path / "bench" / "model_a"
    module = _make_module(
        tmp_path,
        [_blended_batch()],
        datamodule=SimpleNamespace(tile_size=(TILE, TILE), tile_stride=(2, 2)),
        predict_output_dir=str(out),
        predict_output_layout="benchmark",
    )
    module.on_predict_end()

    event_files = _files(out / str(EVENT_ID))
    assert event_files == {
        _output_name("C9"),
        _output_name("C9").replace(".tif", "_prob.tif"),
        _merged_name(),
    }
    with rio.open(out / str(EVENT_ID) / _output_name("C9")) as src:
        assert (src.height, src.width) == (6, 6)
        assert np.all(src.read(1) == 1)
    latest = json.loads((out / BENCHMARK_LATEST_MANIFEST).read_text())
    assert latest["overlap_blended"] is True
    assert len(latest["predictions"]) == 1


# ---------------------------------------------------------------------------
# on_predict_end — production layout (non-regression)
# ---------------------------------------------------------------------------


def test_production_layout_unchanged(tmp_path: Path) -> None:
    out = tmp_path / "prod"
    module = _make_module(
        tmp_path, [_per_tile_batch([("C1", 0), ("C2", 1)])], predict_output_dir=str(out),
    )
    module.on_predict_end()

    base = out / "predictions"
    date_dirs = list((base / str(EVENT_ID)).iterdir())
    assert len(date_dirs) == 1
    date = date_dirs[0].name
    assert len(date) == len("20261008_1200")

    rel = set(_files(base))
    assert rel == {
        "manifest.json",
        f"{EVENT_ID}/{date}/C1/{_output_name('C1')}",
        f"{EVENT_ID}/{date}/C1/{_output_name('C1').replace('.tif', '_prob.tif')}",
        f"{EVENT_ID}/{date}/C2/{_output_name('C2')}",
        f"{EVENT_ID}/{date}/C2/{_output_name('C2').replace('.tif', '_prob.tif')}",
        f"{EVENT_ID}/{date}/{_merged_name()}",
        f"{EVENT_ID}/{date}/merged_all.tif",
    }
    manifest = json.loads((base / "manifest.json").read_text())
    assert "output_layout" not in manifest
    assert "run_name" not in manifest
    assert set(manifest) == {
        "prediction_date",
        "model_name",
        "checkpoint",
        "base_dir",
        "probability_thresholds",
        "predictions",
    }
    assert not (out / BENCHMARK_RUNS_DIRNAME).exists()
    assert not (base / BENCHMARK_LATEST_MANIFEST).exists()


def test_no_predictions_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "bench"
    module = _make_module(
        tmp_path, [], predict_output_dir=str(out), predict_output_layout="benchmark",
    )
    module.on_predict_end()
    assert not out.exists()


# ---------------------------------------------------------------------------
# merge_predictions(write_merged_all=...)
# ---------------------------------------------------------------------------


def _write_tif(path: Path, x0: float) -> Path:
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": 1,
        "nodata": 32767,
        "height": TILE,
        "width": TILE,
        "crs": "EPSG:3979",
        "transform": Affine(PIXEL, 0, x0, 0, -PIXEL, 1000.0),
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(np.ones((1, TILE, TILE), dtype=np.uint16))
    return path


@pytest.mark.parametrize("write_merged_all", [True, False])
def test_merge_predictions_write_merged_all(
    tmp_path: Path, write_merged_all: bool,
) -> None:
    a = _write_tif(tmp_path / "a.tif", 0.0)
    b = _write_tif(tmp_path / "b.tif", TILE * PIXEL)
    key = (
        str(tmp_path),
        "42",
        "2025-07-01",
        "2025-07-25",
        "1",
        "2025-06-20",
        "3",
        "2025-07-10",
        "A",
        "ASC",
    )
    merge_predictions(
        {key: [a, b]}, {str(tmp_path): [a, b]}, write_merged_all=write_merged_all,
    )

    assert (tmp_path / _merged_name()).exists()
    assert (tmp_path / "merged_all.tif").exists() is write_merged_all
