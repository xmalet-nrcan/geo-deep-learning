"""Tests for ``scripts/check_benchmark_outputs.py`` (dev plan P8).

Outputs are produced by the REAL writer (``ChangeDetectionChangeFormer.on_predict_end``,
benchmark layout) so the checker is validated against the actual file layout.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("rasterio")
pytest.importorskip("lightning")
pytest.importorskip("kornia")

import numpy as np  # noqa: E402, F401
import rasterio as rio  # noqa: E402
import yaml  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

from tests.test_change_detection_predict_output_layout import (  # noqa: E402
    EVENT_ID,
    _make_module,
    _per_tile_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_benchmark_outputs", REPO_ROOT / "scripts" / "check_benchmark_outputs.py",
)
cbo = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cbo
_spec.loader.exec_module(cbo)

ERROR, WARNING, INFO = cbo.ERROR, cbo.WARNING, cbo.INFO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _db_style(batch: dict) -> dict:
    """Per-cell names carry the DB sat_pass (``Ascending``), merges the dataset one (``ASC``)."""
    batch["output_name"] = [n.replace("_pass-ASC.tif", "_pass-Ascending.tif") for n in batch["output_name"]]
    return batch


def _write_outputs(tmp_path: Path, model: str, batches: list[dict]) -> None:
    module = _make_module(
        tmp_path,
        batches,
        predict_output_dir=str(tmp_path / "out" / model),
        predict_output_layout="benchmark",
        predict_run_name=model,
    )
    module.on_predict_end()


@pytest.fixture
def bench(tmp_path: Path) -> dict:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    for name, beams in (("m1", ["A"]), ("m2", ["A", "B"])):
        (cfg_dir / f"{name}_predict.yaml").write_text(
            yaml.safe_dump({
                "model": {"init_args": {"num_classes": 1}},
                "data": {"init_args": {"beams": beams}},
            }),
            encoding="utf-8",
        )
    models_file = cfg_dir / "models.yaml"
    models_file.write_text(
        yaml.safe_dump({
            "output_root": str(tmp_path / "out"),
            "models": [{"name": "m1", "config": "m1_predict.yaml"},
                       {"name": "m2", "config": "m2_predict.yaml"}],
        }),
        encoding="utf-8",
    )
    for model in ("m1", "m2"):
        _write_outputs(tmp_path, model, [_db_style(_per_tile_batch([("C1", 0), ("C2", 1)]))])
    return {"tmp": tmp_path, "models_file": models_file, "out": tmp_path / "out",
            "event_dir": tmp_path / "out" / "m1" / str(EVENT_ID)}


def _run(bench: dict, **kwargs) -> tuple[list, list]:
    kwargs.setdefault("csv_dir", None)
    return cbo.run_checks(bench["models_file"], **kwargs)


def _levels(findings, level: str) -> list[str]:
    return [str(f) for f in findings if f.level == level]


def _cell_name(event_dir: Path, cell: str = "C1") -> Path:
    return next(p for p in event_dir.iterdir() if f"_cell-{cell}_" in p.name and not p.name.endswith("_prob.tif"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_constants_match_writer() -> None:
    from geo_deep_learning.datasets.rcm_change_detection_dataset import NO_DATA  # noqa: PLC0415
    from geo_deep_learning.tasks_with_models.change_detection_changeformer import (  # noqa: PLC0415
        BENCHMARK_LATEST_MANIFEST,
        BENCHMARK_RUNS_DIRNAME,
        PROBABILITY_NODATA,
    )

    assert cbo.CLASS_NODATA == NO_DATA
    assert cbo.PROB_NODATA == PROBABILITY_NODATA
    assert cbo.RUNS_DIRNAME == BENCHMARK_RUNS_DIRNAME
    assert cbo.LATEST_MANIFEST == BENCHMARK_LATEST_MANIFEST


@pytest.mark.parametrize(
    ("name", "cell", "beam", "sat_pass", "prob"),
    [
        ("event-42_start-20250701_end_20250725_pre-g1-20250620_post-g3-20250710_cell-C1_beam-A_pass-ASC.tif",
         "C1", "A", "ASC", False),
        ("event-42_start-20250701_end_NA_pre-g1-20250620_post-g3-20250710_cell-12_34_beam-B_pass-Ascending_prob.tif",
         "12_34", "B", "Ascending", True),
        ("event-42_start-20250701_end_20250725_pre-g1-20250620_post-g3-20250710_beam-A_pass-DESC.tif",
         None, "A", "DESC", False),
    ],
)
def test_parse_name(name, cell, beam, sat_pass, prob) -> None:
    out = cbo.parse_name("42", name)
    assert (out.event, out.cell, out.beam, out.sat_pass, out.is_prob) == ("42", cell, beam, sat_pass, prob)
    assert cbo.parse_name("42", "merged_all.tif") is None


def test_real_writer_outputs_are_valid(bench) -> None:
    findings, outputs = _run(bench)
    assert _levels(findings, ERROR) == []
    assert _levels(findings, WARNING) == []
    for m in outputs:
        kinds = sorted(o.kind for o in m.files.values())
        assert kinds == ["cell", "cell", "merged", "prob", "prob"]
    summary = cbo.summary_lines(outputs, findings)
    assert summary[1].split()[1:5] == ["1", "2", "2", "1"]


def test_layout_errors(bench) -> None:
    ev = bench["event_dir"]
    (ev / "merged_all.tif").write_bytes(b"x")
    (bench["out"] / "m1" / "predictions").mkdir()
    (ev / "sub").mkdir()
    findings, _ = _run(bench, rasters=False)
    errors = "\n".join(_levels(findings, ERROR))
    assert "merged_all.tif must not exist" in errors
    assert "production layout folder" in errors
    assert "unexpected sub-folder" in errors


def test_missing_prob_and_merged(bench) -> None:
    ev = bench["event_dir"]
    cell = _cell_name(ev)
    Path(str(cell).replace(".tif", "_prob.tif")).unlink()
    for merged in [p for p in ev.iterdir() if "_cell-" not in p.name]:
        merged.unlink()
    findings, _ = _run(bench, names=["m1"], rasters=False)
    errors = "\n".join(_levels(findings, ERROR))
    assert "probability raster missing" in errors
    assert errors.count("per-pair merged raster") == 2


def test_orphan_prob_and_stale_merged(bench) -> None:
    ev = bench["event_dir"]
    for p in [p for p in ev.iterdir() if "_cell-C2_" in p.name and not p.name.endswith("_prob.tif")]:
        p.unlink()
    for p in [p for p in ev.iterdir() if "_cell-" in p.name and not p.name.endswith("_prob.tif")]:
        p.unlink()  # C1 class too → merged without any per-cell class raster
    findings, _ = _run(bench, names=["m1"], rasters=False)
    assert "class raster missing" in "\n".join(_levels(findings, ERROR))
    assert "merged raster without any per-cell raster" in "\n".join(_levels(findings, WARNING))


def test_beam_filter(bench) -> None:
    # m2 (beams A+B) also predicted beam B; m1 (A only) did not → explained difference
    batch_b = _per_tile_batch([("C1", 0)], gid_post=5)
    batch_b["beam"] = ["B"]
    batch_b["output_name"] = [n.replace("_beam-A_", "_beam-B_") for n in batch_b["output_name"]]
    _write_outputs(bench["tmp"], "m2", [_db_style(_per_tile_batch([("C1", 0), ("C2", 1)])), _db_style(batch_b)])
    findings, _ = _run(bench)
    assert _levels(findings, ERROR) == []
    assert _levels(findings, WARNING) == []
    assert any("excluded by the beam/pass filters" in f for f in _levels(findings, INFO))

    # the same beam-B output in m1 violates its filter
    _write_outputs(bench["tmp"], "m1", [_db_style(_per_tile_batch([("C1", 0), ("C2", 1)])), _db_style(batch_b)])
    findings, _ = _run(bench)
    assert any("excluded by the model config" in e for e in _levels(findings, ERROR))


def test_unexplained_cross_model_difference(bench) -> None:
    ev = bench["event_dir"]
    for p in [p for p in ev.iterdir() if "_cell-C2_" in p.name]:
        p.unlink()
    findings, _ = _run(bench, rasters=False)
    warnings = _levels(findings, WARNING)
    assert len(warnings) == 2  # class + prob of C2 present in m2 only
    assert all("m1 / cross-model" in w and "_cell-C2_" in w for w in warnings)


def test_alignment_mismatch_between_models(bench) -> None:
    path = _cell_name(bench["out"] / "m2" / str(EVENT_ID))
    with rio.open(path) as src:
        profile, data = src.profile, src.read()
    profile["transform"] = profile["transform"] * Affine.translation(1, 0)
    with rio.open(path, "w", **profile) as dst:
        dst.write(data)
    findings, _ = _run(bench)
    errors = _levels(findings, ERROR)
    assert any("alignment" in e and path.name in e for e in errors)
    # class / prob of m2 C1 no longer on the same grid either
    assert any("not on the same grid" in e for e in errors)


def test_raster_content_errors(bench) -> None:
    path = _cell_name(bench["event_dir"])
    with rio.open(path) as src:
        profile, data = src.profile, src.read()
    profile["nodata"] = 0
    data[0, 0, 0] = 7
    with rio.open(path, "w", **profile) as dst:
        dst.write(data)
    prob = Path(str(path).replace(".tif", "_prob.tif"))
    with rio.open(prob) as src:
        p_profile, p_data = src.profile, src.read()
    p_data[0, 0, 0] = 1.5
    with rio.open(prob, "w", **p_profile) as dst:
        dst.write(p_data)
    findings, _ = _run(bench, names=["m1"])
    errors = "\n".join(_levels(findings, ERROR))
    assert "dtype/nodata uint16/0.0" in errors
    assert "unexpected class value(s) [7]" in errors
    assert "probabilities outside [0, 1]" in errors


def test_expected_outputs_from_orchestrator_csv(bench) -> None:
    csv_dir = bench["tmp"] / "csv"
    ev = bench["event_dir"]
    produced = sorted(p.name for p in ev.iterdir() if "_cell-" in p.name and not p.name.endswith("_prob.tif"))
    missing = produced[0].replace("_cell-C1_", "_cell-C9_")
    rows = [*({"event_id": EVENT_ID, "beam": "A", "sat_pass": "Ascending", "output_name": n} for n in produced),
            {"event_id": EVENT_ID, "beam": "A", "sat_pass": "Ascending", "output_name": missing},
            {"event_id": EVENT_ID, "beam": "B", "sat_pass": "Ascending", "output_name": "beam_b_filtered.tif"}]
    for model in ("m1", "m2"):
        (csv_dir / model).mkdir(parents=True)
        with (csv_dir / model / cbo.CSV_FILE_NAME).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    findings, _ = _run(bench, csv_dir=str(csv_dir), rasters=False)
    errors = _levels(findings, ERROR)
    assert len(errors) == 3  # C9 for m1 + m2 (m2 also lacks the beam-B row) → 3
    assert sum("_cell-C9_" in e for e in errors) == 2
    assert any("m2" in e and "beam_b_filtered.tif" in e for e in errors)
    infos = [f for f in _levels(findings, INFO) if "m1 / expected" in f]
    assert "3 exported pair(s) checked, 1 excluded" in infos[0]


def test_event_and_model_filters(bench) -> None:
    findings, outputs = _run(bench, names=["m2"], events=["999"], rasters=False)
    assert [m.name for m in outputs] == ["m2"]
    assert "no prediction raster found" in "\n".join(_levels(findings, ERROR))


def test_main_exit_code(bench, capsys) -> None:
    args = ["--models-file", str(bench["models_file"]), "--csv-dir", ""]
    assert cbo.main(args) == 0
    (bench["event_dir"] / "merged_all.tif").write_bytes(b"x")
    assert cbo.main([*args, "--quiet", "--no-rasters"]) == 1
    assert cbo.main([*args, "--model", "nope"]) == 1


def test_missing_model_folder(bench) -> None:
    findings, _ = _run(bench, output_root=str(bench["tmp"] / "elsewhere"), rasters=False)
    assert any("output folder not found" in e for e in _levels(findings, ERROR))
