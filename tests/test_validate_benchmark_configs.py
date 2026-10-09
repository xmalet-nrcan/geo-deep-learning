"""Tests for ``scripts/validate_benchmark_configs.py`` (dev plan P5)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validate_benchmark_configs",
    REPO_ROOT / "scripts" / "validate_benchmark_configs.py",
)
vbc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vbc  # dataclasses need the module registered
_spec.loader.exec_module(vbc)

BANDS = ["LOCALINCANGLE", "PDN", "PSN", "PVN", "S0", "NDSV", "RFDI", "RR", "RL"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(**model_overrides: object) -> dict:
    model_args = {
        "change_detection_model": "changestar2",
        "image_size": [256, 256],
        "max_samples": 10,
        "num_classes": 1,
        "use_metadata_film": True,
        "film_embed_dim": 8,
        "use_cbam": True,
        "use_signed_difference": True,
        "signed_difference_normalize": True,
        "weights_from_checkpoint_path": "/app/models_checkpoints/model.ckpt",
    }
    model_args.update(model_overrides)
    return {
        "model": {
            "class_path": "tasks_with_models.change_detection_changeformer.ChangeDetectionChangeFormer",
            "init_args": model_args,
        },
        "data": {
            "class_path": "datamodules.rcm_change_detection_datamodule.RcmChangeDetectionDataModule",
            "init_args": {
                "csv_root_folder": vbc.OVERRIDE_PLACEHOLDER,
                "csv_file_name": vbc.OVERRIDE_PLACEHOLDER,
                "patches_root_folder": "/data",
                "separate_metadata": True,
                "band_names": BANDS,
            },
        },
    }


def _write(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _registry(
    tmp_path: Path, models: list[dict], output_root: str | None = "/out",
) -> Path:
    data: dict = {"models": models}
    if output_root is not None:
        data["output_root"] = output_root
    return _write(tmp_path / "models.yaml", data)


def _model(name: str = "m1", config: Path | None = None) -> vbc.BenchmarkModel:
    return vbc.BenchmarkModel(name=name, config=config or Path("unused.yaml"))


def _errors(issues: list) -> list:
    return [i for i in issues if i.level == vbc.ERROR]


def _warnings(issues: list) -> list:
    return [i for i in issues if i.level == vbc.WARNING]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_load_registry_ok_and_relative_config(tmp_path: Path) -> None:
    _write(tmp_path / "a_predict.yaml", _config())
    path = _registry(
        tmp_path,
        [
            {"name": "model_a.v1", "config": "a_predict.yaml"},
            {"name": "model-b", "config": str(tmp_path / "b.yaml"), "enabled": False},
        ],
    )
    registry, issues = vbc.load_registry(path)
    assert issues == []
    assert registry.output_root == "/out"
    assert [m.name for m in registry.models] == ["model_a.v1", "model-b"]
    assert registry.models[0].config == (tmp_path / "a_predict.yaml").resolve()
    assert registry.models[1].enabled is False


@pytest.mark.parametrize(
    ("models", "fragment"),
    [
        ([{"name": "bad name", "config": "x.yaml"}], "invalid name"),
        ([{"name": "a/b", "config": "x.yaml"}], "invalid name"),
        (
            [{"name": "a", "config": "x.yaml"}, {"name": "a", "config": "y.yaml"}],
            "duplicate",
        ),
        ([{"name": "a"}], "'config' is required"),
        (
            [{"name": "a", "config": "x.yaml", "enabled": "yes"}],
            "'enabled' must be a boolean",
        ),
        ([], "non-empty list"),
    ],
)
def test_load_registry_errors(tmp_path: Path, models: list, fragment: str) -> None:
    _, issues = vbc.load_registry(_registry(tmp_path, models))
    assert any(fragment in i.message for i in _errors(issues)), issues


def test_load_registry_missing_output_root(tmp_path: Path) -> None:
    _, issues = vbc.load_registry(
        _registry(tmp_path, [{"name": "a", "config": "x"}], output_root=None),
    )
    assert any("output_root" in i.message for i in _errors(issues))


def test_load_registry_missing_file(tmp_path: Path) -> None:
    _, issues = vbc.load_registry(tmp_path / "nope.yaml")
    assert _errors(issues)


def test_select_models(tmp_path: Path) -> None:
    registry, _ = vbc.load_registry(
        _registry(
            tmp_path,
            [
                {"name": "on", "config": "a"},
                {"name": "off", "config": "b", "enabled": False},
            ],
        ),
    )
    assert [m.name for m in vbc.select_models(registry, None)[0]] == ["on"]
    assert [
        m.name for m in vbc.select_models(registry, None, include_disabled=True)[0]
    ] == ["on", "off"]
    assert [m.name for m in vbc.select_models(registry, ["off"])[0]] == ["off"]
    _, issues = vbc.select_models(registry, ["ghost"])
    assert any("not in registry" in i.message for i in _errors(issues))


# ---------------------------------------------------------------------------
# Static config
# ---------------------------------------------------------------------------


def test_check_static_ok() -> None:
    assert vbc.check_static(_model(), _config()) == []


def test_check_static_wrong_task() -> None:
    cfg = _config()
    cfg["model"]["class_path"] = (
        "tasks_with_models.change_detection_segformer.ChangeDetectionSegmentationSegformer"
    )
    assert any(
        "class_path" in i.message for i in _errors(vbc.check_static(_model(), cfg))
    )


def test_check_static_film_requires_separate_metadata() -> None:
    cfg = _config()
    cfg["data"]["init_args"]["separate_metadata"] = False
    assert any(
        "separate_metadata" in i.message
        for i in _errors(vbc.check_static(_model(), cfg))
    )


def test_check_static_missing_band_names_and_weights() -> None:
    cfg = _config(weights_from_checkpoint_path=None)
    del cfg["data"]["init_args"]["band_names"]
    messages = " ".join(i.message for i in _errors(vbc.check_static(_model(), cfg)))
    assert "band_names" in messages
    assert "weights_from_checkpoint_path" in messages


@pytest.mark.parametrize(("in_channels", "ok"), [(10, True), (9, False)])
def test_check_static_explicit_in_channels(in_channels: int, ok: bool) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("rasterio")
    issues = vbc.check_static(_model(), _config(in_channels=in_channels))
    assert (not _errors(issues)) is ok


def test_check_static_warns_on_overridden_keys() -> None:
    cfg = _config(predict_output_dir="/somewhere", predict_output_layout="production")
    cfg["data"]["init_args"]["csv_file_name"] = "prod.csv"
    warnings = _warnings(vbc.check_static(_model(), cfg))
    assert {w.message.split(" ")[0] for w in warnings} == {
        "model.init_args.predict_output_dir",
        "model.init_args.predict_output_layout",
        "data.init_args.csv_file_name='prod.csv'",
    }


# ---------------------------------------------------------------------------
# Checkpoint hyper-parameters
# ---------------------------------------------------------------------------


@pytest.fixture
def defaults() -> dict:
    pytest.importorskip("torch")
    pytest.importorskip("rasterio")
    pytest.importorskip("lightning")
    pytest.importorskip("kornia")
    return vbc.task_defaults()


def _ckpt_hparams(**overrides: object) -> dict:
    hp = {
        "change_detection_model": "changestar2",
        "num_classes": 1,
        "use_metadata_film": True,
        "film_embed_dim": 8,
        "use_cbam": True,
        "cbam_reduction": 4,
        "use_dfa": False,
        "dfa_gate_hidden": 16,
        "use_signed_difference": True,
        "signed_difference_channels": None,
        "signed_difference_normalize": True,
        "backbone_kwargs": None,
        "image_size": (256, 256),
        "in_channels": 10,
    }
    hp.update(overrides)
    return hp


def test_compare_hparams_match(defaults: dict) -> None:
    cfg = _config()
    issues = vbc.compare_hparams(
        "m",
        cfg["model"]["init_args"],
        cfg["data"]["init_args"],
        _ckpt_hparams(),
        defaults,
        expected_in_channels=10,
    )
    assert issues == []


@pytest.mark.parametrize(
    ("ckpt_override", "fragment"),
    [
        ({"use_cbam": False}, "use_cbam"),
        ({"film_embed_dim": 32}, "film_embed_dim"),
        ({"change_detection_model": "changeformer_7"}, "change_detection_model"),
        ({"signed_difference_normalize": False}, "signed_difference_normalize"),
        ({"backbone_kwargs": {"encoder": "mit_b2"}}, "backbone_kwargs"),
        ({"in_channels": 8}, "in_channels"),
    ],
)
def test_compare_hparams_architecture_mismatch(
    defaults: dict, ckpt_override: dict, fragment: str,
) -> None:
    cfg = _config()
    issues = vbc.compare_hparams(
        "m",
        cfg["model"]["init_args"],
        cfg["data"]["init_args"],
        _ckpt_hparams(**ckpt_override),
        defaults,
        expected_in_channels=10,
    )
    assert any(i.message.startswith(fragment) for i in _errors(issues)), issues


def test_compare_hparams_soft_mismatch_is_warning(defaults: dict) -> None:
    cfg = _config()
    issues = vbc.compare_hparams(
        "m",
        cfg["model"]["init_args"],
        cfg["data"]["init_args"],
        _ckpt_hparams(image_size=[512, 512]),
        defaults,
        expected_in_channels=10,
    )
    assert not _errors(issues)
    assert any("image_size" in w.message for w in _warnings(issues))


def test_compare_hparams_default_used_when_key_absent_from_config(
    defaults: dict,
) -> None:
    cfg = _config()
    del cfg["model"]["init_args"]["use_cbam"]  # default False
    issues = vbc.compare_hparams(
        "m",
        cfg["model"]["init_args"],
        cfg["data"]["init_args"],
        _ckpt_hparams(use_cbam=True),
        defaults,
        expected_in_channels=10,
    )
    assert any("use_cbam" in i.message for i in _errors(issues))


def test_compare_hparams_old_checkpoint_without_key(defaults: dict) -> None:
    cfg = _config()
    hp = _ckpt_hparams()
    del hp["use_signed_difference"]
    issues = vbc.compare_hparams(
        "m",
        cfg["model"]["init_args"],
        cfg["data"]["init_args"],
        hp,
        defaults,
        expected_in_channels=10,
    )
    assert not _errors(issues)
    assert any("use_signed_difference" in w.message for w in _warnings(issues))


def test_check_checkpoint_end_to_end(tmp_path: Path, defaults: dict) -> None:  # noqa: ARG001
    import torch

    ckpt_path = tmp_path / "ckpts" / "model.ckpt"
    ckpt_path.parent.mkdir()
    torch.save(
        {"hyper_parameters": _ckpt_hparams(use_cbam=False), "state_dict": {}}, ckpt_path,
    )
    path_maps = vbc.parse_path_maps([f"/app/models_checkpoints={ckpt_path.parent}"])

    issues, ckpt = vbc.check_checkpoint(_model(), _config(), path_maps)
    assert ckpt is not None
    assert [i.message.split(":")[0] for i in _errors(issues)] == ["use_cbam"]


def test_check_checkpoint_missing_file(tmp_path: Path) -> None:
    issues, ckpt = vbc.check_checkpoint(
        _model(), _config(weights_from_checkpoint_path=str(tmp_path / "x.ckpt")),
    )
    assert ckpt is None
    assert any("file not found" in i.message for i in _errors(issues))


# ---------------------------------------------------------------------------
# Paths / override
# ---------------------------------------------------------------------------


def test_map_path() -> None:
    maps = vbc.parse_path_maps(["/app/models_checkpoints=D:/ckpt", "/mnt/geo/=/data"])
    assert vbc.map_path("/app/models_checkpoints/a/b.ckpt", maps) == Path(
        "D:/ckpt/a/b.ckpt",
    )
    assert vbc.map_path("/mnt/geo/x.tif", maps) == Path("/data/x.tif")
    assert vbc.map_path("/app/models_checkpoints_old/a.ckpt", maps) == Path(
        "/app/models_checkpoints_old/a.ckpt",
    )
    with pytest.raises(Exception, match="SRC=DST"):
        vbc.parse_path_maps(["nope"])
    # container checkpoint names contain "=" → split on the last one
    ckpt = "/app/m/x-epoch=24-val_loss=0.484.ckpt"
    assert vbc.map_path(ckpt, vbc.parse_path_maps([f"{ckpt}=D:/local.ckpt"])) == Path(
        "D:/local.ckpt",
    )


def test_build_override() -> None:
    ovr = vbc.build_override("m1", "/out/", "/csv")
    assert ovr["trainer"] == {"devices": 1, "callbacks": [], "logger": False}
    assert ovr["model"]["init_args"] == {
        "predict_output_dir": "/out/m1",
        "predict_output_layout": "benchmark",
        "predict_run_name": "m1",
        "predict_write_merged_all": False,
        "weights_strict": True,
    }
    assert ovr["data"]["init_args"] == {
        "dataset_class": vbc.PREDICT_DATASET_CLASS,
        "csv_root_folder": "/csv/m1",
        "csv_file_name": "vw_input_files_for_model_test.csv",
    }
    # never touches the per-model filters (decision 5)
    assert not {"beams", "satellite_pass", "dataset_years", "band_names"} & set(
        ovr["data"]["init_args"],
    )


# ---------------------------------------------------------------------------
# Sync with the scanfire orchestrator (sibling repository, skipped if absent)
# ---------------------------------------------------------------------------

SCANFIRE_BENCHMARK_DIR = (
    REPO_ROOT.parent / "scanfire" / "scanfire_modules" / "pipeline" / "orchestrator" / "benchmark"
)


def _load_scanfire(name: str):
    path = SCANFIRE_BENCHMARK_DIR / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"scanfire repository not available ({path})")
    spec = importlib.util.spec_from_file_location(f"scanfire_benchmark_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("name", "root", "csv_dir"),
    [("m1", "/out", "/csv"), ("cs2.base-x_1", "/mnt/a/b/", "/mnt/c/")],
)
def test_override_in_sync_with_scanfire(name: str, root: str, csv_dir: str) -> None:
    """The override validated here is exactly the one the orchestrator writes."""
    scanfire_override = _load_scanfire("override")
    assert scanfire_override.build_override(name, root, csv_dir) == vbc.build_override(
        name, root, csv_dir,
    )
    assert scanfire_override.PREDICT_DATASET_CLASS == vbc.PREDICT_DATASET_CLASS
    assert scanfire_override.BENCHMARK_CSV_FILE_NAME == vbc.BENCHMARK_CSV_FILE_NAME


def test_registry_in_sync_with_scanfire() -> None:
    """Both registry loaders read the real models.yaml identically."""
    scanfire_registry = _load_scanfire("registry")
    ours, issues = vbc.load_registry(vbc.DEFAULT_MODELS_FILE)
    assert not [i for i in issues if i.level == vbc.ERROR]
    theirs = scanfire_registry.load_registry(vbc.DEFAULT_MODELS_FILE)
    assert theirs.output_root == ours.output_root
    assert [(m.name, m.config, m.enabled) for m in theirs.models] == [
        (m.name, m.config, m.enabled) for m in ours.models
    ]
    assert scanfire_registry.NAME_PATTERN.pattern == vbc.NAME_PATTERN.pattern


# ---------------------------------------------------------------------------
# Repository registry (guards future edits of configs/benchmark/)
# ---------------------------------------------------------------------------


def test_repository_registry_is_valid() -> None:
    issues = vbc.validate(vbc.DEFAULT_MODELS_FILE, skip_checkpoint=True)
    assert _errors(issues) == []
    registry, _ = vbc.load_registry(vbc.DEFAULT_MODELS_FILE)
    assert all(m.config.is_file() for m in registry.models)


@pytest.mark.skipif(
    os.environ.get("GDL_RUN_CLI_TESTS") != "1", reason="slow: set GDL_RUN_CLI_TESTS=1",
)
def test_repository_registry_cli() -> None:
    issues = vbc.validate(vbc.DEFAULT_MODELS_FILE, skip_checkpoint=True, cli=True)
    assert _errors(issues) == []


@pytest.mark.skipif(
    os.environ.get("GDL_RUN_CLI_TESTS") != "1", reason="slow: set GDL_RUN_CLI_TESTS=1",
)
def test_check_load_weights_detects_architecture_mismatch(defaults: dict) -> None:  # noqa: ARG001
    from geo_deep_learning.tasks_with_models.change_detection_changeformer import (
        ChangeDetectionChangeFormer,
    )

    registry, _ = vbc.load_registry(vbc.DEFAULT_MODELS_FILE)
    model = registry.models[0]
    cfg = yaml.safe_load(model.config.read_text(encoding="utf-8"))
    skip = {
        "main_loss",
        "secondary_loss",
        "optimizer",
        "scheduler",
        "weights_from_checkpoint_path",
    }
    args = {k: v for k, v in cfg["model"]["init_args"].items() if k not in skip}
    task = ChangeDetectionChangeFormer(
        main_loss=None, secondary_loss=None, in_channels=10, **args,
    )
    task.configure_model()
    ckpt = {
        "hyper_parameters": dict(task.hparams),
        "state_dict": {f"model.{k}": v for k, v in task.model.state_dict().items()},
    }

    assert vbc.check_load_weights(model, cfg, ckpt) == []

    no_cbam = yaml.safe_load(model.config.read_text(encoding="utf-8"))
    no_cbam["model"]["init_args"]["use_cbam"] = False
    assert any(
        "unexpected key" in i.message
        for i in vbc.check_load_weights(model, no_cbam, ckpt)
    )

    three_bands = yaml.safe_load(model.config.read_text(encoding="utf-8"))
    three_bands["data"]["init_args"]["band_names"] = ["S0", "RR", "RL"]
    assert any(
        "size mismatch" in i.message
        for i in vbc.check_load_weights(model, three_bands, ckpt)
    )


def test_main_exit_code(tmp_path: Path) -> None:
    assert vbc.main(["--models-file", str(tmp_path / "missing.yaml")]) == 1
    cfg = _write(tmp_path / "a.yaml", _config())
    reg = _registry(tmp_path, [{"name": "a", "config": str(cfg)}])
    assert vbc.main(["--models-file", str(reg), "--skip-checkpoint"]) == 0
