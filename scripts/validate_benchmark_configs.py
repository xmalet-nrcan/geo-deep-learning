#!/usr/bin/env python
"""
Validate the model benchmark registry and per-model predict configs (dev plan P5).

Checks, per enabled model of ``configs/benchmark/models.yaml``:

1. **Registry** — ``output_root`` set, ``name`` matches ``^[A-Za-z0-9_.-]+$`` and is
   unique, ``config`` exists, ``enabled`` is a boolean.
2. **Static config** — ``ChangeDetectionChangeFormer`` task, ``band_names`` set,
   ``use_metadata_film`` ⇒ ``separate_metadata``, explicit ``in_channels`` consistent
   with ``band_names``, ``weights_from_checkpoint_path`` set, keys injected by the
   benchmark override not relied upon.
3. **Checkpoint** (skip: ``--skip-checkpoint``) — file exists; architecture
   hyper-parameters saved in the checkpoint match the config (production loads
   weights with ``strict=False``: a mismatch would only log a warning and leave
   layers randomly initialised); ``in_channels`` of the checkpoint matches the
   config ``band_names`` / ``separate_metadata``.
4. ``--cli`` — ``train.py predict --config <model> --config <override> --print_config``
   (LightningCLI parsing of the exact command run by the orchestrator).
5. ``--load-weights`` — build the network on CPU and load the checkpoint with
   ``strict=True`` (may download backbone pretrained weights).

Exit code: ``0`` if no ERROR, ``1`` otherwise.

Examples::

    python scripts/validate_benchmark_configs.py
    python scripts/validate_benchmark_configs.py --model cs2base_cosine_restarts_e24 --cli --load-weights
    # On a workstation, map container paths to local ones:
    python scripts/validate_benchmark_configs.py --path-map /app/models_checkpoints=D:/checkpoints
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "geo_deep_learning"
DEFAULT_MODELS_FILE = REPO_ROOT / "configs" / "benchmark" / "models.yaml"
DEFAULT_CSV_DIR = "/mnt/geospatial/projet_RCM_scanfire/benchmark_outputs/_csv"

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
TASK_CLASS_NAME = "ChangeDetectionChangeFormer"
PREDICT_DATASET_CLASS = (
    "datasets.rcm_change_detection_predict_dataset.RCMChangeDetectionOnPredictDataset"
)
BENCHMARK_CSV_FILE_NAME = "vw_input_files_for_model_test.csv"

#: init_args injected by the benchmark override (see :func:`build_override`).
OVERRIDDEN_MODEL_ARGS = (
    "predict_output_dir",
    "predict_output_layout",
    "predict_run_name",
    "predict_write_merged_all",
    "weights_strict",
)
OVERRIDDEN_DATA_ARGS = ("dataset_class", "csv_root_folder", "csv_file_name")
#: Value used in model configs for required data args set by the override.
OVERRIDE_PLACEHOLDER = "__set_by_benchmark_override__"

#: Model init_args that change the network (weights shape / forward semantics):
#: must be identical between the checkpoint (training) and the predict config.
ARCHITECTURE_HPARAMS = (
    "change_detection_model",
    "num_classes",
    "use_metadata_film",
    "film_embed_dim",
    "use_cbam",
    "cbam_reduction",
    "use_dfa",
    "dfa_gate_hidden",
    "use_signed_difference",
    "signed_difference_channels",
    "signed_difference_normalize",
    "backbone_kwargs",
)
#: Inference-relevant init_args: a mismatch is suspicious but not fatal.
SOFT_HPARAMS = ("image_size",)

ERROR = "ERROR"
WARNING = "WARNING"

logger = logging.getLogger("validate_benchmark_configs")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Issue:
    """One validation finding (``level`` = ERROR | WARNING)."""

    level: str
    model: str
    check: str
    message: str

    def __str__(self) -> str:
        """Human-readable one-liner."""
        return f"[{self.level}] {self.model} / {self.check}: {self.message}"


@dataclass
class BenchmarkModel:
    """One ``models[]`` entry of the registry (``config`` resolved to a path)."""

    name: str
    config: Path
    enabled: bool = True
    description: str = ""


@dataclass
class Registry:
    """Parsed ``models.yaml``."""

    path: Path
    output_root: str | None
    models: list[BenchmarkModel] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def parse_path_maps(values: Iterable[str]) -> list[tuple[str, str]]:
    """
    Parse ``SRC=DST`` prefix mappings (``--path-map``).

    Split on the **last** ``=``: container paths may contain ``=`` (checkpoint
    names such as ``epoch=24-val_loss=0.484.ckpt``); ``DST`` must not.
    """
    maps = []
    for value in values:
        if "=" not in value:
            msg = f"--path-map expects SRC=DST, got {value!r}"
            raise argparse.ArgumentTypeError(msg)
        src, dst = value.rsplit("=", 1)
        maps.append((src.rstrip("/\\"), dst.rstrip("/\\")))
    return maps


def map_path(path: str | Path, path_maps: Sequence[tuple[str, str]] = ()) -> Path:
    """Translate a container path (e.g. ``/app/...``) to a local one via prefix maps."""
    text = str(path).replace("\\", "/")
    for src, dst in path_maps:
        src_n = src.replace("\\", "/")
        if text == src_n or text.startswith(src_n + "/"):
            return Path(dst + text[len(src_n) :])
    return Path(path)


# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------


def load_registry(models_file: Path) -> tuple[Registry, list[Issue]]:
    """
    Load and validate ``models.yaml``; relative ``config`` paths are resolved
    against the registry directory.
    """
    issues: list[Issue] = []
    registry = Registry(path=models_file, output_root=None)

    def err(model: str, msg: str) -> None:
        issues.append(Issue(ERROR, model, "registry", msg))

    if not models_file.is_file():
        err("<registry>", f"file not found: {models_file}")
        return registry, issues
    try:
        raw = yaml.safe_load(models_file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        err("<registry>", f"invalid YAML: {exc}")
        return registry, issues
    if not isinstance(raw, dict):
        err("<registry>", "top level must be a mapping with 'output_root' and 'models'")
        return registry, issues

    output_root = raw.get("output_root")
    if not isinstance(output_root, str) or not output_root.strip():
        err("<registry>", "'output_root' must be a non-empty string")
    else:
        registry.output_root = output_root

    entries = raw.get("models")
    if not isinstance(entries, list) or not entries:
        err("<registry>", "'models' must be a non-empty list")
        return registry, issues

    seen: set[str] = set()
    for idx, entry in enumerate(entries):
        label = f"models[{idx}]"
        if not isinstance(entry, dict):
            err(label, "entry must be a mapping")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
            err(label, f"invalid name {name!r} (expected {NAME_PATTERN.pattern})")
            continue
        if name in seen:
            err(name, "duplicate name")
            continue
        seen.add(name)

        unknown = set(entry) - {"name", "config", "enabled", "description"}
        if unknown:
            issues.append(
                Issue(
                    WARNING,
                    name,
                    "registry",
                    f"unknown key(s) ignored: {sorted(unknown)}",
                ),
            )

        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            err(name, f"'enabled' must be a boolean, got {enabled!r}")
            continue

        config = entry.get("config")
        if not isinstance(config, str) or not config.strip():
            err(name, "'config' is required")
            continue
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = (models_file.parent / config_path).resolve()

        registry.models.append(
            BenchmarkModel(
                name=name,
                config=config_path,
                enabled=enabled,
                description=str(entry.get("description") or ""),
            ),
        )
    return registry, issues


def select_models(
    registry: Registry, names: Sequence[str] | None, *, include_disabled: bool = False,
) -> tuple[list[BenchmarkModel], list[Issue]]:
    issues: list[Issue] = []
    models = registry.models
    if names:
        known = {m.name for m in models}
        issues.extend(
            Issue(ERROR, n, "registry", "not in registry")
            for n in names
            if n not in known
        )
        models = [m for m in models if m.name in names]
    elif not include_disabled:
        models = [m for m in models if m.enabled]
    if not models and not issues:
        issues.append(Issue(WARNING, "<registry>", "registry", "no model selected"))
    return models, issues


# ---------------------------------------------------------------------------
# 2. Static config checks
# ---------------------------------------------------------------------------


def load_config(
    model: BenchmarkModel, path_maps: Sequence[tuple[str, str]] = (),
) -> tuple[dict | None, list[Issue]]:
    path = map_path(model.config, path_maps)
    if not path.is_file():
        return None, [Issue(ERROR, model.name, "config", f"file not found: {path}")]
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return None, [Issue(ERROR, model.name, "config", f"invalid YAML: {exc}")]
    if not isinstance(cfg, dict):
        return None, [Issue(ERROR, model.name, "config", "top level must be a mapping")]
    return cfg, []


def _init_args(cfg: dict, section: str) -> dict:
    node = cfg.get(section) or {}
    return dict(node.get("init_args") or {}) if isinstance(node, dict) else {}


def _num_input_channels(data_args: dict) -> int:
    """Channels per image tensor — single source of truth: the dataset class."""
    sys.path[:0] = [p for p in (str(REPO_ROOT), str(PACKAGE_ROOT)) if p not in sys.path]
    from geo_deep_learning.datasets.rcm_change_detection_predict_dataset import (  # noqa: PLC0415
        RCMChangeDetectionOnPredictDataset,
    )

    return RCMChangeDetectionOnPredictDataset.num_input_channels(
        bands=data_args.get("bands"),
        band_names=data_args.get("band_names"),
        separate_metadata=data_args.get("separate_metadata", True),
    )


def check_static(model: BenchmarkModel, cfg: dict) -> list[Issue]:
    issues: list[Issue] = []

    def add(level: str, msg: str) -> None:
        issues.append(Issue(level, model.name, "config", msg))

    model_node = cfg.get("model") or {}
    class_path = (
        model_node.get("class_path", "") if isinstance(model_node, dict) else ""
    )
    if not str(class_path).endswith(TASK_CLASS_NAME):
        add(
            ERROR,
            f"model.class_path must be {TASK_CLASS_NAME} (benchmark layout), got {class_path!r}",
        )

    model_args = _init_args(cfg, "model")
    data_args = _init_args(cfg, "data")

    for key in OVERRIDDEN_MODEL_ARGS:
        if key in model_args:
            add(
                WARNING,
                f"model.init_args.{key} is overridden by the benchmark orchestrator - remove it",
            )
    for key in OVERRIDDEN_DATA_ARGS:
        value = data_args.get(key)
        if value is None or str(value) in (OVERRIDE_PLACEHOLDER, PREDICT_DATASET_CLASS):
            continue
        add(
            WARNING,
            f"data.init_args.{key}={value!r} is overridden by the benchmark orchestrator",
        )

    if not model_args.get("weights_from_checkpoint_path"):
        add(ERROR, "model.init_args.weights_from_checkpoint_path is required")

    if not data_args.get("band_names") and not data_args.get("bands"):
        add(
            ERROR,
            "data.init_args.band_names is required (same list / order as training)",
        )

    separate_metadata = data_args.get("separate_metadata", True)
    if model_args.get("use_metadata_film", True) and not separate_metadata:
        add(
            ERROR,
            "use_metadata_film: true requires data.init_args.separate_metadata: true",
        )

    explicit = model_args.get("in_channels")
    if explicit is not None:
        try:
            expected = _num_input_channels(data_args)
        except ImportError as exc:
            add(
                WARNING,
                f"cannot check in_channels (geo_deep_learning import failed: {exc})",
            )
        else:
            if int(explicit) != expected:
                add(
                    ERROR,
                    f"in_channels={explicit} but band_names/separate_metadata give {expected}",
                )
    return issues


# ---------------------------------------------------------------------------
# 3. Checkpoint hyper-parameters
# ---------------------------------------------------------------------------


def _normalize(key: str, value: Any) -> Any:
    if key == "backbone_kwargs" and value is None:
        return {}
    if isinstance(value, tuple):
        return [_normalize(key, v) for v in value]
    if isinstance(value, list):
        return [_normalize(key, v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(k, v) for k, v in value.items()}
    return value


def task_defaults() -> dict[str, Any]:
    """``__init__`` defaults of the task (effective value of keys absent from a config)."""
    sys.path[:0] = [p for p in (str(REPO_ROOT), str(PACKAGE_ROOT)) if p not in sys.path]
    from geo_deep_learning.tasks_with_models.change_detection_changeformer import (  # noqa: PLC0415
        ChangeDetectionChangeFormer,
    )

    sig = inspect.signature(ChangeDetectionChangeFormer.__init__)
    return {
        k: p.default
        for k, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def load_checkpoint(path: Path) -> dict:
    import torch  # noqa: PLC0415

    return torch.load(str(path), map_location="cpu", weights_only=False)


def compare_hparams(
    model_name: str,
    model_args: dict,
    data_args: dict,
    ckpt_hparams: dict,
    defaults: dict,
    expected_in_channels: int | None,
) -> list[Issue]:
    issues: list[Issue] = []
    for key in (*ARCHITECTURE_HPARAMS, *SOFT_HPARAMS):
        cfg_value = _normalize(key, model_args.get(key, defaults.get(key)))
        if key not in ckpt_hparams:
            if cfg_value != _normalize(key, defaults.get(key)):
                issues.append(
                    Issue(
                        WARNING,
                        model_name,
                        "checkpoint",
                        f"{key}={cfg_value!r} in config but absent from checkpoint hparams "
                        "(older code? trained with the feature disabled)",
                    ),
                )
            continue
        ckpt_value = _normalize(key, ckpt_hparams[key])
        if cfg_value != ckpt_value:
            level = ERROR if key in ARCHITECTURE_HPARAMS else WARNING
            issues.append(
                Issue(
                    level,
                    model_name,
                    "checkpoint",
                    f"{key}: config={cfg_value!r} != checkpoint={ckpt_value!r}",
                ),
            )

    ckpt_in = ckpt_hparams.get("in_channels")
    if (
        expected_in_channels is not None
        and ckpt_in is not None
        and int(ckpt_in) != expected_in_channels
    ):
        issues.append(
            Issue(
                ERROR,
                model_name,
                "checkpoint",
                f"in_channels: band_names/separate_metadata give {expected_in_channels} "
                f"!= checkpoint={ckpt_in} (band_names={data_args.get('band_names')})",
            ),
        )
    elif ckpt_in is None:
        issues.append(
            Issue(
                WARNING,
                model_name,
                "checkpoint",
                "in_channels absent from checkpoint hparams - cannot check band_names",
            ),
        )
    return issues


def check_checkpoint(
    model: BenchmarkModel, cfg: dict, path_maps: Sequence[tuple[str, str]] = (),
) -> tuple[list[Issue], dict | None]:
    model_args = _init_args(cfg, "model")
    data_args = _init_args(cfg, "data")
    ckpt_value = model_args.get("weights_from_checkpoint_path")
    if not ckpt_value:
        return [], None  # already reported by check_static
    ckpt_path = map_path(ckpt_value, path_maps)
    if not ckpt_path.is_file():
        return [
            Issue(ERROR, model.name, "checkpoint", f"file not found: {ckpt_path}"),
        ], None
    try:
        ckpt = load_checkpoint(ckpt_path)
    except Exception as exc:  # noqa: BLE001
        return [
            Issue(ERROR, model.name, "checkpoint", f"cannot load {ckpt_path}: {exc}"),
        ], None

    hparams = dict(ckpt.get("hyper_parameters") or {})
    if not hparams:
        return [
            Issue(
                WARNING,
                model.name,
                "checkpoint",
                "no hyper_parameters in checkpoint - not compared",
            ),
        ], ckpt
    try:
        expected_in = _num_input_channels(data_args)
    except ImportError as exc:
        return [
            Issue(
                WARNING,
                model.name,
                "checkpoint",
                f"geo_deep_learning import failed: {exc}",
            ),
        ], ckpt
    return compare_hparams(
        model.name, model_args, data_args, hparams, task_defaults(), expected_in,
    ), ckpt


# ---------------------------------------------------------------------------
# 4. LightningCLI parsing with the benchmark override
# ---------------------------------------------------------------------------


def build_override(model_name: str, output_root: str, csv_dir: str) -> dict:
    """
    Override injected by the orchestrator (plan §4).

    Keep in sync with scanfire ``orchestrator/benchmark/override.py`` (P6).
    """
    return {
        "trainer": {"devices": 1, "callbacks": [], "logger": False},
        "model": {
            "init_args": {
                "predict_output_dir": f"{output_root.rstrip('/')}/{model_name}",
                "predict_output_layout": "benchmark",
                "predict_run_name": model_name,
                "predict_write_merged_all": False,
                # Best weights of the exact same architecture: any key/shape
                # mismatch must fail the run instead of a silent warning.
                "weights_strict": True,
            },
        },
        "data": {
            "init_args": {
                "dataset_class": PREDICT_DATASET_CLASS,
                "csv_root_folder": f"{csv_dir.rstrip('/')}/{model_name}",
                "csv_file_name": BENCHMARK_CSV_FILE_NAME,
            },
        },
    }


def check_cli(
    model: BenchmarkModel,
    output_root: str,
    csv_dir: str,
    path_maps: Sequence[tuple[str, str]] = (),
    timeout: int = 600,
) -> list[Issue]:
    override = build_override(model.name, output_root, csv_dir)
    with tempfile.TemporaryDirectory() as tmp:
        override_path = Path(tmp) / "benchmark_override.yaml"
        override_path.write_text(
            yaml.safe_dump(override, sort_keys=False), encoding="utf-8",
        )
        cmd = [
            sys.executable,
            str(PACKAGE_ROOT / "train.py"),
            "predict",
            "--config",
            str(map_path(model.config, path_maps)),
            "--config",
            str(override_path),
            "--print_config",
        ]
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(REPO_ROOT), str(PACKAGE_ROOT), os.environ.get("PYTHONPATH", "")],
            ).rstrip(os.pathsep),
        }
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return [
                Issue(
                    ERROR,
                    model.name,
                    "cli",
                    f"--print_config timed out after {timeout}s",
                ),
            ]
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
        return [
            Issue(
                ERROR,
                model.name,
                "cli",
                f"LightningCLI parsing failed (rc={proc.returncode}):\n{tail}",
            ),
        ]

    printed = yaml.safe_load(proc.stdout) or {}
    model_args = _init_args(printed, "model")
    if model_args.get("predict_output_layout") != "benchmark":
        return [
            Issue(
                ERROR,
                model.name,
                "cli",
                "override not applied (predict_output_layout != benchmark)",
            ),
        ]
    if model_args.get("weights_strict") is not True:
        return [
            Issue(
                ERROR,
                model.name,
                "cli",
                "override not applied (weights_strict != true)",
            ),
        ]
    return []


# ---------------------------------------------------------------------------
# 5. Strict weight loading
# ---------------------------------------------------------------------------


def check_load_weights(model: BenchmarkModel, cfg: dict, ckpt: dict) -> list[Issue]:
    sys.path[:0] = [p for p in (str(REPO_ROOT), str(PACKAGE_ROOT)) if p not in sys.path]
    from geo_deep_learning.tasks_with_models.change_detection_changeformer import (  # noqa: PLC0415
        ChangeDetectionChangeFormer,
    )

    model_args = _init_args(cfg, "model")
    data_args = _init_args(cfg, "data")
    params = inspect.signature(ChangeDetectionChangeFormer.__init__).parameters
    skip = {
        "self",
        "main_loss",
        "secondary_loss",
        "optimizer",
        "scheduler",
        "scheduler_config",
        "weights_from_checkpoint_path",
        *OVERRIDDEN_MODEL_ARGS,
    }
    kwargs = {k: v for k, v in model_args.items() if k in params and k not in skip}
    kwargs["in_channels"] = _num_input_channels(data_args)
    try:
        task = ChangeDetectionChangeFormer(
            main_loss=None, secondary_loss=None, **kwargs,
        )
        task.configure_model()
        state = {
            k.removeprefix("model."): v
            for k, v in (ckpt.get("state_dict") or {}).items()
            if k.startswith("model.")
        }
        result = task.model.load_state_dict(state, strict=False)
    except Exception as exc:  # noqa: BLE001 — shape mismatch raises RuntimeError
        return [Issue(ERROR, model.name, "weights", f"cannot load weights: {exc}")]
    issues = []
    if result.missing_keys:
        issues.append(
            Issue(
                ERROR,
                model.name,
                "weights",
                f"{len(result.missing_keys)} missing key(s), e.g. {result.missing_keys[:5]}",
            ),
        )
    if result.unexpected_keys:
        issues.append(
            Issue(
                ERROR,
                model.name,
                "weights",
                f"{len(result.unexpected_keys)} unexpected key(s), e.g. {result.unexpected_keys[:5]}",
            ),
        )
    return issues


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def validate(
    models_file: Path,
    *,
    names: Sequence[str] | None = None,
    include_disabled: bool = False,
    skip_checkpoint: bool = False,
    cli: bool = False,
    load_weights: bool = False,
    path_maps: Sequence[tuple[str, str]] = (),
    output_root: str | None = None,
    csv_dir: str = DEFAULT_CSV_DIR,
) -> list[Issue]:
    registry, issues = load_registry(models_file)
    if any(i.level == ERROR and i.model == "<registry>" for i in issues):
        return issues
    models, select_issues = select_models(
        registry, names, include_disabled=include_disabled,
    )
    issues += select_issues
    root = output_root or registry.output_root or ""

    for model in models:
        logger.info("Validating %s (%s)", model.name, model.config)
        cfg, cfg_issues = load_config(model, path_maps)
        issues += cfg_issues
        if cfg is None:
            continue
        issues += check_static(model, cfg)

        ckpt = None
        if not skip_checkpoint or load_weights:
            ckpt_issues, ckpt = check_checkpoint(model, cfg, path_maps)
            issues += ckpt_issues
        if cli:
            issues += check_cli(model, root, csv_dir, path_maps)
        if load_weights and ckpt is not None:
            issues += check_load_weights(model, cfg, ckpt)
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=Path(os.environ.get("BENCHMARK_MODELS_FILE", DEFAULT_MODELS_FILE)),
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="NAME",
        help="Validate only this model (repeatable; disabled models allowed).",
    )
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument(
        "--skip-checkpoint", action="store_true", help="Do not open checkpoints.",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run LightningCLI --print_config with the override.",
    )
    parser.add_argument(
        "--load-weights", action="store_true", help="Strict weight loading on CPU.",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="SRC=DST",
        help="Translate a path prefix (e.g. /app/models_checkpoints=D:/ckpt). Repeatable.",
    )
    parser.add_argument("--output-root", default=os.environ.get("BENCHMARK_OUTPUT_DIR"))
    parser.add_argument(
        "--csv-dir", default=os.environ.get("BENCHMARK_CSV_DIR", DEFAULT_CSV_DIR),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    issues = validate(
        args.models_file,
        names=args.models,
        include_disabled=args.include_disabled,
        skip_checkpoint=args.skip_checkpoint,
        cli=args.cli,
        load_weights=args.load_weights,
        path_maps=parse_path_maps(args.path_map),
        output_root=args.output_root,
        csv_dir=args.csv_dir,
    )
    for issue in issues:
        print(issue)  # noqa: T201
    n_err = sum(i.level == ERROR for i in issues)
    n_warn = sum(i.level == WARNING for i in issues)
    print(f"\n{'FAILED' if n_err else 'OK'} - {n_err} error(s), {n_warn} warning(s)")  # noqa: T201
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
