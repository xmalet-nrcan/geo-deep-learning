#!/usr/bin/env python
"""
Check the outputs of a model benchmark run (dev plan P8 — recette).

Read-only. Run on the GPU server after ``make benchmark-start`` (or in the
orchestrator container: ``make benchmark-check``).

Per model of ``configs/benchmark/models.yaml`` (enabled, or ``--model``), in
``<output_root>/<model>/``:

1. **Layout** — ``<event_id>/`` folders only (+ ``_runs/``, ``manifest_latest.json``);
   no production layout (``predictions/``), no ``merged_all.tif``.
2. **Files** — every per-cell ``…_cell-<id>_beam-…_pass-….tif`` has its
   ``_prob.tif`` (and vice-versa) and its per pre/post pair merged raster
   (same name without ``_cell-<id>``); the event id of the name matches its folder.
3. **Filters** (decision 5) — no output for a beam / satellite pass excluded by
   the model config (``data.init_args.beams`` / ``satellite_pass``).
4. **Expected** (``--csv-dir``, default ``BENCHMARK_CSV_DIR``) — every
   ``output_name`` exported by the orchestrator in
   ``<csv_dir>/<model>/vw_input_files_for_model_test.csv`` (and allowed by the
   filters) was produced.
5. **Rasters** (needs ``rasterio``; skip: ``--no-rasters``) — classes ``uint16`` /
   nodata ``32767`` / values ⊂ {0..num_classes-1, 32767}; probability ``float32`` /
   nodata ``-1.0`` / values ⊂ [0, 1] ∪ {-1}; class and probability on the same grid;
   CRS set.
6. **Cross-model** — same file names per event for every model (differences
   explained by the beam / pass filters are reported as INFO), and the same
   raster grid (shape + transform + CRS) for the same file name (geographic
   alignment).

Exit code: ``0`` if no ERROR, ``1`` otherwise.

Examples::

    python scripts/check_benchmark_outputs.py
    python scripts/check_benchmark_outputs.py --model m1 --model m2 --event 42
    python scripts/check_benchmark_outputs.py --output-root /mnt/.../benchmark_outputs --no-rasters
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_benchmark_configs as vbc  # noqa: E402

#: Must match ``geo_deep_learning.datasets.rcm_change_detection_dataset.NO_DATA`` and
#: ``tasks_with_models.change_detection_changeformer.PROBABILITY_NODATA`` (tested).
CLASS_NODATA = 32767
PROB_NODATA = -1.0
CLASS_DTYPE = "uint16"
PROB_DTYPE = "float32"

PROB_SUFFIX = "_prob.tif"
RUNS_DIRNAME = "_runs"
LATEST_MANIFEST = "manifest_latest.json"
MERGED_ALL = "merged_all.tif"
PRODUCTION_DIRNAME = "predictions"
CSV_FILE_NAME = vbc.BENCHMARK_CSV_FILE_NAME

#: ``event-{id}_start-…_end_…_pre-g…-…_post-g…-…[_cell-{cell}]_beam-{beam}_pass-{pass}[_prob].tif``
NAME_RE = re.compile(
    r"^event-(?P<event>[^_]+)_start-[^_]+_end_[^_]+_pre-g[^_]+_post-g[^_]+"
    r"(?:_cell-(?P<cell>.+?))?_beam-(?P<beam>[^_]+)_pass-(?P<sat_pass>.+?)(?P<prob>_prob)?\.tif$",
)

ERROR = vbc.ERROR
WARNING = vbc.WARNING
INFO = "INFO"

logger = logging.getLogger("check_benchmark_outputs")


@dataclass(frozen=True)
class Finding:
    level: str
    model: str
    check: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.model} / {self.check}: {self.message}"


@dataclass(frozen=True)
class OutputFile:
    """One prediction raster of a model output folder."""

    event_dir: str
    name: str
    event: str
    cell: str | None
    beam: str
    sat_pass: str
    is_prob: bool

    @property
    def kind(self) -> str:
        if self.cell is None:
            return "merged"
        return "prob" if self.is_prob else "cell"

    @property
    def merged_name(self) -> str:
        """Per pre/post pair merged raster this per-cell file belongs to.

        The pass is normalised (:func:`_pass_key`): per-cell names carry the DB
        ``sat_pass`` (e.g. ``Ascending``) whereas the merge writes the dataset
        value (``ASC`` / ``DESC``).
        """
        stem = re.sub(r"_cell-.+?(?=_beam-)", "", self.name)
        stem = stem.removesuffix(PROB_SUFFIX).removesuffix(".tif") + ".tif"
        return _normalise_pass(stem)

    @property
    def merged_key(self) -> str:
        return f"{self.event_dir}/{self.merged_name}"

    @property
    def key(self) -> str:
        return f"{self.event_dir}/{self.name}"


@dataclass
class ModelFilters:
    beams: list[str] = field(default_factory=list)
    satellite_pass: str | None = None
    num_classes: int = 2

    def allows(self, beam: str, sat_pass: str) -> bool:
        if self.beams and beam.upper() not in self.beams:
            return False
        return not (self.satellite_pass and _pass_key(sat_pass) != _pass_key(self.satellite_pass))


@dataclass
class ModelOutputs:
    name: str
    root: Path
    filters: ModelFilters
    files: dict[str, OutputFile] = field(default_factory=dict)  # key → file
    grids: dict[str, tuple] = field(default_factory=dict)  # key → (shape, transform, crs)


def _pass_key(value: str) -> str:
    """``Ascending`` / ``ASC`` / ``ascending`` → ``ASC`` ; ``Descending`` / ``DESC`` → ``DES``."""
    return str(value).strip().upper()[:3]


def _normalise_pass(name: str) -> str:
    return re.sub(r"_pass-(.+?)\.tif$", lambda m: f"_pass-{_pass_key(m[1])}.tif", name)


def parse_name(event_dir: str, name: str) -> OutputFile | None:
    match = NAME_RE.match(name)
    if not match:
        return None
    return OutputFile(
        event_dir=event_dir,
        name=name,
        event=match["event"],
        cell=match["cell"],
        beam=match["beam"],
        sat_pass=match["sat_pass"],
        is_prob=bool(match["prob"]),
    )


def model_filters(cfg: dict | None) -> ModelFilters:
    if not cfg:
        return ModelFilters()
    data_args = vbc._init_args(cfg, "data")  # noqa: SLF001
    model_args = vbc._init_args(cfg, "model")  # noqa: SLF001
    beams = data_args.get("beams") or []
    if isinstance(beams, str):
        beams = [beams]
    num_classes = int(model_args.get("num_classes") or 1)
    return ModelFilters(
        beams=[str(b).upper() for b in beams],
        satellite_pass=str(data_args["satellite_pass"]) if data_args.get("satellite_pass") else None,
        num_classes=num_classes + 1 if num_classes == 1 else num_classes,
    )


# ---------------------------------------------------------------------------
# 1-3. Layout, files, filters
# ---------------------------------------------------------------------------


def scan_model(
    outputs: ModelOutputs, events: set[str] | None,
) -> list[Finding]:
    findings: list[Finding] = []

    def add(level: str, check: str, message: str) -> None:
        findings.append(Finding(level, outputs.name, check, message))

    root = outputs.root
    if not root.is_dir():
        add(ERROR, "layout", f"output folder not found: {root}")
        return findings
    if (root / PRODUCTION_DIRNAME).exists():
        add(ERROR, "layout", f"production layout folder found: {root / PRODUCTION_DIRNAME}")
    if not (root / LATEST_MANIFEST).is_file():
        add(WARNING, "layout", f"{LATEST_MANIFEST} missing (predict never completed?)")
    if not any((root / RUNS_DIRNAME).glob("*/manifest.json")):
        add(WARNING, "layout", f"no {RUNS_DIRNAME}/<date>/manifest.json")

    for entry in sorted(root.iterdir()):
        if entry.name in (RUNS_DIRNAME, PRODUCTION_DIRNAME, LATEST_MANIFEST):
            continue
        if not entry.is_dir():
            add(WARNING, "layout", f"unexpected file at model root: {entry.name}")
            continue
        if events and entry.name not in events:
            continue
        for path in sorted(entry.iterdir()):
            if path.is_dir():
                add(ERROR, "layout", f"unexpected sub-folder {entry.name}/{path.name} (benchmark layout is flat)")
                continue
            if path.name == MERGED_ALL:
                add(ERROR, "layout", f"{entry.name}/{MERGED_ALL} must not exist in the benchmark layout")
                continue
            if path.suffix.lower() != ".tif":
                continue
            out = parse_name(entry.name, path.name)
            if out is None:
                add(WARNING, "files", f"unrecognised raster name {entry.name}/{path.name}")
                continue
            outputs.files[out.key] = out

    by_key = outputs.files
    merged_keys = {
        f"{o.event_dir}/{_normalise_pass(o.name)}": o.key for o in by_key.values() if o.kind == "merged"
    }
    for out in by_key.values():
        if out.event != out.event_dir:
            add(ERROR, "files", f"{out.key}: event-{out.event} stored in folder {out.event_dir}")
        if not outputs.filters.allows(out.beam, out.sat_pass):
            add(ERROR, "filters", f"{out.key}: beam {out.beam} / pass {out.sat_pass} excluded by the model config")
        if out.kind == "cell":
            prob = f"{out.event_dir}/{out.name.removesuffix('.tif')}{PROB_SUFFIX}"
            if prob not in by_key:
                add(ERROR, "files", f"{out.key}: probability raster missing")
            if out.merged_key not in merged_keys:
                add(ERROR, "files", f"{out.key}: per-pair merged raster {out.merged_name} missing")
        elif out.kind == "prob":
            cls = f"{out.event_dir}/{out.name.removesuffix(PROB_SUFFIX)}.tif"
            if cls not in by_key:
                add(ERROR, "files", f"{out.key}: class raster missing")

    merged_with_cells = {o.merged_key for o in by_key.values() if o.kind == "cell"}
    for norm_key, key in merged_keys.items():
        if norm_key not in merged_with_cells:
            add(WARNING, "files", f"{key}: merged raster without any per-cell raster (stale run?)")

    if not by_key:
        add(ERROR, "files", "no prediction raster found")
    return findings


# ---------------------------------------------------------------------------
# 4. Expected outputs (orchestrator CSV export)
# ---------------------------------------------------------------------------


def check_expected(outputs: ModelOutputs, csv_dir: Path | None, events: set[str] | None) -> list[Finding]:
    if csv_dir is None:
        return []
    csv_path = csv_dir / outputs.name / CSV_FILE_NAME
    if not csv_path.is_file():
        return [Finding(WARNING, outputs.name, "expected", f"no exported CSV ({csv_path}) — check skipped")]
    import csv  # noqa: PLC0415

    findings: list[Finding] = []
    n_expected = n_filtered = 0
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("output_name")
            event = str(row.get("event_id", "")).strip()
            if not name or (events and event not in events):
                continue
            if not outputs.filters.allows(str(row.get("beam", "")), str(row.get("sat_pass", ""))):
                n_filtered += 1
                continue
            n_expected += 1
            if f"{event}/{name}" not in outputs.files:
                findings.append(
                    Finding(ERROR, outputs.name, "expected", f"exported pair not produced: {event}/{name}"),
                )
    findings.append(
        Finding(
            INFO, outputs.name, "expected",
            f"{n_expected} exported pair(s) checked, {n_filtered} excluded by the model filters ({csv_path})",
        ),
    )
    return findings


# ---------------------------------------------------------------------------
# 5. Rasters
# ---------------------------------------------------------------------------


def check_rasters(outputs: ModelOutputs) -> list[Finding]:
    import numpy as np  # noqa: PLC0415
    import rasterio  # noqa: PLC0415

    findings: list[Finding] = []

    def add(level: str, message: str) -> None:
        findings.append(Finding(level, outputs.name, "rasters", message))

    allowed_classes = set(range(outputs.filters.num_classes)) | {CLASS_NODATA}
    for key, out in sorted(outputs.files.items()):
        path = outputs.root / key
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
                outputs.grids[key] = (src.shape, tuple(src.transform)[:6], src.crs.to_string() if src.crs else None)
                if src.crs is None:
                    add(ERROR, f"{key}: no CRS")
                if src.count != 1:
                    add(ERROR, f"{key}: {src.count} bands (expected 1)")
                if out.is_prob:
                    if src.dtypes[0] != PROB_DTYPE or src.nodata != PROB_NODATA:
                        add(ERROR, f"{key}: dtype/nodata {src.dtypes[0]}/{src.nodata} "
                                   f"(expected {PROB_DTYPE}/{PROB_NODATA})")
                    valid = data[data != PROB_NODATA]
                    if valid.size and (np.nanmin(valid) < 0.0 or np.nanmax(valid) > 1.0 or np.isnan(valid).any()):
                        add(ERROR, f"{key}: probabilities outside [0, 1] or NaN")
                else:
                    if src.dtypes[0] != CLASS_DTYPE or src.nodata != CLASS_NODATA:
                        add(ERROR, f"{key}: dtype/nodata {src.dtypes[0]}/{src.nodata} "
                                   f"(expected {CLASS_DTYPE}/{CLASS_NODATA})")
                    unexpected = sorted(set(np.unique(data).tolist()) - allowed_classes)
                    if unexpected:
                        add(ERROR, f"{key}: unexpected class value(s) {unexpected[:10]}")
                if not (data != src.nodata).any():
                    add(WARNING, f"{key}: only nodata")
        except Exception as exc:  # noqa: BLE001 — corrupted / unreadable raster
            add(ERROR, f"{key}: unreadable ({exc})")

    for key, out in outputs.files.items():
        if out.kind == "cell":
            prob = f"{out.event_dir}/{out.name.removesuffix('.tif')}{PROB_SUFFIX}"
            if key in outputs.grids and prob in outputs.grids and outputs.grids[key] != outputs.grids[prob]:
                add(ERROR, f"{key}: class and probability rasters are not on the same grid")

    shapes = Counter(g[0] for k, g in outputs.grids.items() if outputs.files[k].kind == "cell")
    if shapes:
        add(INFO, "per-cell raster shapes: " + ", ".join(f"{h}x{w} ×{n}" for (h, w), n in shapes.most_common(5)))
    return findings


# ---------------------------------------------------------------------------
# 6. Cross-model
# ---------------------------------------------------------------------------


def check_cross_model(models: Sequence[ModelOutputs]) -> list[Finding]:
    if len(models) < 2:  # noqa: PLR2004
        return []
    findings: list[Finding] = []
    union: dict[str, OutputFile] = {}
    for m in models:
        union.update(m.files)

    events = sorted({o.event_dir for o in union.values()})
    for m in models:
        missing = [o for k, o in union.items() if k not in m.files]
        explained = [o for o in missing if not m.filters.allows(o.beam, o.sat_pass)]
        unexplained = [o for o in missing if m.filters.allows(o.beam, o.sat_pass)]
        if explained:
            findings.append(Finding(INFO, m.name, "cross-model",
                                    f"{len(explained)} file(s) of other models absent — excluded by the beam/pass filters"))
        for o in sorted(unexplained, key=lambda x: x.key)[:50]:
            findings.append(Finding(WARNING, m.name, "cross-model", f"{o.key} produced by another model only"))
        if len(unexplained) > 50:  # noqa: PLR2004
            findings.append(Finding(WARNING, m.name, "cross-model", f"… {len(unexplained) - 50} more"))

    for key in sorted(union):
        grids = {m.name: m.grids[key] for m in models if key in m.grids}
        if len(set(grids.values())) > 1:
            findings.append(Finding(ERROR, "<all>", "alignment",
                                    f"{key}: different grid between models {sorted(grids)}"))
    findings.append(Finding(INFO, "<all>", "cross-model",
                            f"{len(models)} models, {len(events)} event(s), {len(union)} distinct raster(s)"))
    return findings


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_checks(
    models_file: Path,
    *,
    names: Sequence[str] | None = None,
    output_root: str | None = None,
    csv_dir: str | None = None,
    events: Iterable[str] | None = None,
    rasters: bool = True,
    path_maps: Sequence[tuple[str, str]] = (),
) -> tuple[list[Finding], list[ModelOutputs]]:
    registry, reg_issues = vbc.load_registry(models_file)
    findings = [Finding(i.level, i.model, i.check, i.message) for i in reg_issues]
    if any(i.level == ERROR for i in reg_issues):
        return findings, []
    selected, select_issues = vbc.select_models(registry, names)
    findings += [Finding(i.level, i.model, i.check, i.message) for i in select_issues]

    root = output_root or registry.output_root or ""
    csv_root = vbc.map_path(csv_dir, path_maps) if csv_dir else None
    event_set = {str(e) for e in events} if events else None
    if rasters:
        try:
            import rasterio  # noqa: F401, PLC0415
        except ImportError:
            findings.append(Finding(WARNING, "<all>", "rasters", "rasterio not installed — raster checks skipped"))
            rasters = False

    all_outputs: list[ModelOutputs] = []
    for model in selected:
        cfg, _ = vbc.load_config(model, path_maps)
        outputs = ModelOutputs(
            name=model.name,
            root=vbc.map_path(f"{root.rstrip('/')}/{model.name}", path_maps),
            filters=model_filters(cfg),
        )
        findings += scan_model(outputs, event_set)
        findings += check_expected(outputs, csv_root, event_set)
        if rasters and outputs.files:
            findings += check_rasters(outputs)
        all_outputs.append(outputs)
    findings += check_cross_model([m for m in all_outputs if m.files])
    return findings, all_outputs


def summary_lines(outputs: Sequence[ModelOutputs], findings: Sequence[Finding]) -> list[str]:
    errors = Counter(f.model for f in findings if f.level == ERROR)
    warnings = Counter(f.model for f in findings if f.level == WARNING)
    lines = [f"{'model':<40} {'events':>6} {'cells':>6} {'prob':>6} {'merged':>6} {'errors':>6} {'warn':>6}"]
    for m in outputs:
        kinds = Counter(o.kind for o in m.files.values())
        n_events = len({o.event_dir for o in m.files.values()})
        lines.append(
            f"{m.name:<40} {n_events:>6} {kinds['cell']:>6} {kinds['prob']:>6} {kinds['merged']:>6} "
            f"{errors[m.name]:>6} {warnings[m.name]:>6}",
        )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-file", type=Path, default=vbc.DEFAULT_MODELS_FILE)
    parser.add_argument("--model", dest="models", action="append", default=[],
                        help="Check only this model (repeatable).")
    parser.add_argument("--output-root", default=os.environ.get("BENCHMARK_OUTPUT_DIR") or None,
                        help="Override output_root of the registry (env BENCHMARK_OUTPUT_DIR).")
    parser.add_argument("--csv-dir", default=os.environ.get("BENCHMARK_CSV_DIR", vbc.DEFAULT_CSV_DIR),
                        help="Orchestrator CSV export root ('' to skip the expected-outputs check).")
    parser.add_argument("--event", dest="events", action="append", default=[], help="Only this event id (repeatable).")
    parser.add_argument("--no-rasters", action="store_true", help="Skip raster content checks.")
    parser.add_argument("--path-map", action="append", default=[], metavar="SRC=DST",
                        help="Map container path prefixes to local ones (repeatable).")
    parser.add_argument("--quiet", action="store_true", help="Only print ERROR / WARNING findings and the summary.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    findings, outputs = run_checks(
        args.models_file,
        names=args.models or None,
        output_root=args.output_root,
        csv_dir=args.csv_dir or None,
        events=args.events or None,
        rasters=not args.no_rasters,
        path_maps=vbc.parse_path_maps(args.path_map),
    )
    for finding in findings:
        if args.quiet and finding.level == INFO:
            continue
        logger.info("%s", finding)
    logger.info("")
    for line in summary_lines(outputs, findings):
        logger.info("%s", line)
    n_errors = sum(f.level == ERROR for f in findings)
    n_warnings = sum(f.level == WARNING for f in findings)
    logger.info("\n%s — %d error(s), %d warning(s)", "FAILED" if n_errors else "OK", n_errors, n_warnings)
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
