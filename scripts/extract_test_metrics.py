#!/usr/bin/env python
"""Extract test metrics + model/bands/checkpoint used from a `make train` log, into a CSV.

Parses the Lightning/Rich output produced by `GeoDeepLearningCLI.after_fit`
(see geo_deep_learning/train.py), the "Loading weights from checkpoint" log
lines (geo_deep_learning/utils/models.py), the "Using change_detection_model=..."
log line (geo_deep_learning/models/change_detection/change_detection_model.py),
and the "Treating bands: [...]" log line
(geo_deep_learning/datasets/rcm_change_detection_dataset.py). Designed to be
fed either a log file saved on disk, or piped log text via stdin (e.g. from
`docker compose logs`).

Usage
-----
    # From a saved log file
    python scripts/extract_test_metrics.py /path/to/train.log --csv results/test_metrics.csv

    # Directly from docker compose logs (prod server)
    docker compose logs --no-color geo-deep-learning \\
        | python scripts/extract_test_metrics.py - --csv results/test_metrics.csv

    # Several log files at once (e.g. one per band combo already saved)
    python scripts/extract_test_metrics.py logs/*.log --csv results/test_metrics.csv

Each run found in the log(s) appends one row to the CSV with: the backbone
used (`change_detection_model`), the band combo (`band_names`), the best
checkpoint path used for `test`, and every `test_*` metric reported by the
Rich results table. The CSV header is rewritten each time so it stays a
union of every metric ever seen (robust if the metric set changes later,
e.g. extra classwise IoU columns).
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# "Testing best model from path: /app/models_checkpoints/.../foo.ckpt" (train.py after_fit)
BEST_MODEL_PATH_RE = re.compile(r"Testing best model from path:\s*(\S+)")
# "Loading weights from checkpoint: /app/models_checkpoints/.../foo.ckpt" (utils/models.py, tasks_with_models/*)
LOAD_CHECKPOINT_RE = re.compile(r"Loading weights from checkpoint:\s*(\S+\.ckpt)")
# "Using change_detection_model=changeformer_7 (in_channels=10, ...)" (change_detection_model.py)
MODEL_RE = re.compile(r"[Uu]sing change_detection_model=(\S+?)(?:\s|\()")
# "Treating bands: ['LOCALINCANGLE', 'PDN', ...]" (rcm_change_detection_dataset.py)
BAND_NAMES_RE = re.compile(r"Treating bands:\s*(\[[^\]]*\])")

# Rich table row, e.g.: "│          test_f1          │    0.7909554839134216     │"
# Box borders use either light (│) or heavy (┃) vertical bars depending on the row.
METRIC_ROW_RE = re.compile(
    r"[│┃]\s*(test_[A-Za-z0-9_]+)\s*[│┃]\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*[│┃]"
)

BASE_FIELDS = ["timestamp", "log_source", "label", "model", "band_names", "run_name", "checkpoint_path"]


def strip_ansi(text: str) -> str:
    """Remove ANSI color escape codes (present when docker captured a tty)."""
    return ANSI_RE.sub("", text)


def extract_checkpoint_path(text: str) -> str:
    """Return the checkpoint path used for the `test` step, if found."""
    match = BEST_MODEL_PATH_RE.search(text)
    if match:
        return match.group(1)
    match = LOAD_CHECKPOINT_RE.search(text)
    if match:
        return match.group(1)
    return ""


def extract_model(text: str) -> str:
    """Return the `change_detection_model` backbone key, if logged."""
    match = MODEL_RE.search(text)
    return match.group(1) if match else ""


def extract_band_names(text: str) -> str:
    """Return the configured band_names (comma-joined), if logged."""
    match = BAND_NAMES_RE.search(text)
    if not match:
        return ""
    try:
        bands = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return match.group(1)
    return ",".join(str(b) for b in bands)


def derive_run_name(checkpoint_path: str) -> str:
    """Derive a short run name from a checkpoint filename (strip epoch/val_loss suffix)."""
    if not checkpoint_path:
        return ""
    stem = Path(checkpoint_path).stem
    # Strip trailing "-{epoch:02d}-{val_loss:.3f}" style suffix if present.
    stem = re.sub(r"-epoch=\d+.*$", "", stem)
    return stem


def extract_metrics(text: str) -> dict[str, float]:
    """Return every `test_*` metric found in the Rich results table(s)."""
    metrics: dict[str, float] = {}
    for name, value in METRIC_ROW_RE.findall(text):
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


def parse_log(text: str, source: str, label: str = "") -> dict[str, object] | None:
    """Parse one log's text into a single result row, or None if no metrics found."""
    clean = strip_ansi(text)
    metrics = extract_metrics(clean)
    if not metrics:
        return None
    checkpoint_path = extract_checkpoint_path(clean)
    row: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "log_source": source,
        "label": label,
        "model": extract_model(clean),
        "band_names": extract_band_names(clean),
        "run_name": derive_run_name(checkpoint_path),
        "checkpoint_path": checkpoint_path,
    }
    row.update(metrics)
    return row


def load_existing_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(BASE_FIELDS)
    extra_fields: set[str] = set()
    for row in rows:
        extra_fields.update(k for k in row if k not in BASE_FIELDS)
    fieldnames += sorted(extra_fields)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "logs",
        nargs="+",
        help="Log file path(s) to parse. Use '-' to read from stdin (e.g. piped from docker compose logs).",
    )
    parser.add_argument(
        "--csv",
        default="results/test_metrics.csv",
        help="Output CSV path (appended to; default: results/test_metrics.csv).",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Explicit label applied to every row parsed in this invocation "
             "(e.g. the config name), stored in the 'label' column. Useful when "
             "sweeping several configs since checkpoint-derived run_name alone "
             "may not be distinctive enough.",
    )
    args = parser.parse_args()

    new_rows: list[dict[str, object]] = []
    for log_arg in args.logs:
        if log_arg == "-":
            text = sys.stdin.read()
            source = "stdin"
        else:
            path = Path(log_arg)
            text = path.read_text(encoding="utf-8", errors="replace")
            source = path.name
        row = parse_log(text, source, label=args.label)
        if row is None:
            print(f"[extract_test_metrics] No test metrics found in: {source}", file=sys.stderr)
            continue
        new_rows.append(row)
        print(f"[extract_test_metrics] Found run: {row['run_name'] or row['checkpoint_path']} "
              f"({len(row) - len(BASE_FIELDS)} metrics)", file=sys.stderr)

    if not new_rows:
        print("[extract_test_metrics] Nothing to write, exiting.", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.csv)
    all_rows = load_existing_rows(csv_path) + new_rows
    write_rows(csv_path, all_rows)
    print(f"[extract_test_metrics] Wrote {len(new_rows)} new row(s) to {csv_path} "
          f"({len(all_rows)} total).", file=sys.stderr)


if __name__ == "__main__":
    main()
