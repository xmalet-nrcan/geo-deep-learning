#!/usr/bin/env bash
# Sequentially train the "geo-deep-learning" (ChangeFormer) service once per
# band-combo config, saving each run's full log and appending its test
# metrics + best checkpoint path to a single CSV.
#
# Usage (run from the repo root on the training server):
#   ./scripts/run_band_sweep.sh
#
# Env overrides:
#   SERVICE       docker-compose service to run       (default: geo-deep-learning)
#   COMPOSE_FILE  compose file                          (default: docker-compose.yaml)
#   METRICS_CSV   output CSV                            (default: results/band_sweep_metrics.csv)
#   LOG_DIR       directory for per-run raw logs        (default: logs/band_sweep)
#
# Relies on docker-compose.yaml's ${TRAIN_CONFIG:-...} volume substitution for
# the geo-deep-learning service's /app/configs/model_conf.yaml mount, and on
# scripts/extract_test_metrics.py to parse the Rich test-metrics table + the
# "Loading weights from checkpoint" / "Testing best model from path" log lines.
set -euo pipefail

# Resolve repo root (this script lives in <repo>/scripts/).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SERVICE="${SERVICE:-geo-deep-learning}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yaml}"
METRICS_CSV="${METRICS_CSV:-results/band_sweep_metrics.csv}"
LOG_DIR="${LOG_DIR:-logs/band_sweep}"

CONFIGS=(
  "configs/rcm_change_detection_changeformer_all9bands.yaml"
  "configs/rcm_change_detection_changeformer_lia_rlrr.yaml"
  "configs/rcm_change_detection_changeformer_mchi4bands.yaml"
  "configs/rcm_change_detection_changeformer_mchi5bands.yaml"
  "configs/rcm_change_detection_changeformer_mchi_rrrl7bands.yaml"
)

mkdir -p "$LOG_DIR" "$(dirname "$METRICS_CSV")"

compose() { docker compose -f "$COMPOSE_FILE" "$@"; }

echo "=== Band sweep: ${#CONFIGS[@]} configs, service=$SERVICE, csv=$METRICS_CSV ==="

failures=()

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "!! Config not found, skipping: $cfg" >&2
    failures+=("$cfg")
    continue
  fi

  name="$(basename "$cfg" .yaml)"
  log_file="$LOG_DIR/${name}.log"
  echo ""
  echo "### [$(date -Is)] Training: $cfg (log: $log_file) ###"

  # Absolute host path required: the compose volume mount needs a real path,
  # not one relative to the container's working directory.
  export TRAIN_CONFIG="$REPO_ROOT/$cfg"

  # Remove any stray container from a previous run that shares container_name.
  compose rm -f -s "$SERVICE" >/dev/null 2>&1 || true

  # Foreground run: blocks until the one-shot fit+test entrypoint exits.
  if ! compose up --build --abort-on-container-exit "$SERVICE" 2>&1 | tee "$log_file"; then
    echo "!! Training failed for $cfg — see $log_file" >&2
    failures+=("$cfg")
  fi

  compose rm -f -s "$SERVICE" >/dev/null 2>&1 || true

  if ! python3 scripts/extract_test_metrics.py "$log_file" --csv "$METRICS_CSV" --label "$name"; then
    echo "!! No test metrics found for $cfg — see $log_file" >&2
    failures+=("$cfg (no metrics)")
  fi
done

echo ""
echo "=== Sweep done. Results: $METRICS_CSV ==="
if [[ ${#failures[@]} -gt 0 ]]; then
  echo "Issues encountered with:" >&2
  printf ' - %s\n' "${failures[@]}" >&2
  exit 1
fi
