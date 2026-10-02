#!/usr/bin/env bash
# Sequentially train the "geo-deep-learning" (ChangeFormer) service once per
# band-combo config, saving each run's full log and appending its test
# metrics + best checkpoint path to a single CSV.
#
# Usage (run from the repo root on the training server):
#   ./scripts/run_band_sweep.sh                 # detached (default)
#   ./scripts/run_band_sweep.sh --foreground    # attached/blocking
#   ./scripts/run_band_sweep.sh --status
#
# Env overrides:
#   SERVICE       docker-compose service to run       (default: geo-deep-learning)
#   COMPOSE_FILE  compose file                          (default: docker-compose.yaml)
#   METRICS_CSV   output CSV                            (default: results/band_sweep_metrics.csv)
#   LOG_DIR       directory for per-run raw logs        (default: logs/band_sweep)
#   SWEEP_LOG     detached process log                   (default: logs/band_sweep/sweep.log)
#   PID_FILE      detached process PID file              (default: run/band_sweep.pid)
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
SWEEP_LOG="${SWEEP_LOG:-$LOG_DIR/sweep.log}"
PID_FILE="${PID_FILE:-run/band_sweep.pid}"
LOCK_DIR="${PID_FILE}.lock"
SCRIPT_PATH="$REPO_ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"

mode="detach"
case "${1:-}" in
  ""|-d|--detach) mode="detach" ;;
  -f|--foreground) mode="foreground" ;;
  --status) mode="status" ;;
  -h|--help)
    cat <<EOF
Usage: $0 [--detach|--foreground|--status]

  -d, --detach      Launch the complete sweep in the background (default).
  -f, --foreground  Run synchronously and stream output in this terminal.
      --status      Report whether a sweep is currently running.
  -h, --help        Show this help.
EOF
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    echo "Run '$0 --help' for usage." >&2
    exit 2
    ;;
esac

running_pid() {
  local pid
  [[ -r "$PID_FILE" ]] || return 1
  read -r pid < "$PID_FILE" || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null || return 1
  printf '%s' "$pid"
}

clear_stale_state() {
  if [[ -e "$PID_FILE" ]] && ! running_pid >/dev/null; then
    rm -f "$PID_FILE"
    rm -rf "$LOCK_DIR"
  elif [[ ! -e "$PID_FILE" && -d "$LOCK_DIR" ]]; then
    rm -rf "$LOCK_DIR"
  fi
}

if [[ "$mode" == "status" ]]; then
  if pid="$(running_pid)"; then
    echo "Band sweep is running (PID $pid). Log: $SWEEP_LOG"
    exit 0
  fi
  clear_stale_state
  echo "Band sweep is not running."
  exit 1
fi

mkdir -p "$LOG_DIR" "$(dirname "$METRICS_CSV")" "$(dirname "$PID_FILE")" "$(dirname "$SWEEP_LOG")"
clear_stale_state

if [[ "$mode" == "detach" ]]; then
  if pid="$(running_pid)"; then
    echo "A band sweep is already running (PID $pid). Log: $SWEEP_LOG" >&2
    exit 1
  fi
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Cannot acquire sweep lock: $LOCK_DIR" >&2
    exit 1
  fi

  # Temporarily identify this launcher as the lock owner. The child waits until
  # the file is replaced with its PID, eliminating the launch/lock race.
  printf '%s\n' "$$" > "$PID_FILE"
  printf '\n=== [%s] Launching detached band sweep ===\n' "$(date -Is)" >> "$SWEEP_LOG"
  BAND_SWEEP_LOCK_HELD=1 nohup bash -c '
    while [[ ! -r "$1" ]] || [[ "$(<"$1")" != "$$" ]]; do sleep 0.05; done
    exec bash "$2" --foreground
  ' _ "$PID_FILE" "$SCRIPT_PATH" </dev/null >> "$SWEEP_LOG" 2>&1 &
  child_pid=$!
  printf '%s\n' "$child_pid" > "$PID_FILE"

  echo "Band sweep started in the background (PID $child_pid)."
  echo "Combined log: $SWEEP_LOG"
  echo "Status: $SCRIPT_PATH --status"
  exit 0
fi

if [[ "${BAND_SWEEP_LOCK_HELD:-0}" != "1" ]]; then
  if pid="$(running_pid)"; then
    echo "A band sweep is already running (PID $pid). Log: $SWEEP_LOG" >&2
    exit 1
  fi
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Cannot acquire sweep lock: $LOCK_DIR" >&2
    exit 1
  fi
fi

printf '%s\n' "$$" > "$PID_FILE"
cleanup() {
  local pid=""
  [[ -r "$PID_FILE" ]] && read -r pid < "$PID_FILE" || true
  if [[ "$pid" == "$$" ]]; then
    rm -f "$PID_FILE"
    rm -rf "$LOCK_DIR"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CONFIGS=(
  "configs/rcm_change_detection_changeformer_all9bands.yaml"
  "configs/rcm_change_detection_changeformer_lia_rlrr.yaml"
  "configs/rcm_change_detection_changeformer_mchi4bands.yaml"
  "configs/rcm_change_detection_changeformer_mchi5bands.yaml"
  "configs/rcm_change_detection_changeformer_mchi_rrrl7bands.yaml"
)

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

  # Start detached (-d), then follow its logs synchronously: `logs -f` returns
  # on its own once the one-shot fit+test entrypoint stops the container, so
  # the loop still waits for completion without needing --abort-on-container-exit
  # (which cannot be combined with --detach).
  compose up -d --build "$SERVICE"
  cid="$(compose ps -q "$SERVICE")"
  compose logs -f --no-color "$SERVICE" 2>&1 | tee "$log_file" || true

  exit_code="$(docker inspect "$cid" --format='{{.State.ExitCode}}' 2>/dev/null || echo 1)"
  if [[ "$exit_code" != "0" ]]; then
    echo "!! Training failed for $cfg (exit code $exit_code) — see $log_file" >&2
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
