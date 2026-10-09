# Practical Makefile for geo-deep-learning docker-compose operations
# Usage examples:
#   make up
#   make logs
#   make logs-service SERVICE=geo-deep-learning-cs2
#   make train
#   make train-cs2 TRAIN_CMD=validate
#   make orchestrator-start

SHELL := /bin/sh

# ---- Compose settings ----
COMPOSE_FILE ?= docker-compose.yaml
COMPOSE := docker compose -f $(COMPOSE_FILE)

# ---- Service names ----
SERVICE_DEFAULT ?= geo-deep-learning
SERVICE_ORCH ?= event_detection_orchestrator
SERVICE_BENCH ?= model_benchmark_orchestrator

# ---- Model benchmark (profile "benchmark", single pass) ----
COMPOSE_BENCH := $(COMPOSE) --profile benchmark
# Extra orchestrator args, e.g. BENCHMARK_ARGS="--model cs2base_cosine_restarts_e24 --test-case 3 --force"
BENCHMARK_ARGS ?=
# Extra validator args, e.g. VALIDATE_ARGS="--model cs2base_cosine_restarts_e24"
VALIDATE_ARGS ?= --cli --load-weights
# Extra output-checker args, e.g. CHECK_ARGS="--event 42 --quiet"
CHECK_ARGS ?=

# Generic service selector for per-service targets
SERVICE ?= $(SERVICE_DEFAULT)

# Training mode used by entrypoint (fit/validate/test/predict...)
TRAIN_CMD ?= fit

.PHONY: help build up stop restart ps status logs logs-follow logs-service logs-follow-service \
        train train-cs2 orchestrator-start down cleanup extract-metrics sweep-bands \
        benchmark-start benchmark-dry-run benchmark-validate benchmark-check logs-benchmark benchmark-status benchmark-stop

# Output CSV for extract-metrics (one row appended per training run)
METRICS_CSV ?= results/test_metrics.csv

help:
	@echo "Available targets:"
	@echo "  build               Build all images from $(COMPOSE_FILE)"
	@echo "  up                  Start all services in detached mode"
	@echo "  stop                Stop all running services"
	@echo "  restart             Restart all services"
	@echo "  ps / status         Show services status"
	@echo "  logs                Show logs for all services"
	@echo "  logs-follow         Follow logs for all services"
	@echo "  logs-service        Show logs for one service (SERVICE=...)"
	@echo "  logs-follow-service Follow logs for one service (SERVICE=...)"
	@echo "  train               Run default training service once (SERVICE_DEFAULT)"
	@echo "  extract-metrics     Parse SERVICE logs (test metrics + best ckpt) into METRICS_CSV"
	@echo "  sweep-bands         Train sequentially on all 5 band-combo configs, log results to CSV"
	@echo "  orchestrator-start  Start orchestrator service in detached mode"
	@echo "  benchmark-start     Model benchmark: single pass in background (BENCHMARK_ARGS=...)"
	@echo "  benchmark-dry-run   Model benchmark: export CSV + override only (foreground, no DB write)"
	@echo "  benchmark-validate  Validate configs/benchmark (VALIDATE_ARGS=$(VALIDATE_ARGS))"
	@echo "  benchmark-check     Check benchmark outputs: layout, rasters, cross-model (CHECK_ARGS=...)"
	@echo "  logs-benchmark      Follow the model benchmark logs"
	@echo "  benchmark-status    Show the model benchmark container state / exit code"
	@echo "  benchmark-stop      Stop + remove the model benchmark container"
	@echo "  down                Stop and remove containers/networks"
	@echo "  cleanup             down + remove orphan containers and volumes"
	@echo ""
	@echo "Variables:"
	@echo "  COMPOSE_FILE=$(COMPOSE_FILE)"
	@echo "  SERVICE_DEFAULT=$(SERVICE_DEFAULT)"
	@echo "  SERVICE_ORCH=$(SERVICE_ORCH)"
	@echo "  SERVICE_BENCH=$(SERVICE_BENCH)"
	@echo "  BENCHMARK_ARGS=$(BENCHMARK_ARGS)"
	@echo "  SERVICE=$(SERVICE)"
	@echo "  TRAIN_CMD=$(TRAIN_CMD)"
	@echo "  METRICS_CSV=$(METRICS_CSV)"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

stop:
	$(COMPOSE) stop

restart:
	$(COMPOSE) restart

ps status:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs --tail=200

logs-follow:
	$(COMPOSE) logs -f --tail=200

logs-service:
	$(COMPOSE) logs --tail=200 $(SERVICE)

logs-follow-service:
	$(COMPOSE) logs -f --tail=200 $(SERVICE)

train:
	$(COMPOSE) up -d --build $(SERVICE_DEFAULT)

logs-train:
	$(COMPOSE) logs -f --tail=200 $(SERVICE_DEFAULT)



orchestrator-start:
	$(COMPOSE) up -d --build $(SERVICE_ORCH)

# Extract test metrics + best checkpoint path from the last run's logs into a CSV.
# Usage: make extract-metrics [SERVICE=geo-deep-learning] [METRICS_CSV=results/test_metrics.csv]
extract-metrics:
	$(COMPOSE) logs --no-color $(SERVICE) | python3 scripts/extract_test_metrics.py - --csv $(METRICS_CSV)

# Train sequentially on every band-combo config, one after another, appending
# each run's test metrics + best checkpoint path to results/band_sweep_metrics.csv.
sweep-bands:
	bash scripts/run_band_sweep.sh

log-sweep-bands:
	tail -f logs/band_sweep/sweep.log

log-all-sweep-bands:
	tail -f logs/band_sweep/*.log

kill-sweep-bands:
	kill "$(cat run/band_sweep.pid)"

logs-orchestrator:
	$(COMPOSE) logs -f --tail=200 $(SERVICE_ORCH)

# ---- Model benchmark (docs/dev-plans/2026-10-08_model_benchmark_predict.md) ----
# Single pass then the container exits (restart: "no"); runs in background so a
# long benchmark survives the SSH session. Exit code: make benchmark-status.
benchmark-start:
	BENCHMARK_ARGS="$(BENCHMARK_ARGS)" $(COMPOSE_BENCH) up -d --build --force-recreate $(SERVICE_BENCH)
	@echo "Benchmark started - follow with: make logs-benchmark ; exit code: make benchmark-status"

# Foreground, removed afterwards: CSV + override + snapshot, no DB write, no predict.
benchmark-dry-run:
	$(COMPOSE_BENCH) run --rm --build $(SERVICE_BENCH) --dry-run $(BENCHMARK_ARGS)

# Registry + configs + checkpoints (+ LightningCLI parsing + strict weight loading on CPU).
benchmark-validate:
	$(COMPOSE_BENCH) run --rm --build --entrypoint python $(SERVICE_BENCH) \
		/app/scripts/validate_benchmark_configs.py $(VALIDATE_ARGS)

# Read-only check of the outputs (P8 recette): layout, _prob / per-pair merges, filters,
# exported pairs produced, raster dtype/nodata/values, same names + grid across models.
benchmark-check:
	$(COMPOSE_BENCH) run --rm --build --entrypoint python $(SERVICE_BENCH) \
		/app/scripts/check_benchmark_outputs.py $(CHECK_ARGS)

logs-benchmark:
	$(COMPOSE_BENCH) logs -f --tail=200 $(SERVICE_BENCH)

benchmark-status:
	@docker inspect --format '{{.Name}}: {{.State.Status}} (exit code {{.State.ExitCode}}, finished {{.State.FinishedAt}})' \
		$(SERVICE_BENCH) 2>/dev/null || echo "$(SERVICE_BENCH): no container"

benchmark-stop:
	$(COMPOSE_BENCH) rm --stop --force $(SERVICE_BENCH)

down:
	$(COMPOSE) down

cleanup:
	$(COMPOSE) down --remove-orphans --volumes
