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

# Generic service selector for per-service targets
SERVICE ?= $(SERVICE_DEFAULT)

# Training mode used by entrypoint (fit/validate/test/predict...)
TRAIN_CMD ?= fit

.PHONY: help build up stop restart ps status logs logs-follow logs-service logs-follow-service \
        train train-cs2 orchestrator-start down cleanup extract-metrics sweep-bands

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
	@echo "  down                Stop and remove containers/networks"
	@echo "  cleanup             down + remove orphan containers and volumes"
	@echo ""
	@echo "Variables:"
	@echo "  COMPOSE_FILE=$(COMPOSE_FILE)"
	@echo "  SERVICE_DEFAULT=$(SERVICE_DEFAULT)"
	@echo "  SERVICE_ORCH=$(SERVICE_ORCH)"
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

logs-orchestrator:
	$(COMPOSE) logs -f --tail=200 $(SERVICE_ORCH)

down:
	$(COMPOSE) down

cleanup:
	$(COMPOSE) down --remove-orphans --volumes
