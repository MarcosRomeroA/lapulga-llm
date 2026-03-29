# lapulga-llm Makefile
# Usage: make <target>

ifneq (,$(wildcard ./.env))
    include .env
    export
endif

SHARDS ?= 10
PROMPT ?= "Once upon a time"
MAX_TOKENS ?= 100
TEMPERATURE ?= 0.8
TOP_K ?= 50
REP_PENALTY ?= 1.2

.PHONY: run generate test test-scoring download-data run-cloud help

## Run the full training + BPB evaluation pipeline
run:
	PYTHONUNBUFFERED=1 python main.py

## Generate text from saved checkpoint
generate:
	PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/infer.py --prompt "$(PROMPT)" --max-tokens $(MAX_TOKENS) --temperature $(TEMPERATURE) --top-k $(TOP_K) --repetition-penalty $(REP_PENALTY)

## Run all tests (spec compliance + official scoring)
test:
	python -m pytest tests/ -v

## Run only the official scoring mock tests
test-scoring:
	python -m pytest tests/test_official_scoring.py -v

## Download FineWeb shards (default: 10 train shards)
download-data:
	python ../parameter-golf/data/cached_challenge_fineweb.py --variant sp1024

## Start RunPod, train, download artifacts, stop Pod (zero-click)
run-cloud:
	@PYTHONUNBUFFERED=1 uv run scripts/runpod_orchestrator.py

## Show available commands
help:
	@echo ""
	@echo "lapulga-llm — Available Commands"
	@echo "---------------------------------"
	@echo "  make run              Train on FineWeb + evaluate BPB"
	@echo "  make generate         Generate text from saved checkpoint"
	@echo "  make generate PROMPT=\"The dog\" MAX_TOKENS=150 TOP_K=50 REP_PENALTY=1.2"
	@echo "  make test             Run all tests (compliance + scoring)"
	@echo "  make test-scoring     Run official scoring mock tests only"
	@echo "  make download-data    Download FineWeb cache via ../parameter-golf"
	@echo "  (uses default settings from parameter-golf cached_challenge_fineweb.py)"
	@echo "  make run-cloud        Start RunPod, train, download artifacts, stop Pod"
	@echo ""
