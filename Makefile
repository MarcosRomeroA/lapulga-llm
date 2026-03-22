# lapulga-llm Makefile
# Usage: make <target>

SHARDS ?= 10
PROMPT ?= "Once upon a time"
MAX_TOKENS ?= 100
TEMPERATURE ?= 0.8
TOP_K ?= 50
REP_PENALTY ?= 1.2

.PHONY: run generate test test-scoring download-data help

## Run the full training + BPB evaluation pipeline
run:
	PYTHONUNBUFFERED=1 uv run main.py

## Generate text from saved checkpoint
generate:
	PYTHONPATH=. PYTHONUNBUFFERED=1 uv run scripts/infer.py --prompt "$(PROMPT)" --max-tokens $(MAX_TOKENS) --temperature $(TEMPERATURE) --top-k $(TOP_K) --repetition-penalty $(REP_PENALTY)

## Run all tests (spec compliance + official scoring)
test:
	uv run pytest tests/ -v

## Run only the official scoring mock tests
test-scoring:
	uv run pytest tests/test_official_scoring.py -v

## Download FineWeb shards (default: 10 train shards)
download-data:
	uv run python data/download_fineweb.py --variant sp1024 --train-shards $(SHARDS)

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
	@echo "  make download-data    Download FineWeb shards (SHARDS=10)"
	@echo "  make download-data SHARDS=1   Download 1 shard (smoke test)"
	@echo ""
