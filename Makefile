# lapulga-llm Makefile
# Usage: make <target>

.PHONY: run tests train-tokenizer help

## Run the full training + evaluation pipeline
run:
	uv run main.py

## Run the spec compliance test suite
test:
	uv run pytest tests/test_spec_compliance.py -v

## Train the custom BPE tokenizer (run once before training)
train-tokenizer:
	uv run src/data/train_tokenizer.py

## Show available commands
help:
	@echo ""
	@echo "lapulga-llm — Available Commands"
	@echo "---------------------------------"
	@echo "  make run              Train + evaluate the model"
	@echo "  make test             Run spec compliance gate"
	@echo "  make train-tokenizer  Build custom BPE tokenizer"
	@echo ""
