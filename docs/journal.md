# lapulga-llm: Journal

## 2026-03-21 (Initial Setup)
- **Goal:** Move from architectural math to functional training loop in MLX.
- **Action:** Created `train_mlx.py`. Defined a 7.8M parameter FP16 architecture to ensure it immediately fits under 15.9MB limit. Added logic for BPE tokenization, GQA, and AdamW. Checkpointing implemented.
- **Outcome:** Architecture validated mathematically. Next step: real training.

## 2026-03-21 (Refactor & Smoke Test)
- **Goal:** Enforce Clean Architecture. Eliminate monolithic `train_mlx.py`.
- **Action:** Restructured codebase into `src/domain`, `src/model`, `src/data`, `src/training`, `src/utils`. Fixed multiple MLX bugs: Weight Tying duplication, FP16 precision bloat, causal mask dtype, attention transpose shapes.
- **Outcome:** Smoke test green. Model: 7,802,112 params. Artifact: 14.89 MiB. Constraint respected.

## 2026-03-21 (Custom Tokenizer - v0.2)
- **Goal:** Eliminate the 92% token corruption caused by using tiktoken's 100k vocab clamped to 8,192.
- **Action:** Built `src/data/train_tokenizer.py`. Trained a custom BPE tokenizer over 100k TinyStories samples using HuggingFace `tokenizers`. Refactored `loader.py` and `generate.py` to use the new tokenizer. Removed all clamping hacks.
- **Outcome:** First coherent text generation. Loss: 9.40→4.40 (stable, no NaN). Baseline perplexity: **72.34**.

## 2026-03-21 02:40 (Extended Training - v0.3)
- **Goal:** Push perplexity below 20 before H100s arrive.
- **Action:** Scaled training from 250k to 2M tokens, 3 epochs. Wired in `src/utils/evaluate.py` for Validation Perplexity measurement using a 50k token held-out split.
- **Outcome:** Loss progression: 4.11 → 3.38 → 2.93. Validation Perplexity: **14.05**. La Pulga now generates grammatically correct and contextually coherent children's stories. 🏆
- **Next steps:** Await H100 credits (~2 days). In the meantime: experiment with LR scheduling (cosine decay + warmup) and gradient clipping to push perplexity into single digits on M4.

## 2026-03-21 16:39 (Infrastructure Migration — PyTorch/CUDA RTX 3090)
- **Goal:** Migrate from Apple Silicon/MLX to WSL2 + RTX 3090 + PyTorch. Make the local environment identical to the H100 target.
- **Action:** Full rewrite of `src/model/` (`mlx_backend.py` → `transformer.py`), `src/training/loop.py`, `src/data/loader.py`, `src/utils/evaluate.py`, `src/utils/generate.py`, and `main.py`. Added `torch.cuda.amp` (GradScaler + autocast), global `DEVICE` auto-detection, and proper `_init_weights()`. Batch size increased 8 → 256. Updated `pyproject.toml` for `uv` + PyTorch deps. Spec compliance tests rewritten with new `test_model_moves_to_gpu` gate.
- **Outcome:** Model trains and saves correctly at **14.90 MiB**. However, first run exposed a critical initialization bug: PyTorch `nn.Embedding` defaults to N(0,1) whereas MLX used N(0, 1/√dim), causing logits 16× too large → initial loss ~82 instead of expected ~9.01. Fixed via explicit `_init_weights()` using std=0.02 (nanoGPT standard) with residual projection scaling.
- **Next steps:** Re-run training with corrected init to confirm perplexity recovers to MLX baseline (~14).

## 2026-03-21 (Parameter-Golf Alignment — v2.0)
- **Goal:** Align fully with the official OpenAI Parameter-Golf challenge standards. Previous architecture (6L/256dim/8192vocab/SwiGLU/TinyStories) was built for prototyping; now we adopt the official dataset, tokenizer, scoring metric, model architecture, and quantization format.
- **Action:** Complete architecture overhaul:
  - **Dataset:** TinyStories -> FineWeb (binary shards via `shard_loader.py`)
  - **Tokenizer:** Custom BPE 8192 -> SentencePiece sp1024 (official)
  - **Model:** 6L/256dim/8192vocab/SwiGLU -> **9L/512dim/1024vocab/relu^2** + U-Net skips + logit softcap + SDPA
  - **Scoring:** Perplexity -> **BPB** (Bits Per Byte, tokenizer-agnostic)
  - **Size limit:** 16 MiB FP16 -> **16,000,000 bytes** decimal (code + int8 + zlib)
  - **Quantization:** FP16 export -> **Int8 per-row + zlib level 9** (official format)
  - New modules: `shard_loader.py`, `tokenizer.py`, `evaluate_bpb.py`, `quantize.py`, `test_official_scoring.py`
  - **Params:** 7.8M -> **17.0M** (fits in ~12 MB after int8+zlib)
- **Outcome:** 14/14 tests passing (9 compliance + 5 official scoring). Model instantiates, quantizes, compresses, and dequantizes correctly within 16MB limit. Ready for first FineWeb training run.
- **Next steps:** Download FineWeb shards (`data/download_fineweb.py --train-shards 10`), run first BPB training, target 1.22 BPB baseline.

## 2026-03-21 (FineWeb Pipeline + First Runs — v2.1/v2.2)
- **Goal:** Wire up end-to-end FineWeb pipeline and get first BPB measurements.
- **Action:** Rewrote `main.py`, `training/loop.py`, `generate.py`, `Makefile` for FineWeb. Added gradient accumulation (32×16K micro-batches) to fit RTX 3090. Added `scripts/infer.py` for checkpoint inference. Improved sampling with top-k + repetition penalty.
- **Outcome:** v2.1 (2M tokens): BPB 3.66, loss 9.87→6.30. v2.2 (200M tokens, 3 epochs): **BPB 1.7429**, loss 9.86→3.06. First coherent English text generation from FineWeb. Artifact 8.51MB.

## 2026-03-21 (Architecture Scaling + torch.compile — v3.0)
- **Goal:** Scale model to fill 16MB budget and add LR scheduler for better convergence. Target BPB ≤ 1.22.
- **Action:** Scaled from 9L/1024hidden (~17M) to **12L/1536hidden (~28.9M)**. Added warmup+cosine decay LR scheduler (peak 6e-4, 5% warmup). Integrated `torch.compile` for ~25% speedup (~3s/step vs 4s). Implemented `DEV_MODE` env var for fast 100-step validation runs.
- **Outcome:** DEV_MODE validation passed: no OOM, torch.compile works, scheduler warmup correct, artifact 9.44MB (under 16MB). Ready for full 500M token run (~48 min).
- **Next steps:** Full training run (`make run`), target BPB ≤ 1.22.
