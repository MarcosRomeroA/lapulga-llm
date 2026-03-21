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
