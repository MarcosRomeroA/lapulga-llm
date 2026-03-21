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
