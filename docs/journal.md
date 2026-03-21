# lapulga-llm: Journal

## 2026-03-21
- **Goal:** Move from architectural math to functional training loop in MLX.
- **Action:** Created `train_mlx.py`. Defined a 7.8M parameter fp16 architecture to ensure it immediately fits under 15.9MB limit. Added logic for BPE tokenization, GQA, and AdamW. Checkpointing implemented.
- **Next steps:** Test the training script on small dataset and measure actual time to convergence and memory usage to ensure it adheres to the 10-minute cap.
