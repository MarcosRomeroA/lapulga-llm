# lapulga-llm: Today I Learned (TIL)

## 2026-03-21: MLX Optimization & Architecture Gotchas

1. **Weight Tying Duplication in MLX:** 
   If you try to tie weights by assigning `self.output.weight = self.tok_embeddings.weight` within an `nn.Linear` layer, MLX's parameter tree (`utils.tree_flatten`) treats them as two distinct keys. This silently duplicates the parameter count and doubles the exported `.safetensors` size on disk! 
   *Solution:* Completely remove the output `nn.Linear` module and compute logits dynamically via cross-multiplication: `hidden_states @ self.tok_embeddings.weight.T`.

2. **Default Precision Bloat:** 
   MLX initializes `mx.ones` and model weights in `Float32` natively. Even if your parameter math (e.g., 7.8M params) says you should be at ~15.6 MB in 16-bit, MLX will export a ~37.7 MB artifact. 
   *Solution:* Explicitly force the model state before the loop using `mlx_model.update(utils.tree_map(lambda arr: arr.astype(mx.float16), mlx_model.parameters()))`.

3. **Additive Causal Mask Dtype Error:** 
   When generating the causal mask (`create_additive_causal_mask`), you **cannot** pass the `dtype` of the token inputs (which are `int32`), otherwise MLX crashes with `[finfo] dtype int32 is not inexact`. 
   *Solution:* The mask must match the precision of the floating-point weights (e.g., `self.tok_embeddings.weight.dtype`).

4. **Attention Transpose Shapes:** 
   Custom Grouped-Query Attention requires extreme care with dimension swapping. `Q` and `K` must be transposed from `[Batch, SeqLength, Heads, HeadDim]` to `[Batch, Heads, SeqLength, HeadDim]` **before** the dot product, so the resulting attention scores correctly broadcast to `[B, H, L, L]`.

5. **Mixed Precision Strategy (Train FP32, Export FP16):**
   Casting the entire model to FP16 *before* training causes AdamW's second-moment estimates to overflow (values squared exceed FP16 max of 65,504), resulting in NaN loss after ~10 batches. The correct approach: train entirely in FP32 for numerical stability, then cast to FP16 only at checkpoint export time via `utils.tree_map(lambda arr: arr.astype(mx.float16), model.parameters())`. This satisfies the 16MB artifact constraint without corrupting the optimization process.

6. **Custom BPE Tokenizer is Non-Negotiable:**
   Using tiktoken's `cl100k_base` (100k vocab) clamped to 8192 tokens corrupts **92%** of the corpus by mapping unknown tokens to ID 0 (`!`). Training a custom 8192-token BPE on the target dataset (TinyStories) ensures zero token corruption. Result: Loss dropped from 9.40 to 4.40 in a single epoch (250k tokens), and the model generated coherent English phrases immediately.

7. **Data Volume is the Fastest Lever for Perplexity:**
   Scaling from 250k tokens (1 epoch, PPL=72.34) to 2M tokens (3 epochs, PPL=14.05) reduced perplexity by **80%** with zero architectural changes. Before tuning attention heads or layer depth, always exhaust data scaling first — it's cheaper and faster on local M4 hardware.

## 2026-03-22: H100 NVL Hardware Constraints

8. **Triton Shared Memory Limit on H100 SM (232 KB):**
   FlashAttention's Triton kernel allocates shared memory proportional to sequence batch size. On H100, each SM has **232 KB of shared memory**. Running `micro_batch_size=131072` (128 seqs × 1024 tokens) exceeded this limit and triggered an OOM inside the Triton kernel — *not* in CUDA device memory, so typical VRAM monitoring won't catch it.
   *Solution:* Halve the micro-batch to **65536 tokens** (64 seqs × 1024). This fits the 232 KB budget and, with an unchanged effective batch of 524K tokens, raises `accum_steps` from 4 → 8 — still ~4× faster than the original RTX 3090 baseline.
   *Rule of thumb:* For FlashAttention on H100, keep sequences-per-micro-batch ≤ 64 at context length 1024.

9. **`hidden_dim` is the Dominant Budget Knob (GQA / ALBERT style):**
   With a fixed layer count and GQA head structure, increasing `hidden_dim` is the most efficient way to fill the 16MB weight budget. Going from 3072 → **3584** (a 16.7% increase) pushes params from ~25.9M → **~29.1M**, filling ~11 MB — two-thirds of the 16MB ceiling — without touching depth, vocab, or batch logic. The non-linear growth comes from the MLP (dimension scales as hidden_dim²). Always recalculate artifact size after changing `hidden_dim`.
