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
