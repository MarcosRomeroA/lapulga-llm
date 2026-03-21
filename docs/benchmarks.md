# lapulga-llm: Benchmarks

| Date/Time | Settings (Layers, Dim, Heads, KV, Vocab) | Params (M) | Prec. | Size (MB) | Perplexity | Notes |
|-----------|------------------------------------------|------------|-------|-----------|------------|-------|
| 2026-03-21 00:40 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | ~7.8M | fp16 | ~15.6 | N/A | Initial baseline math estimate. |
| 2026-03-21 01:28 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | ~7.8M | fp16 | 14.89 | N/A | **Empirical Run (M4/MLX v0.1):** Perfect weight tying (no direct nn.Linear block for outputs). Step time: ~0.56s (b=8, ctx=256). Proves constraint viability. |
| 2026-03-21 02:15 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | ~7.8M | fp16 | 14.89 | **72.34** | **Custom BPE Tokenizer (v0.2):** 250k train tokens, 50k val tokens, 1 epoch. Loss: 9.40→4.40. Step time: ~0.107s. 🎯 First official perplexity baseline. |
| 2026-03-21 02:40 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | ~7.8M | fp16 | 14.89 | **14.05** | **Extended Training (v0.3, MLX/M4):** 2M train tokens, 3 epochs. Avg Loss: 4.11→3.38→2.93. La Pulga generates coherent stories. Massive perplexity drop: 72→14. 🏆 |
| 2026-03-21 16:39 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | ~7.8M | fp16 | 14.90 | **17,612,076** | **PyTorch/CUDA Migration Baseline (v1.0 — BROKEN INIT):** 2M train tokens, 3 epochs, batch=256, AMP. Avg Loss: 82.5→23.4→18.5. Generates repetitive tokens. Root cause: `nn.Embedding` default N(0,1) init (MLX used N(0,1/√dim)). Logits 16× too large at init. Step time: ~0.24s. |
