# lapulga-llm: Benchmarks

| Date | Settings (Layers, Dim, Heads, KV, Vocab) | Params | Prec. | Size (MB) | Perplexity | Notes |
|------------|------------------------------------------|--------|-------|-----------|------------|-------|
| 2026-03-21 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | ~7.8M | fp16 | ~15.6 | N/A | Initial baseline math estimate. |
| 2026-03-21 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | 7,802,112 | fp16 | 14.89 | N/A | **Empirical Run (M4/MLX v0.1):** Perfect weight tying (no direct nn.Linear block for outputs). Step time: ~0.56s (b=8, ctx=256). Proves constraint viability. |
