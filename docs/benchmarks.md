# lapulga-llm: Benchmarks

| Date | Settings (Layers, Dim, Heads, KV, Vocab) | Params | Prec. | Size (MB) | Perplexity | Notes |
|------------|------------------------------------------|--------|-------|-----------|------------|-------|
| 2026-03-21 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | ~7.8M | fp16 | ~15.6 | N/A | Initial baseline in MLX, unquantized. Fits <15.9MB directly in FP16. Tied embeddings used. |
