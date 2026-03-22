# lapulga-llm: Benchmarks

| Date/Time | Settings (Layers, Dim, Heads, KV, Vocab) | Params (M) | Prec. | Size (MB) | Perplexity | BPB | Notes |
|-----------|------------------------------------------|------------|-------|-----------|------------|-----|-------|
| 2026-03-21 00:40 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | ~7.8M | fp16 | ~15.6 | N/A | — | Initial baseline math estimate. |
| 2026-03-21 01:28 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | ~7.8M | fp16 | 14.89 | N/A | — | **Empirical Run (M4/MLX v0.1):** Perfect weight tying. Step time: ~0.56s (b=8, ctx=256). |
| 2026-03-21 02:15 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | ~7.8M | fp16 | 14.89 | **72.34** | — | **Custom BPE Tokenizer (v0.2):** 250k train, 50k val, 1 epoch. Loss: 9.40→4.40. |
| 2026-03-21 02:40 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | ~7.8M | fp16 | 14.89 | **14.05** | — | **Extended Training (v0.3, MLX/M4):** 2M train, 3 epochs. Loss: 4.11→3.38→2.93. 🏆 |
| 2026-03-21 16:39 | L=6, D=256, H=8, KV=2, V=8192, MLP=1024 | ~7.8M | fp16 | 14.90 | **17.6M** | — | **PyTorch/CUDA (v1.0 — BROKEN INIT):** Loss: 82.5→23.4→18.5. N(0,1) init bug. |
| 2026-03-21 (v2.0) | L=9, D=512, H=8, KV=4, V=1024, MLP=1024 | ~17.0M | int8+zlib | ~12 (est) | — | TBD | **Parameter-Golf Alignment:** relu^2, U-Net skips, softcap=30, SDPA. FineWeb/sp1024. 14/14 tests green. |
| 2026-03-21 17:38 | L=9, D=512, H=8, KV=4, V=1024, MLP=1024 | ~17.0M | int8+zlib | 5.15 | — | **3.6595** | **First FineWeb Run (v2.1, RTX 3090):** 2M tokens, 3 epochs, 9 steps. Loss: 9.87→6.30. ~128K tok/s. Grad accum 32×16K. |
| 2026-03-21 20:47 | L=9, D=512, H=8, KV=4, V=1024, MLP=1024 | ~17.0M | int8+zlib | 8.51 | — | **1.7429** | **200M Token Run (v2.2, RTX 3090):** 200M tokens, 3 epochs, 1143 steps. Loss: 9.86→4.87→3.44→3.06. ~78 min. Grad accum 32×16K. Texto coherente. |
| 2026-03-21 22:10 | L=12, D=512, H=8, KV=4, V=1024, MLP=1536 | ~28.9M | int8+zlib | 9.44 | — | **3.5511** | **Scaled Architecture DEV_MODE (v3.0, RTX 3090):** 100 steps only. torch.compile (~3s/step vs 4s). Warmup+Cosine LR 6e-4. Artifact under 16MB. Validation run. |
| 2026-03-22 06:07 | L=12 eff (4×3 ALBERT), D=768, H=12, KV=4, V=1024, MLP=3072 | 25.97M | int8+zlib | 9.32 | — | **2.0680** | **Full RunPod Run (v4.0, RunPod CUDA):** 500M tokens, 1 epoch, 953 steps. Loss: 14.09→3.49. ~1.57s/step. LR: 6e-4 warmup+cosine. Artifact 9.32MB. |
