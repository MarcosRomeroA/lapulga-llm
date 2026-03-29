# lapulga-llm: Project Architecture & Technical Standards

## The "La Pulga" Philosophy
Efficiency is king. Every line of code written in this project must prioritize:
1. **Memory Management:** Extreme diligence to strictly stay under the 16,000,000 byte artifact limit (code + int8+zlib model).
2. **Compute Speed:** Zero overhead abstraction to ensure the model can train from scratch in under the 10-minute time constraint on 8xH100s.

## Modular Design
To prevent spaghetti code, the repository must be strictly organized into the following logical modules:
- **/src/data**: Data loading (FineWeb shard loader, SentencePiece tokenizer), dataset streaming.
- **/src/model**: Transformer block architecture (PyTorch) and Weight initialization.
- **/src/training**: Optimization loops (AdamW/Muon), gradient updates, and Loss tracking.
- **/src/utils**: Int8 quantization, BPB evaluation, text generation, and benchmarking scripts.

## Clean Architecture (Separation of Concerns)
- **Domain Entities (Inner Ring):** Business rules and definition schemas (e.g., `ModelConfig`) must be pure, vanilla Python `dataclasses`. They must **NOT** import or depend on frameworks like `torch`.
- **Dependency Inversion:** Outer rings (Frameworks, CLIs, Tensors) must depend on the inner rings (Configs, Abstract Interfaces), never the other way around.
- **Framework Isolation:** PyTorch/CUDA tensor operations must be strictly isolated to their own specific modules (e.g., `/src/model/transformer.py`).

## CUDA Memory Management
- **Device Management:** A global `DEVICE` constant in `src/model/transformer.py` auto-detects CUDA availability.
- **Batch Transfer:** Data batches are kept on CPU during preparation and transferred to GPU on-the-fly via `.to(device)`.
- **Int8+zlib Export:** The model trains in FP32 (with AMP), then exports via int8 per-row quantization + zlib level 9 compression to meet the 16MB decimal artifact budget.
- **RTX 3090 (24GB VRAM):** Pipeline is identical to the 8xH100 target environment.

## Mixed Precision Training (torch.cuda.amp)
- **GradScaler + autocast:** Automatically runs eligible operations in FP16/BF16 while keeping master weights in FP32.
- **Logit Softcap:** `logit_softcap * tanh(logits / logit_softcap)` prevents logit explosion without gradient clipping. Set to 30.0 (official default).
- **Loss Upcast:** Logits are always upcast to FP32 before `cross_entropy`.

## Competitive Strategy vs Parameter-Golf Leaderboard

### Our Baseline Architecture (aligned with official baseline: 1.2244 BPB)
- **9 layers, 512 dim, 8 heads, 4 KV heads** — matches official baseline exactly
- **relu^2 MLP** — 2 weight matrices per layer instead of SwiGLU's 3, saving ~33% MLP params and allowing 9 layers
- **Weight Tying** — embedding shared as output head (saves 524K params)
- **U-Net Skip Connections** — encoder layers (0-3) feed into decoder layers (5-8) via learnable `skip_weights`, improving gradient flow for ~2K extra params
- **GQA (4 KV heads)** — saves 50% KV projection parameters vs full MHA
- **Logit Softcap (30.0)** — stabilizes training without gradient clipping
- **SDPA (F.scaled_dot_product_attention)** — enables Flash Attention automatically on CUDA

### Why relu^2 instead of SwiGLU?
SwiGLU uses 3 matrices per MLP layer (w1, w2, w3 with SiLU gate). At dim=512 with 9 layers, that would be 9 x 3 x 512 x 1024 = 14.16M MLP params alone — exceeding our total budget. relu^2 uses 2 matrices (fc + proj) for 9 x 2 x 512 x 1024 = 9.44M MLP params. Every top Parameter-Golf entry uses relu^2.

### Techniques Roadmap (by priority)
1. **MLP 3x expansion** (hidden=1536) — top 3 all use it, requires int5/int6 quantization
2. **BigramHash** — hash consecutive token pairs into embedding table for n-gram awareness
3. **SmearGate** — gating mechanism on attention/MLP outputs
4. **Muon Optimizer** — Newton-Schulz orthogonalization for matrix-shaped params
5. **Int6 QAT** — quantization-aware training for tighter compression
6. **Sliding Window Evaluation** — stride < seq_len for better BPB during eval
7. **SWA (Stochastic Weight Averaging)** — average late-training checkpoints

## Clean Code Standards
- **PEP8 Compliance:** All Python code must respect standard PEP8 styling.
- **Descriptive Naming:** Variables must be highly semantic and descriptive.
- **Single Responsibility Principle (SRP):** Every function or class must execute exactly one logical task.

## Type Hinting
- **Strict Typing:** Python type hints must be enforced for all function signatures.

## Documentation
- **Docstrings:** Every module, class, and function must include a clear docstring.
- **Explain the "What" and "Why":** The docstring must explicitly explain *What* the code does, and exactly *Why* it exists.

---
**Agent Directive:** The AI assistant (*Antigravity*) acting on this repository is strictly commanded to **evaluate and reject** any code suggestion or draft that fails to adhere to these modular, type-hinted, and clean-code standards.

---

## Validation Pipeline (Spec-Driven Development)

All architectural decisions are encoded in **`SPEC.md`**, which serves as the machine-readable single source of truth. The compliance gate is enforced via:

```bash
python -m unittest tests.test_spec_compliance -v
python -m unittest tests.test_official_scoring -v
```

### Gate Rules — No PR or code change that touches `src/model/` or `src/domain/config.py` is accepted unless ALL pass:

| Test | What it validates |
|:---|:---|
| `test_n_layers_matches_spec` | Transformer block count == `SPEC.n_layers` |
| `test_embedding_dim_matches_spec` | Embedding dim == `SPEC.n_embd` |
| `test_vocab_size_matches_spec` | Vocab size == `SPEC.vocab_size` |
| `test_attention_heads_match_spec` | Query heads == `SPEC.n_head` in every layer |
| `test_param_count_within_tolerance` | Total params <= `target_params` x (1 + 1%) |
| `test_int8_zlib_artifact_under_16mb_decimal` | int8+zlib+code < 16,000,000 bytes |
| `test_int8_roundtrip_fidelity` | Quantize -> dequantize preserves shapes |
| `test_model_moves_to_gpu` | Model device matches CUDA when available |
| `test_mock_eval_pipeline` | Full quantize+compress+dequantize roundtrip |
| `test_submission_artifact_format` | Quantized object has correct keys |

### SDD Workflow
1. **Edit `SPEC.md` first.** The YAML block is the contract.
2. **Update `src/domain/config.py`** to reflect the new spec values.
3. **Run the compliance suite.** If it fails, the architecture change is rejected.
4. **Commit only when green.**
