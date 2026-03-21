# lapulga-llm: Project Architecture & Technical Standards

## The "La Pulga" Philosophy
Efficiency is king. Every line of code written in this project must prioritize:
1. **Memory Management:** Extreme diligence to strictly stay under the critical 16MB artifact boundary.
2. **Compute Speed:** Zero overhead abstraction to ensure the model can train from scratch in under the 10-minute time constraint on 8xH100s.

## Modular Design
To prevent spaghetti code, the repository must be strictly organized into the following logical modules:
- **/src/data**: Data loading, dataset streaming, and Tokenization pipelines.
- **/src/model**: Transformer block architecture (PyTorch) and Weight initialization.
- **/src/training**: Optimization loops (AdamW), gradient updates, and Loss/Perplexity tracking.
- **/src/utils**: Weight quantization routines, file size calculators, and benchmarking scripts.

## Clean Architecture (Separation of Concerns)
Beyond just writing readable syntax, the project strictly enforces **Clean Architecture** patterns:
- **Domain Entities (Inner Ring):** Business rules and definition schemas (e.g., `ModelConfig`) must be pure, vanilla Python `dataclasses`. They must **NOT** import or depend on frameworks like `torch`.
- **Dependency Inversion:** Outer rings (Frameworks, CLIs, Tensors) must depend on the inner rings (Configs, Abstract Interfaces), never the other way around.
- **Framework Isolation:** PyTorch/CUDA tensor operations must be strictly isolated to their own specific modules (e.g., `/src/model/transformer.py`). The core training orchestrator should ideally not care if it's feeding tensors to a consumer GPU or an H100.

## CUDA Memory Management
- **Device Management:** A global `DEVICE` constant (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) in `src/model/transformer.py` ensures all operations target the GPU when available.
- **Batch Transfer:** Data batches are kept on CPU during preparation and transferred to GPU on-the-fly via `.to(device)` in `get_batches()`. This avoids pinning the entire dataset in VRAM.
- **FP16 Export:** The model trains in FP32 for numerical stability but exports `state_dict` weights in FP16 via `v.half()` to meet the 16 MiB artifact budget.
- **RTX 3090 (24GB VRAM):** With batch_size=256 and context_length=256, the model + activations + optimizer state fit comfortably in 24GB. This pipeline is identical to the H100 target.

## Mixed Precision Training (torch.cuda.amp)
- **GradScaler + autocast:** The training loop uses `torch.cuda.amp.GradScaler` and `autocast` to automatically run eligible operations in FP16 while keeping master weights in FP32.
- **Why AMP?** On Ampere+ GPUs (RTX 3090, H100), Tensor Cores accelerate FP16 matmuls by ~2x. AMP gives this speedup without manual dtype management.
- **Matmul Precision:** `torch.set_float32_matmul_precision("high")` can be enabled to allow TF32 on Ampere GPUs for additional throughput with negligible accuracy impact.
- **Loss Upcast:** Logits are always upcast to FP32 before `cross_entropy` to prevent NaN from FP16 overflow in the softmax exponential.

## Clean Code Standards
- **PEP8 Compliance:** All Python code must respect standard PEP8 styling.
- **Descriptive Naming:** Variables must be highly semantic and descriptive. Generic variable names like `x`, `y`, or `data` are **strictly prohibited** (unless tracking standard mathematical tensors where $x$ and $y$ are universally accepted, though `input_tensor` and `target_tokens` are always preferred).
- **Single Responsibility Principle (SRP):** Every function or class must execute exactly one logical task.

## Type Hinting
- **Strict Typing:** Python type hints (e.g., `def forward(self, input_ids: torch.Tensor) -> torch.Tensor:`) must be enforced for all function signatures.

## Documentation
- **Docstrings:** Every module, class, and function must include a clear docstring.
- **Explain the "What" and "Why":** The docstring must explicitly explain *What* the code does, and exactly *Why* it exists in the codebase.

---
**Agent Directive:** The AI assistant (*Antigravity*) acting on this repository is strictly commanded to **evaluate and reject** any code suggestion or draft that fails to adhere to these modular, type-hinted, and clean-code standards. All future code must be refactored to align with this file before being proposed.

---

## Validation Pipeline (Spec-Driven Development)

All architectural decisions are encoded in **`SPEC.md`**, which serves as the machine-readable single source of truth. The compliance gate is enforced via:

```bash
uv run pytest tests/test_spec_compliance.py -v
```

### Gate Rules — No PR or code change that touches `src/model/` or `src/domain/config.py` is accepted unless ALL of the following pass:

| Test | What it validates |
|:---|:---|
| `test_n_layers_matches_spec` | Transformer block count == `SPEC.n_layers` |
| `test_embedding_dim_matches_spec` | Embedding dim == `SPEC.n_embd` |
| `test_vocab_size_matches_spec` | Vocab size == `SPEC.vocab_size` |
| `test_attention_heads_match_spec` | Query heads == `SPEC.n_head` in every layer |
| `test_param_count_within_tolerance` | Total params <= `target_params` x (1 + 1%) |
| `test_fp16_artifact_within_16mib` | `state_dict` params x 2 bytes < 16 MiB |
| `test_int4_quantization_headroom` | `total_params x 0.5` bytes < 16 MiB |
| `test_model_moves_to_gpu` | Model device matches CUDA when available |

### SDD Workflow
1. **Edit `SPEC.md` first.** The YAML block is the contract.
2. **Update `src/domain/config.py`** to reflect the new spec values.
3. **Run the compliance suite.** If it fails, the architecture change is rejected.
4. **Commit only when green.**
