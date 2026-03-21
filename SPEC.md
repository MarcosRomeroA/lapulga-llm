---
model_name: lapulga-llm
architecture: decoder-only-transformer
framework: pytorch-2.x
compute: rtx-3090-cuda-12.1
n_layers: 6
n_embd: 256
n_head: 8
n_kv_heads: 2
hidden_dim: 1024
vocab_size: 8192
context_length: 256
constraints:
  max_size_mib: 16.0
  target_params: 7802112
  tolerance_pct: 1.0
tokenizer: bpe-custom-tinystories
precision_training: fp32
precision_export: fp16
batch_size: 256
---

# SPEC.md — lapulga-llm Technical Specification

This file is the **single source of truth** for the La Pulga model architecture.
All code must conform to this specification. Compliance is enforced automatically by
`tests/test_spec_compliance.py`, which is a hard gate — no code change is accepted if it fails.

## Architecture

| Parameter | Value | Rationale |
|:---|:---|:---|
| `architecture` | Decoder-only Transformer | Generative task: next-token prediction |
| `framework` | PyTorch 2.x | CUDA-native framework for RTX 3090 / H100 |
| `compute` | RTX 3090 (CUDA 12.1) | 24 GB VRAM, identical pipeline to H100 target |
| `n_layers` | 6 | Depth vs. parameter budget balance |
| `n_embd` | 256 | Embedding dimension — keeps budget under 16 MiB |
| `n_head` | 8 | Query heads for multi-head attention |
| `n_kv_heads` | 2 | GQA compression: 4:1 ratio (KV heads << Q heads) |
| `hidden_dim` | 1024 | MLP inner size = 4 x n_embd |
| `vocab_size` | 8192 | Custom BPE trained on TinyStories — zero token corruption |
| `context_length` | 256 | Context window for training |
| `norm_eps` | 1e-5 | RMSNorm stability epsilon |
| `batch_size` | 256 | Increased from 32 — RTX 3090 VRAM allows high throughput |

## Constraints

| Constraint | Value |
|:---|:---|
| Max artifact size | **16.0 MiB** (hard limit — set by Parameter Golf rules) |
| Target parameter count | **7,802,112** (1% tolerance enforced in CI) |
| Max training time | **10 minutes** on 8x H100 |
| Precision (training) | FP32 (stability), with AMP (torch.cuda.amp) for throughput |
| Precision (export) | FP16 (~14.89 MiB) |
| Size validation | PyTorch `state_dict` serialized size via `torch.save` |

## Parameter Budget Breakdown

| Component | Params | Size (FP16) |
|:---|:---|:---|
| Embeddings (8192 x 256, tied) | 2,097,152 | 4.00 MiB |
| Attention x 6 (GQA kv=2) | 983,040 | 1.88 MiB |
| MLP x 6 (SwiGLU, dim=1024) | 4,718,592 | 9.00 MiB |
| RMSNorm x 13 | 3,328 | 0.01 MiB |
| **Total** | **7,802,112** | **14.89 MiB** |

## Compliance Gate

```
uv run pytest tests/test_spec_compliance.py -v
```

This test **must pass** before any commit that touches `src/model/` or `src/domain/config.py`.
