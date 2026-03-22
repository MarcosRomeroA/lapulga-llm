---
model_name: lapulga-llm
architecture: decoder-only-transformer
framework: pytorch-2.x
compute: rtx-3090-cuda-12.1
n_layers: 12
n_embd: 512
n_head: 8
n_kv_heads: 4
hidden_dim: 1536
vocab_size: 1024
context_length: 1024
constraints:
  max_size_bytes: 16000000
  target_params: 28876384
  tolerance_pct: 1.0
tokenizer: sentencepiece-sp1024
dataset: fineweb-sp1024
scoring_metric: bpb
target_bpb: 1.22
precision_training: fp32
precision_export: int8-zlib
batch_size: 524288
logit_softcap: 30.0
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
| `n_layers` | 12 | Deeper model for better BPB at ~29M param budget |
| `n_embd` | 512 | Official baseline width |
| `n_head` | 8 | Query heads for multi-head attention |
| `n_kv_heads` | 4 | GQA compression: 2:1 ratio (official baseline) |
| `hidden_dim` | 1536 | MLP inner size = 3 x n_embd (relu^2, 2 matrices) |
| `vocab_size` | 1024 | SentencePiece sp1024 — official challenge tokenizer |
| `context_length` | 1024 | Official training sequence length |
| `norm_eps` | 1e-5 | RMSNorm stability epsilon |
| `logit_softcap` | 30.0 | Tanh softcap prevents logit explosion |
| `batch_size` | 524,288 tokens | Global token-based batching (official default) |
| `scoring_metric` | BPB (Bits Per Byte) | Tokenizer-agnostic compression on FineWeb val set |
| `dataset` | FineWeb (sp1024 shards) | Official challenge dataset |

## Constraints

| Constraint | Value |
|:---|:---|
| Max artifact size | **16,000,000 bytes** (decimal) = code bytes + zlib(int8 model) |
| Target parameter count | **28,876,384** (1% tolerance enforced in CI) |
| Target BPB | **1.22** (official baseline: 1.2244) |
| Max training time | **10 minutes** on 8x H100 |
| Precision (training) | FP32 with AMP (torch.cuda.amp) |
| Precision (export) | Int8 + zlib level 9 compression |
| Size validation | Int8 quantized state_dict + zlib + code file bytes |

## Parameter Budget Breakdown

| Component | Params | Int8 Bytes |
|:---|:---|:---|
| Embeddings (1024 x 512, tied) | 524,288 | ~0.5 MB |
| Attention x 12 (GQA kv=4) | 9,437,184 | ~9.4 MB |
| MLP x 12 (relu^2, hidden=1536) | 18,874,368 | ~9.4 MB (after zlib) |
| RMSNorm x 25 + scales | ~40,544 | ~40 KB (fp16/fp32) |
| **Total** | **~28,876,384** | **~14.5 MB after zlib** |

## Compliance Gate

```
uv run pytest tests/test_spec_compliance.py -v
```

This test **must pass** before any commit that touches `src/model/` or `src/domain/config.py`.
