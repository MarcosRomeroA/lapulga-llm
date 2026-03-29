---
model_name: lapulga-llm
architecture: decoder-only-transformer
framework: pytorch-2.x
compute: runpod-8xh100-cuda-12.x
n_embd: 512
n_head: 8
n_kv_heads: 4
physical_layers: 10
repeat_count: 1
hidden_dim: 1536
vocab_size: 1024
context_length: 1024
constraints:
  max_size_bytes: 16000000
  target_params: 24120478
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
| 
| framework | PyTorch 2.x | CUDA-native framework for 8x H100 production training |
| compute | Runpod 8x H100 (CUDA 12.x) | Official production training environment |
| `n_embd` | 512 | Width tuned for 10 physical layers within 16MB export budget |
| `n_head` | 8 | Query heads (head_dim = 64) |
| `n_kv_heads` | 4 | GQA compression: 3:1 ratio |
| `physical_layers` | 10 | Unique banked layer slices stored on disk |
| `repeat_count` | 1 | No ALBERT-style repetition |
| `effective_layers` | 10 | 10 physical × 1 repeat |
| `hidden_dim` | 1536 | MLP inner size selected to keep int6+zlib artifact under 16MB |
| `vocab_size` | 1024 | SentencePiece sp1024 — official challenge tokenizer |
| `context_length` | 1024 | Official training sequence length |
| `norm_eps` | 1e-5 | RMSNorm stability epsilon |
| `logit_softcap` | 30.0 | Tanh softcap prevents logit explosion |
| `batch_size` | 524,288 tokens | Global token-based batching (official default) |
| `scoring_metric` | BPB (Bits Per Byte) | Tokenizer-agnostic compression on FineWeb val set |
| `dataset` | FineWeb (sp1024 shards) | Official challenge dataset |

## Parameter Banking + Parallel Muon

La Pulga stores **10 physical blocks** as **3D parameter banks** and runs each block once.
This layout reduces kernel launch overhead and enables batched Muon orthogonalization with `torch.bmm`.

**Why this works:**
- No ALBERT sharing overhead during training.
- Better GPU utilization from banked matmul paths.
- Muon can process all layer slices in parallel.

**Budget math:**
- Stored model parameters: ~24,120,478
- 10 effective transformer steps (10 physical × 1 repeat)
- Int6 export (stored as int8 + zlib) remains the artifact gate.

## Constraints

| Constraint | Value |
|:---|:---|
| Max artifact size | **16,000,000 bytes** (decimal) = code bytes + zlib(int8 model) |
| Target parameter count | **~24,120,478** (1% tolerance enforced in CI) |
| Target BPB | **1.22** (official baseline: 1.2244) |
| Max training time | **10 minutes** on 8x H100 |
| Precision (training) | FP32 with AMP (torch.cuda.amp) |
| Precision (export) | Int8 + zlib level 9 compression |
| Size validation | Int8 quantized state_dict + zlib + code file bytes |

## Parameter Budget Breakdown

| Component | Params | Notes |
|:---|:---|:---|
| Embeddings | ~0.5M | Factored Tied Embeddings (254 dim) |
| Attention × 10 physical (Banked, GQA kv=4) | ~7.3M | bank_qkv, bank_o, q_gain |
| MLP × 10 physical (LeakyReLU, hidden=1536) | ~15.7M | bank_fc, bank_proj |
| Scales + residual mix × 10 physical | ~0.03M | attn_scale, mlp_scale, resid_mix |
| Skip weights | 0 | Not used in banked architecture |
| **Total stored** | **~24.1M (logical) -> ~15.9MB (quantized)** | Export size strictly constrained to <16MB |

## Compliance Gate

```
python -m unittest tests.test_spec_compliance -v
```

This test **must pass** before any commit that touches `src/model/` or `src/domain/config.py`.

