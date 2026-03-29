---
model_name: lapulga-llm
architecture: decoder-only-transformer
framework: pytorch-2.x
compute: rtx-3090-cuda-12.1
n_embd: 768
n_head: 12
n_kv_heads: 4
physical_layers: 4
repeat_count: 1
hidden_dim: 4096
vocab_size: 1024
context_length: 1024
constraints:
  max_size_bytes: 16000000
  target_params: 32257584
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
| `n_embd` | 768 | H100 228KB L1 cache fits fused RMSNorm backward at this width |
| `n_head` | 12 | Query heads (head_dim = 64) |
| `n_kv_heads` | 4 | GQA compression: 3:1 ratio |
| `physical_layers` | 4 | Unique layer blocks stored on disk |
| `repeat_count` | 3 | Times the block is traversed during forward pass |
| `effective_layers` | 12 | 4 physical × 3 repeats — effective depth |
| `hidden_dim` | 1280 | MLP inner size selected to keep int6+zlib artifact safely under 16,000,000 bytes |
| `vocab_size` | 1024 | SentencePiece sp1024 — official challenge tokenizer |
| `context_length` | 1024 | Official training sequence length |
| `norm_eps` | 1e-5 | RMSNorm stability epsilon |
| `logit_softcap` | 30.0 | Tanh softcap prevents logit explosion |
| `batch_size` | 524,288 tokens | Global token-based batching (official default) |
| `scoring_metric` | BPB (Bits Per Byte) | Tokenizer-agnostic compression on FineWeb val set |
| `dataset` | FineWeb (sp1024 shards) | Official challenge dataset |

## Fat ALBERT — Cross-Layer Parameter Sharing on H100

Instead of 12 distinct blocks, La Pulga stores only **4 physical blocks** and routes the forward
pass through them **3 times**. This is the ALBERT strategy adapted for Parameter Golf.

**Why this works on H100 (not RTX 3090):**
The H100 has **228 KB of L1 Shared Memory per SM** — more than double the RTX 3090's 101 KB.
Triton's fused RMSNorm backward kernel at `dim=768` requires ~170 KB, which fits comfortably
on H100 but caused `OutOfMemoryError` locally.

**Budget math:**
- Stored model parameters: 14,959,152 (10 distinct physical layers across 10 effective steps)
- 12 effective transformer steps maintain full representational depth
- U-Net skip weights indexed over virtual (effective) layer indices 0–11

## Constraints

| Constraint | Value |
|:---|:---|
| Max artifact size | **16,000,000 bytes** (decimal) = code bytes + zlib(int8 model) |
| Target parameter count | **14,959,152** (1% tolerance enforced in CI) |
| Target BPB | **1.22** (official baseline: 1.2244) |
| Max training time | **10 minutes** on 8x H100 |
| Precision (training) | FP32 with AMP (torch.cuda.amp) |
| Precision (export) | Int8 + zlib level 9 compression |
| Size validation | Int8 quantized state_dict + zlib + code file bytes |

## Parameter Budget Breakdown

| Component | Params | Notes |
|:---|:---|:---|
| Embeddings | ~0.5M | Factored Tied Embeddings (254 dim) |
| Attention × 4 physical (GQA kv=4) | ~6.3M | wq/wk/wv/wo + q_gain with ALBERT sharing |
| MLP × 4 physical (LeakyReLU, hidden=2048) | ~12.5M | fc + proj with ALBERT sharing |
| Scales + residual mix × 4 physical | ~0.01M | attn_scale, mlp_scale, resid_mix |
| Skip weights (6 × 768) | ~0.004M | U-Net over 12 layer indices |
| **Total stored** | **~29.5M (logical) -> ~15.9MB (quantized)** | Export size strictly constrained to <16MB |

## Compliance Gate

```
python -m unittest tests.test_spec_compliance -v
```

This test **must pass** before any commit that touches `src/model/` or `src/domain/config.py`.
