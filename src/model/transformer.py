"""
PyTorch Implementation of La Pulga Transformer.
Aligned with the official OpenAI Parameter-Golf architecture.
Uses relu^2 MLP, U-Net skip connections, logit softcap, and SDPA.

Optimizations for H100:
  - CastedLinear: FP32 weights, BF16 compute (better optimizer precision)
  - Unrolled ALBERT loop: torch.compile sees a linear sequence, no graph breaks
  - Gradient checkpointing: ~70-80% VRAM savings, allows huge micro-batches
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional
from src.domain.config import ModelConfig

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CastedLinear(nn.Linear):
    """
    Keeps weights in FP32 for optimizer precision, casts to input dtype at matmul time.
    Under torch.autocast(bf16), the .to(x.dtype) is essentially free.
    Matches the official parameter-golf baseline pattern.
    """
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no learnable weight for compile compat)."""
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    """
    Rotary Position Embeddings with pre-computed cos/sin tables.
    Tables are computed once at init for the fixed context_length,
    stored as buffers (no mutation during forward — torch.compile safe).
    """
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos_table", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_table", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        return (
            self.cos_table[:, :, :seq_len, :].to(dtype=dtype),
            self.sin_table[:, :, :seq_len, :].to(dtype=dtype),
        )


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Applies rotary embeddings using the half-rotation formula."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class Attention(nn.Module):
    """
    Grouped-Query Attention with RoPE, QK-norm, and learnable q_gain.
    Uses F.scaled_dot_product_attention for Flash Attention on CUDA.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads: int = config.n_heads
        self.n_kv_heads: int = config.n_kv_heads
        self.head_dim: int = config.dim // config.n_heads
        kv_dim: int = self.n_kv_heads * self.head_dim

        self.wq = CastedLinear(config.dim, config.dim, bias=False)
        self.wk = CastedLinear(config.dim, kv_dim, bias=False)
        self.wv = CastedLinear(config.dim, kv_dim, bias=False)
        self.wo = CastedLinear(config.dim, config.dim, bias=False)
        self.wo._zero_init = True

        self.q_gain = nn.Parameter(torch.full((config.n_heads,), 1.0, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim)

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, seq_length, _ = input_tensor.shape

        q = self.wq(input_tensor).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(input_tensor).view(batch_size, seq_length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(input_tensor).view(batch_size, seq_length, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seq_length, input_tensor.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.n_kv_heads != self.n_heads),
        )
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, -1)
        return self.wo(output)


class MLP(nn.Module):
    """
    ReLU^2 Feed-Forward block (official Parameter-Golf baseline).
    Uses 2 weight matrices vs SwiGLU's 3, saving ~33% MLP parameters.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc = CastedLinear(config.dim, config.hidden_dim, bias=False)
        self.proj = CastedLinear(config.hidden_dim, config.dim, bias=False)
        self.proj._zero_init = True

    def forward(self, input_tensor: Tensor) -> Tensor:
        hidden = torch.relu(self.fc(input_tensor))
        return self.proj(hidden.square())


class TransformerBlock(nn.Module):
    """
    Single Transformer layer with learnable attn/mlp scales and residual mixing.
    resid_mix blends the current residual with the original embedding at each pass,
    which is especially useful under ALBERT-style repetition for gradient flow.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = MLP(config)
        self.attention_norm = RMSNorm()
        self.ffn_norm = RMSNorm()
        self.attn_scale = nn.Parameter(torch.ones(config.dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(config.dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(config.dim), torch.zeros(config.dim))).float()
        )

    def forward(self, input_tensor: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=input_tensor.dtype)
        x = mix[0][None, None, :] * input_tensor + mix[1][None, None, :] * x0

        attn_out = self.attention(self.attention_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.feed_forward(self.ffn_norm(x))
        return x


class LanguageModel(nn.Module):
    """
    La Pulga — Fat ALBERT variant for H100.

    ALBERT-style Cross-Layer Sharing: 4 physical blocks repeated 3× = 12 effective layers.
    The ALBERT loop is UNROLLED into self.full_sequence for torch.compile compatibility.
    Modules are repeated references (shared weights), so param count stays the same.

    Gradient checkpointing is supported via self.use_gradient_checkpointing flag.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size: int = config.vocab_size
        self.logit_softcap: float = config.logit_softcap
        self.use_gradient_checkpointing: bool = False

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # U-Net skip connections indexed over effective (virtual) layer depth
        num_encoder: int = config.effective_layers // 2
        num_decoder: int = config.effective_layers - num_encoder
        self.num_skip_weights: int = min(num_encoder, num_decoder)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, config.dim, dtype=torch.float32)
        )

        # Physical layers — only these hold unique parameters
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.physical_layers)])

        # UNROLLED sequence: repeated references to the physical layers.
        # torch.compile sees 12 linear calls, NOT a Python loop.
        # Weights are shared — self.full_sequence[0] IS self.layers[0].
        self.full_sequence = nn.ModuleList()
        for _ in range(config.repeat_count):
            for layer in self.layers:
                self.full_sequence.append(layer)

        self.norm = RMSNorm()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for VRAM savings."""
        self.use_gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.use_gradient_checkpointing = False

    def _run_block(self, block: TransformerBlock, x: Tensor, x0: Tensor) -> Tensor:
        """Wrapper for gradient checkpointing compatibility."""
        return block(x, x0)

    def forward(self, input_ids: Tensor, target_ids: Optional[Tensor] = None) -> Tensor:
        x = self.tok_embeddings(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        num_encoder: int = self.config.effective_layers // 2
        decoder_step: int = 0

        # Unrolled linear sequence — no Python loop at trace time
        for virtual_idx, block in enumerate(self.full_sequence):
            if virtual_idx < num_encoder:
                if self.use_gradient_checkpointing and self.training:
                    x = checkpoint(self._run_block, block, x, x0, use_reentrant=False)
                else:
                    x = block(x, x0)
                skips.append(x)
            else:
                if skips:
                    sw = self.skip_weights[decoder_step].to(dtype=x.dtype)[None, None, :]
                    x = x + sw * skips.pop()
                if self.use_gradient_checkpointing and self.training:
                    x = checkpoint(self._run_block, block, x, x0, use_reentrant=False)
                else:
                    x = block(x, x0)
                decoder_step += 1

        x = self.norm(x)

        # Weight Tying: reuse tok_embeddings.weight as the output head
        logits = F.linear(x, self.tok_embeddings.weight.to(x.dtype))
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        if target_ids is not None:
            return F.cross_entropy(
                logits.float().reshape(-1, self.vocab_size),
                target_ids.reshape(-1),
                reduction="mean",
            )
        return logits

    @property
    def device(self) -> torch.device:
        return self.tok_embeddings.weight.device
