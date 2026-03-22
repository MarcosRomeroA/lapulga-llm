"""
PyTorch Implementation of La Pulga Transformer.
Aligned with the official OpenAI Parameter-Golf architecture.
Uses relu^2 MLP, U-Net skip connections, logit softcap, and SDPA.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from src.domain.config import ModelConfig

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (parameterless variant)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, input_tensor: Tensor) -> Tensor:
        return F.rms_norm(input_tensor, (self.dim,), self.weight, self.eps)


class Rotary(nn.Module):
    """
    Rotary Position Embeddings with cached cos/sin tables.
    Caches are rebuilt when sequence length or device changes.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached: int = 0
        self._cos_cached: Optional[Tensor] = None
        self._sin_cached: Optional[Tensor] = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(positions, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


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

        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wk = nn.Linear(config.dim, kv_dim, bias=False)
        self.wv = nn.Linear(config.dim, kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.wo._zero_init = True

        self.q_gain = nn.Parameter(torch.full((config.n_heads,), 1.0, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim)

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, seq_length, _ = input_tensor.shape

        q = self.wq(input_tensor).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(input_tensor).view(batch_size, seq_length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(input_tensor).view(batch_size, seq_length, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-norm for training stability
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seq_length, input_tensor.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Learnable per-head gain on queries
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        # Flash Attention via SDPA with native GQA support
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

    Why relu^2 instead of SwiGLU?
    SwiGLU uses 3 weight matrices per layer; relu^2 uses only 2.
    This saves ~33% of MLP parameters, allowing more depth (9 layers)
    within the 16MB budget. All top Parameter-Golf entries use relu^2.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.proj = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.proj._zero_init = True

    def forward(self, input_tensor: Tensor) -> Tensor:
        hidden = torch.relu(self.fc(input_tensor))
        return self.proj(hidden.square())


class TransformerBlock(nn.Module):
    """
    Single Transformer layer with learnable attn/mlp scales and residual mixing.
    The resid_mix parameter blends the current residual (x) with the original
    embedding (x0) at each layer, improving gradient flow.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = MLP(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
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
    The complete La Pulga Model architecture in PyTorch.
    Aligned with the official Parameter-Golf baseline.

    Key features:
    - Weight Tying: tok_embeddings shared as output head
    - U-Net Skip Connections: encoder layers feed into decoder layers
    - Logit Softcap: tanh(logits/cap)*cap prevents logit explosion
    - relu^2 MLP: 2 matrices instead of SwiGLU's 3
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size: int = config.vocab_size
        self.logit_softcap: float = config.logit_softcap

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # U-Net: split layers into encoder and decoder halves
        self.num_encoder_layers: int = config.n_layers // 2
        self.num_decoder_layers: int = config.n_layers - self.num_encoder_layers
        self.num_skip_weights: int = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, config.dim, dtype=torch.float32)
        )

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Applies Parameter-Golf weight initialization.
        - Embedding: N(0, 0.02) for stable initial logits
        - Residual projections (wo, proj): zero-init for clean residual stream at start
        """
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Optional[Tensor] = None) -> Tensor:
        x = self.tok_embeddings(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # Encoder half: store skip connections
        for i in range(self.num_encoder_layers):
            x = self.layers[i](x, x0)
            skips.append(x)

        # Decoder half: reuse skips in reverse order
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.layers[self.num_encoder_layers + i](x, x0)

        x = self.norm(x)

        # Weight Tying: reuse tok_embeddings.weight as the output head
        logits = F.linear(x, self.tok_embeddings.weight)

        # Logit softcap: prevents explosion without gradient clipping
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
        """Returns the device of the model parameters."""
        return self.tok_embeddings.weight.device
