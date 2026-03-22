"""
PyTorch Implementation of La Pulga Transformer.
Aligned with the official OpenAI Parameter-Golf architecture.
Uses relu^2 MLP, U-Net skip connections, logit softcap, and SDPA.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from src.domain.config import ModelConfig

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
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
        self.fc = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.proj = nn.Linear(config.hidden_dim, config.dim, bias=False)
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
    La Pulga — Fat ALBERT variant for H100.

    ALBERT-style Cross-Layer Sharing: 4 physical blocks repeated 3× = 12 effective layers.
    At dim=768 the H100's 228KB L1 shared memory easily fits fused RMSNorm backward kernels
    (vs the RTX 3090's 101KB limit which forced dim=512).

    Budget breakdown (stored params ~26M, artifact ~12MB int8+zlib):
    - 4 physical layers × ~6.3M params/layer = ~25.2M
    - Embeddings 1024×768 (tied as output head)  = 0.79M
    - Skip weights 6×768 + final norm             = ~5K
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size: int = config.vocab_size
        self.logit_softcap: float = config.logit_softcap

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # U-Net skip connections indexed over effective (virtual) layer depth
        num_encoder: int = config.effective_layers // 2
        num_decoder: int = config.effective_layers - num_encoder
        self.num_skip_weights: int = min(num_encoder, num_decoder)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, config.dim, dtype=torch.float32)
        )

        # Only physical_layers unique blocks stored — shared across repeats
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.physical_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Optional[Tensor] = None) -> Tensor:
        x = self.tok_embeddings(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        num_encoder: int = self.config.effective_layers // 2
        virtual_idx: int = 0
        decoder_step: int = 0

        # ALBERT loop: traverse the 4 physical blocks repeat_count times
        for _ in range(self.config.repeat_count):
            for layer in self.layers:
                if virtual_idx < num_encoder:
                    x = layer(x, x0)
                    skips.append(x)
                else:
                    if skips:
                        sw = self.skip_weights[decoder_step].to(dtype=x.dtype)[None, None, :]
                        x = x + sw * skips.pop()
                    x = layer(x, x0)
                    decoder_step += 1
                virtual_idx += 1

        x = self.norm(x)

        # Weight Tying: reuse tok_embeddings.weight as the output head
        logits = F.linear(x, self.tok_embeddings.weight)
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
