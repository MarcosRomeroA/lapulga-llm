"""
PyTorch Implementation of La Pulga Transformer.
Strictly isolated tensor operations for CUDA (RTX 3090 / H100).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from src.domain.config import ModelConfig

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps: float = eps

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(input_tensor.pow(2).mean(-1, keepdim=True) + self.eps)
        return input_tensor * norm * self.weight


def precompute_rope_frequencies(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes Rotary Position Embedding (RoPE) frequency table.
    Returns complex-valued tensor of shape [max_seq_len, head_dim // 2].
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to a query or key tensor.
    tensor shape: [B, H, L, D] -> view as complex [B, H, L, D//2]
    """
    complex_tensor = torch.view_as_complex(tensor.float().reshape(*tensor.shape[:-1], -1, 2))
    freqs = freqs[:tensor.shape[2], :].unsqueeze(0).unsqueeze(0)
    rotated = complex_tensor * freqs
    return torch.view_as_real(rotated).reshape(tensor.shape).type_as(tensor)


class Attention(nn.Module):
    """Grouped-Query Attention Mechanism (GQA) with RoPE."""
    def __init__(self, config: ModelConfig, rope_freqs: torch.Tensor):
        super().__init__()
        self.n_heads: int = config.n_heads
        self.n_kv_heads: int = config.n_kv_heads
        self.head_dim: int = config.dim // config.n_heads
        self.scale: float = self.head_dim ** -0.5

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

    def forward(self, input_tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = input_tensor.shape

        q = self.wq(input_tensor).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(input_tensor).view(batch_size, seq_length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(input_tensor).view(batch_size, seq_length, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        if self.n_heads != self.n_kv_heads:
            repeats: int = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores, dim=-1)

        output = (scores @ v).transpose(1, 2).reshape(batch_size, seq_length, -1)
        return self.wo(output)


class MLP(nn.Module):
    """Feed-Forward SwiGLU block."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(input_tensor)) * self.w3(input_tensor))


class TransformerBlock(nn.Module):
    """Single layer of the Transformer."""
    def __init__(self, config: ModelConfig, rope_freqs: torch.Tensor):
        super().__init__()
        self.attention = Attention(config, rope_freqs)
        self.feed_forward = MLP(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, input_tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_tensor = input_tensor + self.attention(self.attention_norm(input_tensor), mask)
        out_tensor = h_tensor + self.feed_forward(self.ffn_norm(h_tensor))
        return out_tensor


class LanguageModel(nn.Module):
    """
    The complete La Pulga Model architecture in PyTorch.

    Weight Tying: The output projection head shares weights with the input embedding
    layer (tok_embeddings), saving ~2M parameters (~30% of total). This is implemented
    via a direct matmul with the embedding weight matrix in the forward pass.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size: int = config.vocab_size

        rope_freqs = precompute_rope_frequencies(
            head_dim=config.dim // config.n_heads,
            max_seq_len=config.context_length,
        )

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config, rope_freqs) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)

        self._init_weights(config)

    def _init_weights(self, config: ModelConfig) -> None:
        """
        Applies transformer-standard weight initialization.

        Why: PyTorch's nn.Embedding defaults to N(0,1), which produces logit std ≈ sqrt(dim) = 16
        at init. This causes initial loss ~82 instead of the expected ln(vocab) ≈ 9.01, and
        severely slows convergence. MLX used N(0, 1/sqrt(dim)) by default.
        We use std=0.02 (nanoGPT standard) for all weights, and scale residual projections
        (wo, w2) by 1/sqrt(2 * n_layers) to prevent residual stream growth.
        """
        residual_std: float = 0.02 / math.sqrt(2 * config.n_layers)
        nn.init.normal_(self.tok_embeddings.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, (Attention, MLP)):
                # Scale residual projections: output of attention (wo) and MLP (w2)
                residual_proj = getattr(module, "wo", None) or getattr(module, "w2", None)
                if residual_proj is not None:
                    nn.init.normal_(residual_proj.weight, std=residual_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_length = input_ids.shape
        mask = torch.full((seq_length, seq_length), float("-inf"), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        hidden_states = self.tok_embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)

        # Weight Tying: reuse tok_embeddings.weight as the output head
        return self.norm(hidden_states) @ self.tok_embeddings.weight.T

    @property
    def device(self) -> torch.device:
        """Returns the device of the model parameters."""
        return self.tok_embeddings.weight.device


def cross_entropy_loss(model: LanguageModel, input_tensors: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    """Calculates cross entropy loss for autoregressive training."""
    logits = model(input_tensors)
    logits_fp32 = logits.float()
    loss = F.cross_entropy(logits_fp32.reshape(-1, logits.shape[-1]), target_tokens.reshape(-1))
    return loss
