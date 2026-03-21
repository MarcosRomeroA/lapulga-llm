"""
MLX Framework Implementation of La Pulga.
Strictly isolated tensor operations for Apple Silicon.
"""
import mlx.core as mx
import mlx.nn as nn
from src.domain.config import ModelConfig

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight: mx.array = mx.ones((dims,))
        self.eps: float = eps

    def __call__(self, input_tensor: mx.array) -> mx.array:
        return mx.fast.rms_norm(input_tensor, self.weight, self.eps)

class Attention(nn.Module):
    """Grouped-Query Attention Mechanism (GQA)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads: int = config.n_heads
        self.n_kv_heads: int = config.n_kv_heads
        self.head_dim: int = config.dim // config.n_heads
        self.scale: float = self.head_dim ** -0.5

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False)

    def __call__(self, input_tensor: mx.array, mask: mx.array | None = None) -> mx.array:
        batch_size, seq_length, _ = input_tensor.shape
        q: mx.array = self.wq(input_tensor).reshape(batch_size, seq_length, self.n_heads, self.head_dim)
        k: mx.array = self.wk(input_tensor).reshape(batch_size, seq_length, self.n_kv_heads, self.head_dim)
        v: mx.array = self.wv(input_tensor).reshape(batch_size, seq_length, self.n_kv_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        if self.n_heads != self.n_kv_heads:
            repeats: int = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, repeats, axis=2)
            v = mx.repeat(v, repeats, axis=2)

        scores: mx.array = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        output: mx.array = (scores @ v).reshape(batch_size, seq_length, -1)
        return self.wo(output)

class MLP(nn.Module):
    """Feed-Forward SwiGLU block."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

    def __call__(self, input_tensor: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(input_tensor)) * self.w3(input_tensor))

class TransformerBlock(nn.Module):
    """Single layer of the Transformer."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = MLP(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def __call__(self, input_tensor: mx.array, mask: mx.array | None = None) -> mx.array:
        h_tensor: mx.array = input_tensor + self.attention(self.attention_norm(input_tensor), mask)
        out_tensor: mx.array = h_tensor + self.feed_forward(self.ffn_norm(h_tensor))
        return out_tensor

class LanguageModel(nn.Module):
    """The complete La Pulga Model architecture in MLX."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.vocab_size: int = config.vocab_size
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.norm = RMSNorm(config.dim, config.norm_eps)
        # Weight Tying explicitly
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

    def __call__(self, input_ids: mx.array) -> mx.array:
        _, seq_length = input_ids.shape
        mask: mx.array = nn.MultiHeadAttention.create_additive_causal_mask(seq_length, input_ids.dtype)
        
        hidden_states: mx.array = self.tok_embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        return self.output(self.norm(hidden_states))

def cross_entropy_loss(model: LanguageModel, input_tensors: mx.array, target_tokens: mx.array) -> mx.array:
    """Calculates cross entropy loss for autoregressive training."""
    logits: mx.array = model(input_tensors)
    loss: mx.array = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_tokens.reshape(-1))
    return mx.mean(loss)
