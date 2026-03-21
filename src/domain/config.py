"""
Domain Entities for lapulga-llm.
Contains pure Python configurations. Agnostic to ML frameworks.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    """Configuration schema for the La Pulga Transformer architecture."""
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    hidden_dim: int = 1024
    vocab_size: int = 8192
    norm_eps: float = 1e-5

@dataclass(frozen=True)
class TrainingConfig:
    """Configuration schema for the training orchestration."""
    batch_size: int = 8
    context_length: int = 256
    learning_rate: float = 3e-4
    epochs: int = 3
    train_tokens: int = 2_000_000
    val_tokens: int = 50_000
    checkpoint_path: str = "lapulga_weights.safetensors"
    tokenizer_path: str = "tokenizer.json"
