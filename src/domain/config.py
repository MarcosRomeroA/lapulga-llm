"""
Domain Entities for lapulga-llm.
Contains pure Python configurations. Agnostic to ML frameworks.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration schema for the La Pulga Transformer architecture."""
    dim: int = 768
    physical_layers: int = 4
    repeat_count: int = 3
    n_heads: int = 12
    n_kv_heads: int = 4
    hidden_dim: int = 3072
    vocab_size: int = 1024
    norm_eps: float = 1e-5
    context_length: int = 1024
    logit_softcap: float = 30.0

    @property
    def effective_layers(self) -> int:
        """Total effective depth = physical layers × repeats (ALBERT-style sharing)."""
        return self.physical_layers * self.repeat_count


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration schema for the training orchestration."""
    batch_size: int = 524288
    micro_batch_size: int = 16384
    context_length: int = 1024
    learning_rate: float = 6e-4
    epochs: int = 1
    train_tokens: int = 500_000_000
    val_tokens: int = 50_000
    checkpoint_path: str = "lapulga_weights.safetensors"
    tokenizer_path: str = "data/tokenizers/fineweb_1024_bpe.model"
    data_path: str = "data/datasets/fineweb10B_sp1024"
    scoring_metric: str = "bpb"
