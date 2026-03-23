"""
Data loading and tokenization interfaces.
Abstracts the processing of raw text into PyTorch tensors.
Uses La Pulga's custom BPE tokenizer (trained on TinyStories).
"""
import glob
from pathlib import Path

import torch
from tokenizers import Tokenizer
from typing import Generator, Tuple, Iterable

from src.data.shard_loader import load_data_shard


def load_tokenizer(tokenizer_path: str = "tokenizer.json") -> Tokenizer:
    """Loads the pre-trained custom BPE tokenizer from disk."""
    return Tokenizer.from_file(tokenizer_path)


def tokenize_stream(
    text_stream: Iterable[str],
    tokenizer: Tokenizer,
    target_tokens: int = 250_000,
) -> torch.Tensor:
    """
    Consumes a stream of raw text and tokenizes via our custom BPE.
    No more clamping needed since every token ID is already in range [0, 8191].
    Stops dynamically when target_tokens count is reached.
    """
    tokens: list[int] = []

    for text in text_stream:
        encoded: list[int] = tokenizer.encode(text).ids
        tokens.extend(encoded)
        if len(tokens) >= target_tokens:
            break

    return torch.tensor(tokens[:target_tokens], dtype=torch.long)


def get_batches(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device = torch.device("cpu"),
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Yields (inputs, targets) tensors for autoregressive training.
    Transfers each batch to the specified device (GPU) on the fly.
    """
    # Clamp batch_size to the number of full sequences available in this token set.
    # This prevents reshape errors when val_tokens < batch_size * context_length.
    n_full_sequences: int = (len(tokens) - 1) // context_length
    effective_batch_size: int = min(batch_size, max(1, n_full_sequences))

    n_batches: int = max(1, n_full_sequences // effective_batch_size)
    tokens = tokens[: n_batches * effective_batch_size * context_length + 1]
    input_tensors = tokens[:-1].reshape(effective_batch_size, -1)
    target_tokens_arr = tokens[1:].reshape(effective_batch_size, -1)

    for i in range(0, input_tensors.shape[1], context_length):
        inputs = input_tensors[:, i : i + context_length].to(device)
        targets = target_tokens_arr[:, i : i + context_length].to(device)
        yield inputs, targets


def preload_train_tokens(shard_pattern: str) -> torch.Tensor:
    """
    Pre-loads ALL training shards into a single contiguous CPU tensor.
    For 500M tokens (~1GB), this fits easily in RAM and eliminates
    all disk I/O during training, which is critical for small models
    on fast GPUs like the H100 NVL.
    """
    files = sorted(glob.glob(shard_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {shard_pattern}")

    print(f"Pre-loading {len(files)} training shards into memory...")
    all_shards = [load_data_shard(Path(f)) for f in files]
    tokens = torch.cat(all_shards).contiguous().pin_memory()
    print(f"Loaded {tokens.numel():,} tokens ({tokens.numel() * 2 / 1e9:.2f} GB)")
    return tokens
