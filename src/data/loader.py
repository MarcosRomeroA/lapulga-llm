"""
Data loading and tokenization interfaces.
Abstracts the processing of raw text into PyTorch tensors.
Uses La Pulga's custom BPE tokenizer (trained on TinyStories).
"""
import torch
from torch.utils.data import IterableDataset, get_worker_info
from tokenizers import Tokenizer
from typing import Generator, Tuple, Iterable
from src.data.shard_loader import TokenStream, load_data_shard


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


class FineWebDataset(IterableDataset):
    """
    Wraps TokenStream into a PyTorch IterableDataset yielding (x, y) sequences.
    Safely distributes shard files across multiple DataLoader workers to prevent duplicated data.
    """
    def __init__(self, shard_pattern: str, seq_len: int):
        super().__init__()
        self.shard_pattern = shard_pattern
        self.seq_len = seq_len
        
    def __iter__(self):
        worker_info = get_worker_info()
        stream = TokenStream(self.shard_pattern)
        
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # Distribute shards (files) based on worker ID securely
            worker_files = [f for i, f in enumerate(stream.files) if i % num_workers == worker_id]
            if not worker_files:
                worker_files = stream.files  # Fallback
                
            stream.files = worker_files
            stream.file_idx = 0
            stream.tokens = load_data_shard(stream.files[0])
            stream.pos = 0

        while True:
            # Yield (x, y) token pairs natively
            chunk = stream.take(self.seq_len + 1)
            x = chunk[:-1].to(torch.int64)
            y = chunk[1:].to(torch.int64)
            yield x, y
