"""
Data loading and tokenization interfaces.
Abstracts the processing of raw text into framework-specific tensors.
Uses La Pulga's custom BPE tokenizer (trained on TinyStories).
"""
import mlx.core as mx
from tokenizers import Tokenizer
from typing import Generator, Tuple, Iterable


def load_tokenizer(tokenizer_path: str = "tokenizer.json") -> Tokenizer:
    """Loads the pre-trained custom BPE tokenizer from disk."""
    return Tokenizer.from_file(tokenizer_path)


def tokenize_stream(
    text_stream: Iterable[str],
    tokenizer: Tokenizer,
    target_tokens: int = 250_000,
) -> mx.array:
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

    return mx.array(tokens[:target_tokens])


def get_batches(
    tokens: mx.array,
    batch_size: int,
    context_length: int,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    Yields (inputs, targets) tensors for autoregressive training.
    """
    n_batches: int = len(tokens) // (batch_size * context_length)
    if n_batches == 0:
        n_batches = 1

    tokens = tokens[: n_batches * batch_size * context_length + 1]
    input_tensors: mx.array = tokens[:-1].reshape(batch_size, -1)
    target_tokens_arr: mx.array = tokens[1:].reshape(batch_size, -1)

    for i in range(0, input_tensors.shape[1], context_length):
        yield input_tensors[:, i : i + context_length], target_tokens_arr[:, i : i + context_length]
