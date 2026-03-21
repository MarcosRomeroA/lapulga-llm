"""
Data loading and tokenization interfaces.
Abstracts the processing of raw text into framework-specific tensors.
"""
import tiktoken
import mlx.core as mx
from typing import Generator, Tuple

def tokenize_text(text: str, vocab_size: int) -> mx.array:
    """
    Tokenizes raw text using BPE, clamping to the restricted vocab limit.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    # Clamp tokens to prevent Out-Of-Bounds indexing in the restricted embedding matrix
    encoded_list: list[int] = [t if t < vocab_size else 0 for t in enc.encode(text)]
    return mx.array(encoded_list)

def get_batches(
    tokens: mx.array, 
    batch_size: int, 
    context_length: int
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    Yields (inputs, targets) tensors for autoregressive training.
    """
    n_batches: int = len(tokens) // (batch_size * context_length)
    if n_batches == 0:
        n_batches = 1
    
    tokens = tokens[:n_batches * batch_size * context_length + 1]
    input_tensors: mx.array = tokens[:-1].reshape(batch_size, -1)
    target_tokens: mx.array = tokens[1:].reshape(batch_size, -1)
    
    for i in range(0, input_tensors.shape[1], context_length):
        yield input_tensors[:, i:i+context_length], target_tokens[:, i:i+context_length]
