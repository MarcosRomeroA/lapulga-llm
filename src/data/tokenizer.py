"""
SentencePiece Tokenizer Adapter for Parameter-Golf BPB Evaluation.
Builds the lookup tables required for tokenizer-agnostic Bits-Per-Byte scoring.
"""
import numpy as np
import torch
from torch import Tensor
import sentencepiece as spm


def load_sentencepiece(path: str) -> spm.SentencePieceProcessor:
    """Loads a SentencePiece .model file from disk."""
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def build_bpb_lookup_tables(
    sp: spm.SentencePieceProcessor,
    vocab_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Builds three lookup tables for BPB (Bits Per Byte) calculation.

    Why BPB instead of perplexity?
    BPB is tokenizer-agnostic: it measures compression in bits per UTF-8 byte,
    so models with different tokenizers can be compared fairly. This is the
    official scoring metric for the Parameter-Golf challenge.

    Returns:
        base_bytes_lut: UTF-8 byte count per token (int16)
        has_leading_space_lut: whether the token starts with the SentencePiece
            space marker (U+2581 "▁") (bool)
        is_boundary_token_lut: whether the token is a control/unknown/unused
            marker that contributes zero bytes (bool)
    """
    sp_vocab_size: int = int(sp.vocab_size())
    table_size: int = max(sp_vocab_size, vocab_size)

    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)

    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece: str = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))

    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )
