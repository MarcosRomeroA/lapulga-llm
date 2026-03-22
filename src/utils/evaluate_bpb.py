"""
BPB (Bits Per Byte) Evaluation Module for La Pulga.
Implements the official Parameter-Golf scoring metric.

BPB is tokenizer-agnostic: it measures compression quality in bits per UTF-8 byte,
so models with different tokenizers can be compared fairly on the FineWeb validation set.
"""
import math
import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def compute_bpb(
    model: torch.nn.Module,
    val_tokens: Tensor,
    seq_len: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """
    Computes both validation loss and BPB on a token tensor.

    The BPB formula:
        bits_per_token = val_loss / ln(2)
        tokens_per_byte = total_tokens / total_utf8_bytes
        bpb = bits_per_token * tokens_per_byte

    Returns:
        (val_loss, val_bpb) — loss in nats and BPB score
    """
    model.eval()

    total_seqs: int = (val_tokens.numel() - 1) // seq_len
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    for seq_idx in range(total_seqs):
        start: int = seq_idx * seq_len
        local = val_tokens[start : start + seq_len + 1].to(device=device, dtype=torch.int64)
        x = local[:-1].unsqueeze(0)
        y = local[1:].unsqueeze(0)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss = model(x, y).detach()

        batch_token_count: float = float(y.numel())
        val_loss_sum += loss.to(torch.float64) * batch_token_count
        val_token_count += batch_token_count

        # Count UTF-8 bytes using the SentencePiece lookup tables
        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
        # Leading space contributes +1 byte when the previous token is not a boundary
        token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
        val_byte_count += token_bytes.to(torch.float64).sum()

    val_loss: float = (val_loss_sum / val_token_count).item()
    bits_per_token: float = val_loss / math.log(2.0)
    tokens_per_byte: float = val_token_count.item() / val_byte_count.item()
    val_bpb: float = bits_per_token * tokens_per_byte

    return val_loss, val_bpb
