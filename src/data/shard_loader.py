"""
Binary Shard Loader for FineWeb (Official Parameter-Golf Format).
Reads .bin shard files with 256×int32 header + uint16 token payload.
Compatible with the official OpenAI parameter-golf data pipeline.
"""
import glob
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import torch
from torch import Tensor

SHARD_MAGIC: int = 20240520
SHARD_VERSION: int = 1


def load_data_shard(file: Path) -> Tensor:
    """
    Reads a single FineWeb binary shard.

    Format: 256 × little-endian int32 header, followed by uint16 tokens.
    Header[0] = magic (20240520), Header[1] = version (1), Header[2] = num_tokens.
    """
    header_bytes: int = 256 * np.dtype("<i4").itemsize
    token_bytes: int = np.dtype("<u2").itemsize

    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")

    num_tokens: int = int(header[2])
    expected_size: int = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")

    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")

    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    """
    Reads shards sequentially and wraps around forever.
    Deterministic streaming — no sampling, no workers.
    """
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx: int = 0
        self.tokens: Tensor = load_data_shard(self.files[0])
        self.pos: int = 0

    def _advance_file(self) -> None:
        """Move to the next shard, wrapping around to the first when exhausted."""
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        """Consume exactly n tokens from the stream, crossing shard boundaries."""
        chunks: list[Tensor] = []
        remaining: int = n
        while remaining > 0:
            avail: int = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k: int = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    """
    Loads and concatenates all validation shard files into a single token tensor.
    Trims to a multiple of seq_len + 1 for clean (x, y) batch slicing.
    """
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens: Tensor = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable: int = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def get_fineweb_batches(
    tokens: Tensor,
    seq_len: int,
    device: torch.device = torch.device("cpu"),
) -> Generator[Tuple[Tensor, Tensor], None, None]:
    """
    Yields (inputs, targets) from a flat validation token tensor.
    Each batch is a single sequence of length seq_len.
    """
    total_seqs: int = (tokens.numel() - 1) // seq_len
    for i in range(total_seqs):
        start: int = i * seq_len
        local = tokens[start : start + seq_len + 1].to(dtype=torch.int64)
        x = local[:-1].unsqueeze(0).to(device, non_blocking=True)
        y = local[1:].unsqueeze(0).to(device, non_blocking=True)
        yield x, y
