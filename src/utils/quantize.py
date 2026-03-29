"""
Post-Training Int8 Quantization and Compression for La Pulga.
Matches the official Parameter-Golf quantization format exactly.

Format: per-row int8 for 2D tensors, per-tensor int8 for vectors,
fp16 passthrough for small tensors (<65k elements), zlib level 9 compression.
"""
import io
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

INT8_KEEP_FLOAT_MAX_NUMEL: int = 16_384
INT8_KEEP_FLOAT_STORE_DTYPE: torch.dtype = torch.float16
INT8_PER_ROW_SCALE_DTYPE: torch.dtype = torch.float16
INT8_CLIP_PERCENTILE: float = 99.99984
INT8_CLIP_Q: float = INT8_CLIP_PERCENTILE / 100.0

CONTROL_TENSOR_PATTERNS: tuple[str, ...] = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight",
)


def _tensor_nbytes(t: Tensor) -> int:
    """Returns the byte size of a tensor's data."""
    return int(t.numel()) * int(t.element_size())


def _keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    """Decides how to store small/control tensors that skip int8 quantization."""
    if any(pattern in name for pattern in CONTROL_TENSOR_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    """
    Quantizes a single float tensor to int8.
    - 2D tensors: per-row quantization (one scale per output channel)
    - 1D/scalar tensors: per-tensor quantization
    """
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars: single scale for the whole tensor
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]) -> tuple[dict, dict]:
    """
    Quantizes a full model state_dict to int8.

    Returns:
        (quantized_obj, stats) — the serializable quantized object and size statistics
    """
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats: dict[str, int] = {
        "param_count": 0,
        "num_tensors": 0,
        "baseline_tensor_bytes": 0,
        "int8_payload_bytes": 0,
    }

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += _tensor_nbytes(t)

        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_payload_bytes"] += _tensor_nbytes(t)
            continue

        # Small tensors are cheap enough to keep without int8
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = _keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += _tensor_nbytes(kept)
            continue

        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += _tensor_nbytes(q) + _tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    """Dequantizes an int8 quantized object back to float tensors."""
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()

    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t

    return out


def compute_artifact_size(state_dict: dict[str, Tensor], code_path: Path) -> tuple[int, int, int]:
    """
    Computes the total submission artifact size: code bytes + zlib(int8 model).

    Returns:
        (total_bytes, code_bytes, model_bytes)
    """
    code_bytes: int = len(code_path.read_bytes()) if code_path.exists() else 0

    obj, _stats = quantize_state_dict_int8(state_dict)
    buf = io.BytesIO()
    torch.save(obj, buf)
    raw_bytes = buf.getvalue()
    compressed = zlib.compress(raw_bytes, level=9)
    model_bytes: int = len(compressed)

    total_bytes: int = code_bytes + model_bytes
    return total_bytes, code_bytes, model_bytes
