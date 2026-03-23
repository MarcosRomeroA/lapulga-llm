"""
Training Orchestrator for Lapulga.
Pre-loads all training data into RAM, then trains with pure tensor slicing.
Exports via int8 quantization + zlib compression (official Parameter-Golf format).

Optimizations applied (matching official parameter-golf baseline):
  - Native bfloat16 model + autocast (no GradScaler needed on H100)
  - torch.compile(dynamic=False, fullgraph=True) for max kernel fusion
  - TF32 matmuls + Flash Attention only (no fallback SDP backends)
  - 20 warmup steps with model/optimizer state reset to eliminate Step 0 penalty
  - Fused Adam optimizer
"""
import copy
import io
import math
import os
import time
import zlib

import torch
import torch.nn.functional as F
from torch.backends.cuda import (
    enable_cudnn_sdp,
    enable_flash_sdp,
    enable_math_sdp,
    enable_mem_efficient_sdp,
)
from safetensors.torch import save_file as safetensors_save
from src.domain.config import TrainingConfig
from src.model.transformer import LanguageModel, DEVICE
from src.data.loader import preload_train_tokens
from src.utils.quantize import quantize_state_dict_int8

_COMPILE_WARMUP_STEPS: int = 20


def _next_chunk(
    all_tokens: torch.Tensor,
    token_cursor: int,
    need: int,
    total_available: int,
) -> tuple[torch.Tensor, int]:
    if token_cursor + need > total_available:
        token_cursor = 0
    chunk = all_tokens[token_cursor : token_cursor + need]
    return chunk, token_cursor + (need - 1)


def execute_training(model: LanguageModel, train_config: TrainingConfig) -> None:
    """
    Executes the main optimization loop on FineWeb shards.
    All training data is pre-loaded into RAM to eliminate I/O bottlenecks.
    Trains in native bfloat16 with torch.compile, then exports via int8+zlib.
    """
    dev_mode: bool = os.environ.get("DEV_MODE", "0") == "1"

    # ------------------------------------------------------------
    # CUDA Knobs — match official baseline
    # ------------------------------------------------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    # ------------------------------------------------------------
    # Model — native bfloat16, keep small params in fp32
    # ------------------------------------------------------------
    model.to(DEVICE)
    model.bfloat16()
    model.train()

    print("Compiling model (dynamic=False, fullgraph=True)...")
    compiled_model = torch.compile(model, dynamic=False, fullgraph=True)

    # ------------------------------------------------------------
    # Optimizer — fused AdamW, no GradScaler needed in bf16
    # ------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        fused=True,
    )

    seq_len: int = train_config.context_length
    batch_tokens: int = train_config.batch_size
    micro_tokens: int = train_config.micro_batch_size
    accum_steps: int = max(1, batch_tokens // micro_tokens)
    grad_scale: float = 1.0 / accum_steps

    # ------------------------------------------------------------
    # Pre-load ALL training data into pinned RAM
    # ------------------------------------------------------------
    shard_pattern: str = os.path.join(train_config.data_path, "fineweb_train_*.bin")
    all_tokens: torch.Tensor = preload_train_tokens(shard_pattern)
    token_cursor: int = 0
    total_available: int = all_tokens.numel()

    # ------------------------------------------------------------
    # Training schedule
    # ------------------------------------------------------------
    steps_per_epoch: int = max(1, train_config.train_tokens // batch_tokens)
    total_steps: int = steps_per_epoch * train_config.epochs

    if dev_mode:
        total_steps = min(total_steps, 100)
        steps_per_epoch = min(steps_per_epoch, 100)
        print("*** DEV_MODE: capped at 100 steps ***")

    log_interval: int = 10 if dev_mode else 50
    lr_warmup_steps: int = max(1, int(total_steps * 0.05))

    def lr_lambda(step: int) -> float:
        if step < lr_warmup_steps:
            return step / lr_warmup_steps
        progress = (step - lr_warmup_steps) / max(1, total_steps - lr_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------
    # Compiler warmup: run N throwaway steps, then reset state
    # This eliminates the ~50s penalty on Step 0 during real training
    # ------------------------------------------------------------
    warmup_steps_count = 0 if dev_mode else _COMPILE_WARMUP_STEPS
    if warmup_steps_count > 0:
        print(f"Running {warmup_steps_count} compiler warmup steps (state will be reset)...")
        initial_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        initial_optimizer_state = copy.deepcopy(optimizer.state_dict())

        for _ in range(warmup_steps_count):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum_steps):
                need = micro_tokens + 1
                chunk, token_cursor = _next_chunk(all_tokens, token_cursor, need, total_available)
                x = chunk[:-1].reshape(-1, seq_len).to(device=DEVICE, dtype=torch.int64, non_blocking=True)
                y = chunk[1:].reshape(-1, seq_len).to(device=DEVICE, dtype=torch.int64, non_blocking=True)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                    loss = compiled_model(x, y)
                (loss * grad_scale).backward()
            optimizer.step()

        # Reset to true initial state — timer starts here
        model.load_state_dict(initial_model_state, strict=True)
        optimizer.load_state_dict(initial_optimizer_state)
        optimizer.zero_grad(set_to_none=True)
        token_cursor = 0
        torch.cuda.synchronize()
        print("Warmup complete. Weights reset to init. Starting real training...")

    print(f"--- Starting La Pulga Training Loop on {DEVICE} ---")
    print(f"    Batch tokens: {batch_tokens:,} | Micro tokens: {micro_tokens:,} | Accum steps: {accum_steps}")
    print(f"    Seq len: {seq_len} | Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")
    print(f"    LR: {train_config.learning_rate} | LR warmup: {lr_warmup_steps} steps | Log every {log_interval} steps")

    # ------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------
    torch.cuda.synchronize()
    for epoch in range(train_config.epochs):
        epoch_loss: float = 0.0

        for step in range(steps_per_epoch):
            t0: float = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=DEVICE)

            for _ in range(accum_steps):
                need = micro_tokens + 1
                chunk, token_cursor = _next_chunk(all_tokens, token_cursor, need, total_available)
                x = chunk[:-1].reshape(-1, seq_len).to(device=DEVICE, dtype=torch.int64, non_blocking=True)
                y = chunk[1:].reshape(-1, seq_len).to(device=DEVICE, dtype=torch.int64, non_blocking=True)

                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                    loss = compiled_model(x, y)

                step_loss += loss.detach()
                (loss * grad_scale).backward()

            optimizer.step()
            scheduler.step()

            # Single GPU→CPU sync per step
            step_loss_float: float = (step_loss / accum_steps).item()
            t1: float = time.perf_counter()
            epoch_loss += step_loss_float

            if step % log_interval == 0:
                bpb_est: float = step_loss_float / math.log(2)
                lr_now: float = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch} | Step {step:04d}/{steps_per_epoch} | "
                    f"Loss {step_loss_float:.4f} | BPB ~{bpb_est:.4f} | "
                    f"LR {lr_now:.2e} | Time {t1-t0:.4f}s"
                )

        avg_epoch_loss: float = epoch_loss / steps_per_epoch
        print(f"--- Epoch {epoch} complete | Avg Loss: {avg_epoch_loss:.4f} ---")

    # ------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------
    orig_model = getattr(model, "_orig_mod", model)
    raw_state_dict = orig_model.state_dict()

    print("--- Saving Checkpoint (.safetensors) ---")
    cpu_state_dict = {k: v.cpu().contiguous() for k, v in raw_state_dict.items()}
    safetensors_save(cpu_state_dict, train_config.checkpoint_path)
    print(f"Checkpoint saved to {train_config.checkpoint_path}")

    print("--- Building Submission Artifact (int8 + zlib) ---")
    obj, stats = quantize_state_dict_int8(raw_state_dict)
    buf = io.BytesIO()
    torch.save(obj, buf)
    compressed = zlib.compress(buf.getvalue(), level=9)

    artifact_path: str = train_config.checkpoint_path.replace(".safetensors", "_submission.pt.zlib")
    with open(artifact_path, "wb") as f:
        f.write(compressed)

    model_bytes: int = len(compressed)
    print(f"Submission artifact saved to {artifact_path}")
    print(f"Int8+zlib size: {model_bytes:,} bytes ({model_bytes / 1_000_000:.2f} MB)")
    print(f"Params quantized: {stats['param_count']:,}")
