"""
Training Orchestrator for Lapulga.
Reads FineWeb shards via TokenStream and trains the model with AMP.
Exports via int8 quantization + zlib compression (official Parameter-Golf format).
"""
import io
import math
import os
import time
import zlib

import torch
from safetensors.torch import save_file as safetensors_save
from src.domain.config import TrainingConfig
from src.model.transformer import LanguageModel, DEVICE
from src.data.shard_loader import TokenStream
from src.utils.quantize import quantize_state_dict_int8


def execute_training(model: LanguageModel, train_config: TrainingConfig) -> None:
    """
    Executes the main optimization loop on FineWeb shards.
    Trains in FP32 (with AMP on CUDA), then exports via int8+zlib.
    """
    dev_mode: bool = os.environ.get("DEV_MODE", "0") == "1"

    model.to(DEVICE)
    model.train()

    # torch.compile disabled: Triton exceeds RTX 3090 shared memory limit
    # during backward pass compilation of fused RMSNorm kernels.
    compiled_model = model

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    use_amp: bool = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    seq_len: int = train_config.context_length
    batch_tokens: int = train_config.batch_size  # total tokens per optimizer step
    micro_tokens: int = train_config.micro_batch_size  # tokens per forward pass
    accum_steps: int = max(1, batch_tokens // micro_tokens)

    # Open the shard stream
    shard_pattern: str = os.path.join(train_config.data_path, "fineweb_train_*.bin")
    stream = TokenStream(shard_pattern)

    # Calculate steps: train_tokens / batch_tokens * epochs
    steps_per_epoch: int = max(1, train_config.train_tokens // batch_tokens)
    total_steps: int = steps_per_epoch * train_config.epochs

    # DEV_MODE: cap at 100 steps for fast validation
    if dev_mode:
        total_steps = min(total_steps, 100)
        steps_per_epoch = min(steps_per_epoch, 100)
        print("*** DEV_MODE: capped at 100 steps ***")

    log_interval: int = 10 if dev_mode else 50

    # LR Scheduler: Warmup 5% + Cosine Decay
    warmup_steps: int = max(1, int(total_steps * 0.05))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"--- Starting La Pulga Training Loop on {DEVICE} ---")
    print(f"    Batch tokens: {batch_tokens:,} | Micro tokens: {micro_tokens:,} | Accum steps: {accum_steps}")
    print(f"    Seq len: {seq_len} | Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")
    print(f"    LR: {train_config.learning_rate} | Warmup: {warmup_steps} steps | Log every {log_interval} steps")

    for epoch in range(train_config.epochs):
        epoch_loss: float = 0.0

        for step in range(steps_per_epoch):
            global_step: int = epoch * steps_per_epoch + step
            t0: float = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            step_loss: float = 0.0

            for micro_step in range(accum_steps):
                # Read a micro-batch of tokens and build (x, y) pairs
                chunk = stream.take(micro_tokens + 1)
                x = chunk[:-1].reshape(-1, seq_len).to(device=DEVICE, dtype=torch.int64)
                y = chunk[1:].reshape(-1, seq_len).to(device=DEVICE, dtype=torch.int64)

                with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                    loss_val = compiled_model(x, y)
                    loss_val = loss_val / accum_steps  # scale for accumulation

                scaler.scale(loss_val).backward()
                step_loss += loss_val.item()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            t1: float = time.perf_counter()
            epoch_loss += step_loss

            if step % log_interval == 0:
                bpb_est: float = step_loss / math.log(2)
                lr_now: float = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch} | Step {step:04d}/{steps_per_epoch} | "
                      f"Loss {step_loss:.4f} | BPB ~{bpb_est:.4f} | "
                      f"LR {lr_now:.2e} | Time {t1-t0:.4f}s")

        avg_epoch_loss: float = epoch_loss / steps_per_epoch
        print(f"--- Epoch {epoch} complete | Avg Loss: {avg_epoch_loss:.4f} ---")

    orig_model = getattr(model, '_orig_mod', model)
    raw_state_dict = orig_model.state_dict()

    # Primary checkpoint: safetensors (safe, portable, no pickle)
    print("--- Saving Checkpoint (.safetensors) ---")
    cpu_state_dict = {k: v.cpu().contiguous() for k, v in raw_state_dict.items()}
    safetensors_save(cpu_state_dict, train_config.checkpoint_path)
    print(f"Checkpoint saved to {train_config.checkpoint_path}")

    # Submission artifact: int8 quantization + zlib compression (official format)
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
