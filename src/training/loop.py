"""
Training Orchestrator for Lapulga.
Connects domain configurations, data loaders, and the PyTorch backend.
Supports CUDA Auto Mixed Precision (AMP) for high-throughput training.
"""
import time
import os
import torch
from src.domain.config import TrainingConfig
from src.model.transformer import LanguageModel, DEVICE, cross_entropy_loss
from src.data.loader import get_batches


def execute_training(model: LanguageModel, tokens: torch.Tensor, train_config: TrainingConfig) -> None:
    """
    Executes the main optimization loop tracking loss over time.
    Trains in FP32 (with optional AMP on CUDA) for stability,
    then exports weights in FP16 to meet the 16MB limit.
    """
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    use_amp: bool = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    print(f"--- Starting La Pulga Training Loop on {DEVICE} ---")

    for epoch in range(train_config.epochs):
        epoch_loss: float = 0.0
        epoch_batches: int = 0
        batch_generator = get_batches(tokens, train_config.batch_size, train_config.context_length, device=DEVICE)

        for batch_idx, (b_inputs, b_targets) in enumerate(batch_generator):
            t0: float = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                loss_val = cross_entropy_loss(model, b_inputs, b_targets)

            scaler.scale(loss_val).backward()
            scaler.step(optimizer)
            scaler.update()

            t1: float = time.perf_counter()

            loss_float: float = loss_val.item()
            epoch_loss += loss_float
            epoch_batches += 1

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:04d} | Loss {loss_float:.4f} | Time {t1-t0:.4f}s")

        avg_epoch_loss: float = epoch_loss / max(epoch_batches, 1)
        print(f"--- Epoch {epoch} complete | Avg Loss: {avg_epoch_loss:.4f} | Batches: {epoch_batches} ---")

    # Model Export: save state_dict in FP16 to meet 16MiB budget
    print("--- Saving Model Weights ---")
    fp16_state_dict = {k: v.half() for k, v in model.state_dict().items()}
    torch.save(fp16_state_dict, train_config.checkpoint_path)
    actual_size_bytes: int = os.path.getsize(train_config.checkpoint_path)
    print(f"Model saved to {train_config.checkpoint_path}")
    print(f"Total Disk Size: {actual_size_bytes / (1024*1024):.2f} MiB")
