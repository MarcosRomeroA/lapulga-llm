"""
Training Orchestrator for Lapulga.
Connects domain configurations, data loaders, and framework backends.
"""
import time
import os
import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
from src.domain.config import ModelConfig, TrainingConfig
from src.model.mlx_backend import LanguageModel, cross_entropy_loss
from src.data.loader import get_batches

def execute_training(model: LanguageModel, tokens: mx.array, train_config: TrainingConfig) -> None:
    """
    Executes the main optimization loop tracking loss over time.
    """
    optimizer = optim.AdamW(learning_rate=train_config.learning_rate)

    def step(input_tensors: mx.array, target_tokens: mx.array) -> mx.array:
        loss_and_grad_fn = nn.value_and_grad(model, cross_entropy_loss)
        loss_val, grads = loss_and_grad_fn(model, input_tensors, target_tokens)
        optimizer.update(model, grads)
        return loss_val

    print("--- 🚀 Starting La Pulga Training Loop ---")

    for epoch in range(train_config.epochs):
        batch_generator = get_batches(tokens, train_config.batch_size, train_config.context_length)
        
        for batch_idx, (b_inputs, b_targets) in enumerate(batch_generator):
            t0: float = time.perf_counter()
            loss_val = step(b_inputs, b_targets)
            mx.eval(loss_val, model.parameters(), optimizer.state) # Ensure computation completes synchronously
            t1: float = time.perf_counter()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:04d} | Loss {loss_val.item():.4f} | Time {t1-t0:.4f}s")
                
    # Model Export
    print(f"--- 💾 Saving Model Weights ---")
    # 🚨 CRITICAL FIX: Cast to Float16 ONLY right before saving. 
    # This keeps AdamW optimizers and gradients in mathematically stable FP32 during the loop!
    import mlx.utils as utils
    model.update(utils.tree_map(lambda arr: arr.astype(mx.float16), model.parameters()))
    
    model.save_weights(train_config.checkpoint_path)
    actual_size_bytes = os.path.getsize(train_config.checkpoint_path)
    print(f"Model saved to {train_config.checkpoint_path}")
    print(f"Total Disk Size: {actual_size_bytes / (1024*1024):.2f} MiB")
