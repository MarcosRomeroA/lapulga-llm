"""
Perplexity Evaluation Module for La Pulga.
Measures how well the model predicts the next token on unseen validation data.
Lower perplexity = better model quality.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from src.model.mlx_backend import LanguageModel
from src.data.loader import get_batches


def compute_perplexity(
    model: LanguageModel,
    val_tokens: mx.array,
    batch_size: int = 8,
    context_length: int = 256,
) -> float:
    """
    Computes perplexity over a validation set.

    Why perplexity?
    Perplexity = exp(average cross-entropy loss). It represents how many tokens
    the model is "confused" between on average. A perplexity of 1.0 means perfect
    prediction. For the Parameter Golf challenge, lower perplexity = higher score.
    """
    total_loss: float = 0.0
    total_batches: int = 0

    for batch_inputs, batch_targets in get_batches(val_tokens, batch_size, context_length):
        logits: mx.array = model(batch_inputs)

        # Upcast to FP32 for stable softmax (same fix as training)
        logits_fp32: mx.array = logits.astype(mx.float32)

        loss: mx.array = nn.losses.cross_entropy(
            logits_fp32.reshape(-1, logits.shape[-1]),
            batch_targets.reshape(-1),
        )
        batch_loss: float = mx.mean(loss).item()

        if not math.isnan(batch_loss):
            total_loss += batch_loss
            total_batches += 1

    if total_batches == 0:
        return float("inf")

    avg_loss: float = total_loss / total_batches
    perplexity: float = math.exp(avg_loss)
    return perplexity
