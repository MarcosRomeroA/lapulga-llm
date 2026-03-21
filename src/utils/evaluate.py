"""
Perplexity Evaluation Module for La Pulga.
Measures how well the model predicts the next token on unseen validation data.
Lower perplexity = better model quality.
"""
import math
import torch
import torch.nn.functional as F
from src.model.transformer import LanguageModel, DEVICE
from src.data.loader import get_batches


@torch.no_grad()
def compute_perplexity(
    model: LanguageModel,
    val_tokens: torch.Tensor,
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
    model.eval()
    total_loss: float = 0.0
    total_batches: int = 0

    for batch_inputs, batch_targets in get_batches(val_tokens, batch_size, context_length, device=DEVICE):
        logits = model(batch_inputs)
        logits_fp32 = logits.float()

        loss = F.cross_entropy(
            logits_fp32.reshape(-1, logits.shape[-1]),
            batch_targets.reshape(-1),
        )
        batch_loss: float = loss.item()

        if not math.isnan(batch_loss):
            total_loss += batch_loss
            total_batches += 1

    if total_batches == 0:
        return float("inf")

    avg_loss: float = total_loss / total_batches
    perplexity: float = math.exp(avg_loss)
    return perplexity
