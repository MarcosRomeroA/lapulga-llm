"""
Generation Use Case (Inference).
Implements the autoregressive loop to make La Pulga dream and generate text.
Uses La Pulga's custom BPE tokenizer for encode/decode.
"""
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from src.model.transformer import LanguageModel


@torch.no_grad()
def generate_text(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.8,
) -> str:
    """
    Takes a string prompt, tokenizes it, and iteratively predicts the next tokens.
    Iterates autoregressively by appending the newly generated token back to the context.
    """
    model.eval()
    device = model.device

    input_ids: list[int] = tokenizer.encode(prompt).ids
    print(f"Thinking with temperature {temperature}...")

    for _ in range(max_tokens):
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        logits = model(input_tensor)
        next_token_logits = logits[0, -1, :]

        if temperature > 0:
            scaled_logits = next_token_logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            next_token_id: int = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token_id: int = torch.argmax(next_token_logits).item()

        input_ids.append(next_token_id)

    generated_story: str = tokenizer.decode(input_ids)
    return generated_story
