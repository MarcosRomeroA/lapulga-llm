"""
Generation Use Case (Inference).
Implements the autoregressive loop to make La Pulga dream and generate text.
Uses SentencePiece tokenizer for encode/decode (official Parameter-Golf format).
"""
import torch
import torch.nn.functional as F
import sentencepiece as spm
from src.model.transformer import LanguageModel


@torch.compiler.disable
@torch.no_grad()
def generate_text(
    model: LanguageModel,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> str:
    """
    Takes a string prompt, tokenizes it, and iteratively predicts the next tokens.
    Iterates autoregressively by appending the newly generated token back to the context.
    """
    model.eval()
    device = model.device
    ctx_len: int = model.config.context_length

    input_ids: list[int] = sp.encode(prompt)
    print(f"Thinking with temperature {temperature}...")

    for _ in range(max_tokens):
        # Truncate to context window
        context = input_ids[-ctx_len:]
        input_tensor = torch.tensor([context], dtype=torch.long, device=device)

        logits = model(input_tensor)
        next_token_logits = logits[0, -1, :].clone()

        # Repetition penalty (HuggingFace standard)
        if repetition_penalty != 1.0:
            seen = set(input_ids)
            for token_id in seen:
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= repetition_penalty
                else:
                    next_token_logits[token_id] *= repetition_penalty

        if temperature > 0:
            scaled_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k > 0 and top_k < scaled_logits.size(0):
                top_k_vals, top_k_idx = torch.topk(scaled_logits, top_k)
                filtered = torch.full_like(scaled_logits, float('-inf'))
                filtered.scatter_(0, top_k_idx, top_k_vals)
                scaled_logits = filtered

            probs = F.softmax(scaled_logits, dim=-1)
            next_token_id: int = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token_id: int = torch.argmax(next_token_logits).item()

        input_ids.append(next_token_id)

    generated_text: str = sp.decode(input_ids)
    return generated_text
