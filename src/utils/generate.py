"""
Generation Use Case (Inference).
Implements the autoregressive loop to make La Pulga dream and generate text.
Uses La Pulga's custom BPE tokenizer for encode/decode.
"""
import mlx.core as mx
from tokenizers import Tokenizer
from src.model.mlx_backend import LanguageModel


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
    # 1. Tokenize input using our custom BPE (all IDs guaranteed in range)
    input_ids: list[int] = tokenizer.encode(prompt).ids

    print(f"Thinking with temperature {temperature}...")

    # 2. Generation Loop (Autoregressive)
    for _ in range(max_tokens):
        # Prepare context tensor [Batch=1, SequenceLength=N]
        input_tensor: mx.array = mx.array([input_ids])

        # Forward pass through all layers
        logits: mx.array = model(input_tensor)

        # Take the logits of the absolute last token generated in the sequence
        next_token_logits: mx.array = logits[0, -1, :]

        # 3. Sampling filter (Temperature logic)
        if temperature > 0:
            scaled_logits: mx.array = next_token_logits / temperature
            next_token_id: int = mx.random.categorical(scaled_logits).item()
        else:
            # Greedy search (always pick the highest score)
            next_token_id: int = mx.argmax(next_token_logits).item()

        # 4. Append to the sequence
        input_ids.append(next_token_id)

    # 5. Decode back to human language
    generated_story: str = tokenizer.decode(input_ids)
    return generated_story
