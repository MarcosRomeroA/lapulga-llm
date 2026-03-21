"""
Generation Use Case (Inference).
Implements the autoregressive loop to make La Pulga dream and generate text.
"""
import mlx.core as mx
import tiktoken
from src.model.mlx_backend import LanguageModel

def generate_text(model: LanguageModel, prompt: str, max_tokens: int = 50, temperature: float = 0.8) -> str:
    """
    Takes a string prompt, tokenizes it, and iteratively predicts the next tokens.
    Iterates autoregressively by appending the newly generated token back to the context.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    
    # 1. Tokenize input and clamp to our restricted 8k vocab size
    input_ids: list[int] = [t if t < model.vocab_size else 0 for t in enc.encode(prompt)]
    
    print(f"🧠 Thinking with temperature {temperature}... ")
    
    # 2. Generation Loop (Autoregressive)
    for _ in range(max_tokens):
        # Prepare context tensor [Batch=1, SequenceLength=N]
        input_tensor: mx.array = mx.array([input_ids])
        
        # Forward pass through all layers
        logits: mx.array = model(input_tensor)
        
        # Take the logits of the absolute last token generated in the sequence
        next_token_logits: mx.array = logits[0, -1, :]
        
        # 3. Sampling filter (Temperature logic)
        # Using mx.random.categorical allows statistical variety (creativity) rather than a repetitive bot
        if temperature > 0:
            scaled_logits: mx.array = next_token_logits / temperature
            # Categorical sampling natively drops a 0-D array, .item() converts it to pure Python int
            next_token_id: int = mx.random.categorical(scaled_logits).item()
        else:
            # Greedy search (always pick the highest score mathematically)
            next_token_id: int = mx.argmax(next_token_logits).item()
            
        # 4. Append to the sequence
        input_ids.append(next_token_id)

    # 5. Decode back to human language
    generated_story: str = enc.decode(input_ids)
    return generated_story
