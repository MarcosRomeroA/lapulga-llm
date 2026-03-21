"""
Entrypoint for lapulga-llm.
Runs the training loop locally using the MLX backend with custom BPE tokenizer.
"""
import mlx.core as mx
import mlx.utils as utils
from src.domain.config import ModelConfig, TrainingConfig
from src.model.mlx_backend import LanguageModel
from src.data.dataset import fetch_tinystories_stream
from src.data.loader import load_tokenizer, tokenize_stream
from src.training.loop import execute_training
from src.utils.generate import generate_text
from src.utils.evaluate import compute_perplexity


def main() -> None:
    """Main execution block."""
    # 1. Domain Configs (Pure Python)
    model_config = ModelConfig()
    train_config = TrainingConfig()

    # 2. Framework Backend Setup
    mlx_model = LanguageModel(model_config)
    num_params: int = sum(v.size for _, v in utils.tree_flatten(mlx_model.parameters()))
    print(f"Model Parameters: {num_params:,}")

    # 3. Load custom BPE tokenizer
    tokenizer = load_tokenizer(train_config.tokenizer_path)
    print(f"Tokenizer loaded: {tokenizer.get_vocab_size()} tokens")

    # 4. Data Interface Setup (Train + Validation splits)
    print("--- Loading TinyStories ---")
    train_stream = fetch_tinystories_stream(split="train")
    tokens: mx.array = tokenize_stream(train_stream, tokenizer, target_tokens=train_config.train_tokens)
    print(f"Train tokens: {len(tokens):,}")

    val_stream = fetch_tinystories_stream(split="validation")
    val_tokens: mx.array = tokenize_stream(val_stream, tokenizer, target_tokens=train_config.val_tokens)
    print(f"Validation tokens: {len(val_tokens):,}")

    # 5. Training
    execute_training(model=mlx_model, tokens=tokens, train_config=train_config)

    # 6. Perplexity Evaluation
    print("\n--- Measuring Perplexity ---")
    ppl: float = compute_perplexity(
        mlx_model, val_tokens,
        batch_size=train_config.batch_size,
        context_length=train_config.context_length,
    )
    print(f"Validation Perplexity: {ppl:.2f}")

    # 7. Text Generation
    print("\n--- Evaluating La Pulga ---")
    test_prompt: str = "Once upon a time, there was a little dog"
    output_text: str = generate_text(mlx_model, tokenizer, test_prompt, max_tokens=60, temperature=0.7)

    print("-" * 40)
    print(f"Output:\n\n{output_text}")
    print("-" * 40)


if __name__ == "__main__":
    main()
