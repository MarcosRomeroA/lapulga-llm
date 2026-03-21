"""
Entrypoint for lapulga-llm.
Runs the initial smoke test locally using the MLX backend.
"""
import mlx.core as mx
import mlx.utils as utils
from src.domain.config import ModelConfig, TrainingConfig
from src.model.mlx_backend import LanguageModel
from src.data.dataset import fetch_tinystories_stream
from src.data.loader import tokenize_stream
from src.training.loop import execute_training
from src.utils.generate import generate_text

def main() -> None:
    """Main execution block."""
    # 1. Domain Configs (Pure Python)
    model_config = ModelConfig()
    train_config = TrainingConfig()

    # 2. Framework Backend Setup
    mlx_model = LanguageModel(model_config)
    num_params: int = sum(v.size for _, v in utils.tree_flatten(mlx_model.parameters()))
    print(f"Model Parameters: {num_params:,}")

    # 3. Data Interface Setup (Clean Architecture Streaming)
    print("--- 📥 Streaming TinyStories from HuggingFace ---")
    data_stream = fetch_tinystories_stream(split="train")
    
    # Cap to 250k tokens for local fast iteration
    tokens: mx.array = tokenize_stream(data_stream, model_config.vocab_size, target_tokens=250_000)
    print(f"Loaded {len(tokens):,} tokens for training.")

    # 4. Use Case Orchestrator Execution
    execute_training(model=mlx_model, tokens=tokens, train_config=train_config)

    # 5. Evaluation / Text Generation
    print("\n--- 🗣️ Evaluating La Pulga ---")
    test_prompt: str = "Once upon a time, there was a little dog"
    output_text: str = generate_text(mlx_model, test_prompt, max_tokens=40, temperature=0.7)
    
    print("-" * 40)
    print(f"Output:\n\n{output_text}")
    print("-" * 40)

if __name__ == "__main__":
    main()
