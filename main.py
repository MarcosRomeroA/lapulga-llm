"""
Entrypoint for lapulga-llm.
Runs the initial smoke test locally using the MLX backend.
"""
import mlx.core as mx
from src.domain.config import ModelConfig, TrainingConfig
from src.model.mlx_backend import LanguageModel
from src.data.loader import tokenize_text
from src.training.loop import execute_training

def _create_fallback_data() -> str:
    """Returns fallback text if no dataset is found locally."""
    fallback_string: str = "La Pulga is small but exceptionally high-performing. " * 500
    try:
        with open("train.txt", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print("Warning: train.txt not found. Utilizing fallback generated text.")
        return fallback_string

def main() -> None:
    """Main execution block."""
    # 1. Domain Configs (Pure Python)
    model_config = ModelConfig()
    train_config = TrainingConfig()

    # 2. Framework Backend Setup
    mlx_model = LanguageModel(model_config)
    num_params: int = sum(v.size for _, v in mlx_model.parameters().items())
    print(f"Model Parameters: {num_params:,}")

    # 3. Data Interface Setup
    raw_text: str = _create_fallback_data()
    tokens: mx.array = tokenize_text(raw_text, model_config.vocab_size)

    # 4. Use Case Orchestrator Execution
    execute_training(model=mlx_model, tokens=tokens, train_config=train_config)

if __name__ == "__main__":
    main()
