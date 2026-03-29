"""
Inference script for La Pulga.
Loads a saved int8+zlib checkpoint and generates text from a prompt.

Usage:
    python scripts/infer.py --prompt "Once upon a time"
    python scripts/infer.py --prompt "The cat" --max-tokens 100 --temperature 0.9
"""
import argparse
import io
import zlib

import torch

from src.domain.config import ModelConfig, TrainingConfig
from src.model.transformer import LanguageModel, DEVICE
from src.data.tokenizer import load_sentencepiece
from src.utils.quantize import dequantize_state_dict_int8
from src.utils.generate import generate_text


def load_checkpoint(path: str, model: LanguageModel) -> None:
    with open(path, "rb") as f:
        compressed = f.read()
    raw = zlib.decompress(compressed)
    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    state_dict = dequantize_state_dict_int8(obj)
    model.load_state_dict(state_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with La Pulga")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Tokens to generate (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering (default: 50, 0=disabled)")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Repetition penalty (default: 1.2, 1.0=disabled)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: TrainingConfig.checkpoint_path)")
    args = parser.parse_args()

    train_config = TrainingConfig()
    checkpoint_path = args.checkpoint or train_config.checkpoint_path

    model_config = ModelConfig()
    model = LanguageModel(model_config).to(DEVICE)

    print(f"Loading checkpoint: {checkpoint_path}")
    load_checkpoint(checkpoint_path, model)
    print(f"Device: {DEVICE}")

    output = generate_text(
        model,
        load_sentencepiece(train_config.tokenizer_path),
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    print("-" * 40)
    print(output)
    print("-" * 40)


if __name__ == "__main__":
    main()
