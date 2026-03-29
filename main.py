"""
Entrypoint for lapulga-llm.
Trains on FineWeb shards, evaluates with BPB, and reports artifact size.
"""
import os
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, assuming env vars are set externally
import torch
from pathlib import Path
from src.domain.config import ModelConfig, TrainingConfig
from src.model.transformer import LanguageModel, DEVICE
from src.data.shard_loader import load_validation_tokens
from src.data.tokenizer import load_sentencepiece, build_bpb_lookup_tables
from src.training.loop import execute_training
from src.utils.evaluate_bpb import compute_bpb
from src.utils.quantize import compute_artifact_size
from src.utils.generate import generate_text


def main() -> None:
    """Main execution block."""
    # Enable TF32 on Ampere+ GPUs for faster matmuls
    torch.set_float32_matmul_precision("high")

    # 1. Domain Configs (Pure Python)
    model_config = ModelConfig()
    train_config = TrainingConfig()

    # 2. Framework Backend Setup
    model = LanguageModel(model_config).to(DEVICE)
    model.train()
    num_params: int = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params:,}")
    print(f"Device: {DEVICE}")

    # 3. Load SentencePiece tokenizer + BPB lookup tables
    sp = load_sentencepiece(train_config.tokenizer_path)
    print(f"SentencePiece tokenizer loaded: {sp.vocab_size()} tokens")
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_bpb_lookup_tables(
        sp, model_config.vocab_size, DEVICE,
    )

    # 4. Load validation tokens
    val_pattern: str = os.path.join(train_config.data_path, "fineweb_val_*.bin")
    val_tokens = load_validation_tokens(val_pattern, train_config.context_length)
    print(f"Validation tokens: {val_tokens.numel():,}")

    # 5. Training (reads FineWeb shards internally via TokenStream)
    execute_training(model=model, train_config=train_config)

    # 6. BPB Evaluation (official scoring metric)
    print("\n--- Measuring BPB ---")
    val_loss, val_bpb = compute_bpb(
        model, val_tokens, train_config.context_length,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        device=DEVICE,
    )
    print(f"Validation Loss: {val_loss:.4f} | Validation BPB: {val_bpb:.4f}")

    # 7. Artifact Size Report
    print("\n--- Artifact Size ---")
    code_path = Path(__file__)
    total_bytes, code_bytes, model_bytes = compute_artifact_size(model.state_dict(), code_path)
    print(f"Code: {code_bytes:,} bytes | Model (int8+zlib): {model_bytes:,} bytes")
    print(f"Total: {total_bytes:,} / 16,000,000 bytes")
    if total_bytes < 16_000_000:
        print("PASS: Under 16MB limit")
    else:
        print("FAIL: Exceeds 16MB limit!")

    # 8. Text Generation (quick sample)
    print("\n--- Generating Text ---")
    sample: str = generate_text(model, sp, "The little dog", max_tokens=60, temperature=0.7, top_k=50, repetition_penalty=1.2)
    print("-" * 40)
    print(f"Output:\n\n{sample}")
    print("-" * 40)


if __name__ == "__main__":
    main()
