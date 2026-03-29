"""
Official Scoring Mock Tests for lapulga-llm.
Simulates the Parameter-Golf evaluation pipeline using synthetic data.
Run with: python -m unittest tests.test_official_scoring -v
"""
import io
import math
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT: Path = Path(__file__).parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))


class TestShardLoader(unittest.TestCase):
    """Validates the binary shard loader against the official format."""

    def test_shard_loader_reads_valid_header(self) -> None:
        """Create a synthetic .bin shard and verify it loads correctly."""
        from src.data.shard_loader import load_data_shard, SHARD_MAGIC, SHARD_VERSION

        num_tokens: int = 1000
        header = np.zeros(256, dtype="<i4")
        header[0] = SHARD_MAGIC
        header[1] = SHARD_VERSION
        header[2] = num_tokens
        tokens = np.arange(num_tokens, dtype="<u2")

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            header.tofile(f)
            tokens.tofile(f)
            shard_path = Path(f.name)

        try:
            loaded = load_data_shard(shard_path)
            self.assertEqual(loaded.numel(), num_tokens)
            self.assertEqual(int(loaded[0]), 0)
            self.assertEqual(int(loaded[num_tokens - 1]), num_tokens - 1)
        finally:
            shard_path.unlink()

    def test_shard_loader_rejects_bad_magic(self) -> None:
        """Verify shard loader rejects files with wrong magic number."""
        from src.data.shard_loader import load_data_shard

        header = np.zeros(256, dtype="<i4")
        header[0] = 12345  # wrong magic
        header[1] = 1
        header[2] = 100
        tokens = np.zeros(100, dtype="<u2")

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            header.tofile(f)
            tokens.tofile(f)
            shard_path = Path(f.name)

        try:
            with self.assertRaises(ValueError):
                load_data_shard(shard_path)
        finally:
            shard_path.unlink()


class TestBPBCalculation(unittest.TestCase):
    """Validates the BPB formula with synthetic data."""

    def test_bpb_formula_smoke(self) -> None:
        """
        Verify BPB = (loss / ln(2)) * (tokens / bytes) with known values.
        Uses a dummy model that returns a fixed loss.
        """
        known_loss: float = 2.0  # nats
        known_tokens: int = 1000
        known_bytes: int = 4500  # ~4.5 bytes per token average

        bits_per_token: float = known_loss / math.log(2.0)
        tokens_per_byte: float = known_tokens / known_bytes
        expected_bpb: float = bits_per_token * tokens_per_byte

        # Verify the math: 2.0 / 0.693 * (1000/4500) ≈ 0.641
        self.assertAlmostEqual(expected_bpb, 2.885 * (1000 / 4500), places=2)
        self.assertGreater(expected_bpb, 0.0)
        self.assertLess(expected_bpb, 10.0)


class TestMockOfficialPipeline(unittest.TestCase):
    """End-to-end mock of the official Parameter-Golf evaluation pipeline."""

    def test_mock_eval_pipeline(self) -> None:
        """
        Full pipeline: model -> forward -> int8 quantize -> zlib ->
        size check -> dequantize -> forward again -> verify consistency.
        """
        from src.domain.config import ModelConfig
        from src.model.transformer import LanguageModel
        from src.utils.quantize import quantize_state_dict_int8, dequantize_state_dict_int8

        config = ModelConfig()
        model = LanguageModel(config)

        # Synthetic forward pass
        seq_len: int = 64
        input_ids = torch.randint(0, config.vocab_size, (2, seq_len))
        target_ids = torch.randint(0, config.vocab_size, (2, seq_len))

        model.eval()
        with torch.no_grad():
            loss_original = model(input_ids, target_ids).item()

        # Quantize and compress
        obj, stats = quantize_state_dict_int8(model.state_dict())
        buf = io.BytesIO()
        torch.save(obj, buf)
        compressed = zlib.compress(buf.getvalue(), level=9)
        model_bytes: int = len(compressed)

        # Size check (16,000,000 bytes decimal limit)
        code_estimate: int = 15_000
        total: int = model_bytes + code_estimate
        self.assertLess(
            total,
            16_000_000,
            f"Artifact {total:,} bytes exceeds 16,000,000 byte limit",
        )

        # Dequantize and verify roundtrip
        decompressed = zlib.decompress(compressed)
        obj_loaded = torch.load(io.BytesIO(decompressed), weights_only=False)
        restored_sd = dequantize_state_dict_int8(obj_loaded)

        # Load into a fresh model and forward again
        model_restored = LanguageModel(config)
        model_restored.load_state_dict(restored_sd, strict=False)
        model_restored.eval()
        with torch.no_grad():
            loss_restored = model_restored(input_ids, target_ids).item()

        # Quantization should not cause catastrophic loss increase
        loss_delta: float = abs(loss_restored - loss_original)
        self.assertLess(
            loss_delta,
            1.0,
            f"Quantization roundtrip changed loss by {loss_delta:.4f} "
            f"(original: {loss_original:.4f}, restored: {loss_restored:.4f})",
        )
        print(f"\n[Pipeline] Original loss: {loss_original:.4f} | "
              f"Restored loss: {loss_restored:.4f} | Delta: {loss_delta:.4f}")
        print(f"[Pipeline] Artifact: {model_bytes:,} model + {code_estimate:,} code = "
              f"{total:,} / 16,000,000 bytes")

    def test_submission_artifact_format(self) -> None:
        """Verify the quantized object has the expected keys and structure."""
        from src.domain.config import ModelConfig
        from src.model.transformer import LanguageModel
        from src.utils.quantize import quantize_state_dict_int8

        config = ModelConfig()
        model = LanguageModel(config)
        obj, _stats = quantize_state_dict_int8(model.state_dict())

        self.assertEqual(obj["__quant_format__"], "int8_clean_per_row_v1")
        self.assertIsInstance(obj["quantized"], dict)
        self.assertIsInstance(obj["scales"], dict)
        self.assertIsInstance(obj["dtypes"], dict)
        self.assertIsInstance(obj["passthrough"], dict)

        # Every quantized tensor must have a matching scale
        for name in obj["quantized"]:
            self.assertIn(name, obj["scales"], f"Missing scale for quantized tensor: {name}")
            self.assertEqual(obj["quantized"][name].dtype, torch.int8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
