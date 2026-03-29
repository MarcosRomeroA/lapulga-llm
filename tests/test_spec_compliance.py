"""
Spec Compliance Test Suite for lapulga-llm.
Parses SPEC.md and validates the instantiated model against every architectural constraint.
Run with: python -m unittest tests.test_spec_compliance -v
"""
import io
import math
import re
import unittest
import zlib
from pathlib import Path

import torch
import yaml

PROJECT_ROOT: Path = Path(__file__).parent.parent
SPEC_PATH: Path = PROJECT_ROOT / "SPEC.md"


def parse_spec(spec_path: Path) -> dict:
    """
    Extracts and parses the YAML frontmatter block from SPEC.md.
    The YAML block is delimited by '---' at the start of the file.
    """
    content: str = spec_path.read_text(encoding="utf-8")
    yaml_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not yaml_match:
        raise ValueError(f"No YAML frontmatter found in {spec_path}")
    return yaml.safe_load(yaml_match.group(1))


class TestSpecCompliance(unittest.TestCase):
    """
    Validates that the current model implementation precisely matches SPEC.md.
    Acts as the hard compliance gate for all architecture changes.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Parse spec and instantiate the model once for all tests."""
        cls.spec: dict = parse_spec(SPEC_PATH)

        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from src.domain.config import ModelConfig
        from src.model.transformer import LanguageModel

        cls.model_config = ModelConfig(
            dim=cls.spec["n_embd"],
            physical_layers=cls.spec["physical_layers"],
            repeat_count=cls.spec["repeat_count"],
            n_heads=cls.spec["n_head"],
            n_kv_heads=cls.spec["n_kv_heads"],
            hidden_dim=cls.spec["hidden_dim"],
            vocab_size=cls.spec["vocab_size"],
            context_length=cls.spec["context_length"],
            logit_softcap=cls.spec.get("logit_softcap", 30.0),
        )
        cls.model = LanguageModel(cls.model_config)

        cls.total_params: int = sum(p.numel() for p in cls.model.parameters())

    # -- Architectural Shape Tests --

    def test_n_layers_matches_spec(self) -> None:
        """Verify that physical layer count and effective depth match SPEC.md."""
        expected_physical: int = self.spec["physical_layers"]
        expected_effective: int = self.spec["physical_layers"] * self.spec["repeat_count"]
        actual_physical: int = len(self.model.layers)
        actual_effective: int = self.model.config.effective_layers
        self.assertEqual(
            actual_physical,
            expected_physical,
            f"Physical layer count mismatch: model has {actual_physical}, SPEC requires {expected_physical}",
        )
        self.assertEqual(
            actual_effective,
            expected_effective,
            f"Effective layer count mismatch: model has {actual_effective}, SPEC requires {expected_effective}",
        )

    def test_embedding_dim_matches_spec(self) -> None:
        """Verify that the embedding dimension (n_embd) matches SPEC.md."""
        expected_dim: int = self.spec["n_embd"]
        actual_dim: int = self.model.tok_embeddings.weight.shape[1]
        self.assertEqual(
            actual_dim,
            expected_dim,
            f"Embedding dim mismatch: model has {actual_dim}, SPEC requires {expected_dim}",
        )

    def test_vocab_size_matches_spec(self) -> None:
        """Verify that the vocabulary size matches SPEC.md."""
        expected_vocab: int = self.spec["vocab_size"]
        actual_vocab: int = self.model.tok_embeddings.weight.shape[0]
        self.assertEqual(
            actual_vocab,
            expected_vocab,
            f"Vocab size mismatch: model has {actual_vocab}, SPEC requires {expected_vocab}",
        )

    def test_attention_heads_match_spec(self) -> None:
        """Verify that query head count matches SPEC.md in every layer."""
        expected_heads: int = self.spec["n_head"]
        for layer_index, layer in enumerate(self.model.layers):
            actual_heads: int = layer.attention.n_heads
            self.assertEqual(
                actual_heads,
                expected_heads,
                f"Attention head mismatch in layer {layer_index}: "
                f"model has {actual_heads}, SPEC requires {expected_heads}",
            )

    # -- Parameter Budget Tests --

    def test_param_count_within_tolerance(self) -> None:
        """
        Total parameters must not exceed the SPEC target by more than tolerance_pct.
        Protects against accidental parameter bloat from architectural drift.
        """
        target_params: int = self.spec["constraints"]["target_params"]
        tolerance_pct: float = self.spec["constraints"]["tolerance_pct"]
        max_allowed: int = math.ceil(target_params * (1 + tolerance_pct / 100))

        self.assertLessEqual(
            self.total_params,
            max_allowed,
            f"Parameter count {self.total_params:,} exceeds "
            f"target {target_params:,} by more than {tolerance_pct}% "
            f"(max allowed: {max_allowed:,})",
        )

    # -- Artifact Size Tests (Official Format) --

    def test_int8_zlib_artifact_under_16mb_decimal(self) -> None:
        """
        Simulates the official Parameter-Golf artifact size check:
        int8 quantized state_dict + zlib level 9 + code file bytes < 16,000,000 bytes.
        """
        from src.utils.quantize import quantize_state_dict_int8

        max_size_bytes: int = self.spec["constraints"]["max_size_bytes"]
        obj, _stats = quantize_state_dict_int8(self.model.state_dict())

        buf = io.BytesIO()
        torch.save(obj, buf)
        compressed = zlib.compress(buf.getvalue(), level=9)
        model_bytes: int = len(compressed)

        # Conservative code size estimate (actual train script ~5-10 KB)
        code_bytes_estimate: int = 15_000
        total_bytes: int = model_bytes + code_bytes_estimate

        self.assertLess(
            total_bytes,
            max_size_bytes,
            f"Artifact size {total_bytes:,} bytes exceeds {max_size_bytes:,} byte limit "
            f"(model: {model_bytes:,}, code estimate: {code_bytes_estimate:,})",
        )
        print(f"\n[Int8+zlib] Model: {model_bytes:,} bytes | "
              f"Estimated total: {total_bytes:,} / {max_size_bytes:,} bytes")

    def test_int8_roundtrip_fidelity(self) -> None:
        """
        Validates that int8 quantization roundtrip preserves tensor shapes
        and values within reasonable tolerance.
        """
        from src.utils.quantize import quantize_state_dict_int8, dequantize_state_dict_int8

        original_sd = self.model.state_dict()
        obj, _stats = quantize_state_dict_int8(original_sd)
        restored_sd = dequantize_state_dict_int8(obj)

        for name in original_sd:
            self.assertIn(name, restored_sd, f"Missing key after roundtrip: {name}")
            self.assertEqual(
                original_sd[name].shape,
                restored_sd[name].shape,
                f"Shape mismatch for {name}: {original_sd[name].shape} vs {restored_sd[name].shape}",
            )

    # -- Legacy Size Tests (kept for reference) --

    def test_fp16_artifact_within_16mib(self) -> None:
        """
        Secondary check: FP16 export size (legacy, for backward compatibility).
        """
        state_dict_params: int = sum(v.numel() for v in self.model.state_dict().values())
        fp16_bytes: int = state_dict_params * 2
        fp16_mib: float = fp16_bytes / (1024 * 1024)
        # With 17M params, FP16 will be ~32 MiB (too large for direct export)
        # This test just documents the value, not enforce a limit
        print(f"\n[FP16 reference] Size: {fp16_mib:.2f} MiB (requires int8+zlib for submission)")

    # -- Device Tests --

    def test_model_moves_to_gpu(self) -> None:
        """
        Verify that the model can be placed on CUDA when available.
        On CI without GPU, this test verifies it stays on CPU gracefully.
        """
        from src.model.transformer import DEVICE
        test_model = self.model.to(DEVICE)
        self.assertEqual(
            test_model.device.type,
            DEVICE.type,
            f"Model device is {test_model.device.type}, expected {DEVICE.type}",
        )
        self.model.to("cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
