"""
Spec Compliance Test Suite for lapulga-llm.
Parses SPEC.md and validates the instantiated model against every architectural constraint.
Run with: uv run pytest tests/test_spec_compliance.py -v
"""
import math
import re
import unittest
from pathlib import Path

import mlx.utils as mlx_utils
import yaml

# The project root is two levels up from this file (tests/)
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
        from src.model.mlx_backend import LanguageModel

        cls.model_config = ModelConfig(
            dim=cls.spec["n_embd"],
            n_layers=cls.spec["n_layers"],
            n_heads=cls.spec["n_head"],
            n_kv_heads=cls.spec["n_kv_heads"],
            hidden_dim=cls.spec["hidden_dim"],
            vocab_size=cls.spec["vocab_size"],
        )
        cls.model = LanguageModel(cls.model_config)

        flat_params = mlx_utils.tree_flatten(cls.model.parameters())
        cls.total_params: int = sum(arr.size for _, arr in flat_params)

    # ── Architectural Shape Tests ───────────────────────────────────────────

    def test_n_layers_matches_spec(self) -> None:
        """Verify that the number of Transformer blocks matches SPEC.md."""
        expected_layers: int = self.spec["n_layers"]
        actual_layers: int = len(self.model.layers)
        self.assertEqual(
            actual_layers,
            expected_layers,
            f"Layer count mismatch: model has {actual_layers}, SPEC requires {expected_layers}",
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

    # ── Parameter Budget Tests ──────────────────────────────────────────────

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

    # ── Artifact Size Tests ─────────────────────────────────────────────────

    def test_fp16_artifact_within_16mib(self) -> None:
        """
        Simulates FP16 export: total_params × 2 bytes must stay under max_size_mib.
        This is the hard constraint set by the Parameter Golf challenge rules.
        """
        max_size_mib: float = self.spec["constraints"]["max_size_mib"]
        fp16_bytes: int = self.total_params * 2  # 2 bytes per FP16 float
        fp16_mib: float = fp16_bytes / (1024 * 1024)

        self.assertLess(
            fp16_mib,
            max_size_mib,
            f"FP16 artifact size {fp16_mib:.2f} MiB exceeds the {max_size_mib} MiB limit",
        )

    def test_int4_quantization_headroom(self) -> None:
        """
        Simulates 4-bit quantization: total_params × 0.5 bytes.
        We must fit under 16 MiB with significant headroom for INT4 experiments.
        """
        max_size_mib: float = self.spec["constraints"]["max_size_mib"]
        int4_bytes: int = self.total_params // 2  # 0.5 bytes per INT4 nibble
        int4_mib: float = int4_bytes / (1024 * 1024)

        self.assertLess(
            int4_mib,
            max_size_mib,
            f"INT4 artifact size {int4_mib:.2f} MiB exceeds the {max_size_mib} MiB limit",
        )
        # Also report the headroom available for parameter scaling
        headroom_mib: float = max_size_mib - int4_mib
        print(f"\n[INT4] Size: {int4_mib:.2f} MiB | Headroom: {headroom_mib:.2f} MiB")


if __name__ == "__main__":
    unittest.main(verbosity=2)
