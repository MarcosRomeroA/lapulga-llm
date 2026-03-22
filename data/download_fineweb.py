"""
FineWeb Data Download Script for La Pulga.
Downloads pre-tokenized FineWeb shards and SentencePiece tokenizer from HuggingFace Hub.

Usage:
    uv run python data/download_fineweb.py --variant sp1024 --train-shards 10
    uv run python data/download_fineweb.py --variant sp1024 --train-shards 1   # smoke test
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID: str = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX: str = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
ROOT: Path = Path(__file__).resolve().parent
DATASETS_DIR: Path = ROOT / "datasets"
TOKENIZERS_DIR: Path = ROOT / "tokenizers"


def dataset_dir_for_variant(name: str) -> str:
    """Maps a tokenizer variant name to its dataset directory name."""
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"Unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")


def local_path_for_remote(relative_path: str) -> Path:
    """Resolves a remote artifact path to its local filesystem location."""
    remote_path = Path(relative_path)
    if REMOTE_ROOT_PREFIX and remote_path.parts[:1] == (REMOTE_ROOT_PREFIX,):
        remote_path = remote_path.relative_to(REMOTE_ROOT_PREFIX)
    if remote_path.parts[:1] == ("datasets",):
        return DATASETS_DIR.joinpath(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return TOKENIZERS_DIR.joinpath(*remote_path.parts[1:])
    return ROOT / remote_path


def download_artifact(relative_path: str) -> None:
    """Downloads a single artifact from HuggingFace Hub if not already cached locally."""
    destination = local_path_for_remote(relative_path)
    if destination.exists():
        return
    if destination.is_symlink():
        destination.unlink()

    remote_path = Path(relative_path)
    cached_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type="dataset",
        )
    )
    cached_source = cached_path.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached_source, destination)
    except OSError:
        shutil.copy2(cached_source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FineWeb shards for Parameter-Golf challenge")
    parser.add_argument("--train-shards", type=int, default=10, help="Number of training shards to download (default: 10)")
    parser.add_argument("--variant", default="sp1024", help="Tokenizer variant: sp1024, sp4096, or byte260")
    args = parser.parse_args()

    dataset_dir: str = dataset_dir_for_variant(args.variant)

    # Download manifest
    manifest_remote = f"{REMOTE_ROOT_PREFIX}/manifest.json"
    download_artifact(manifest_remote)
    manifest_local = local_path_for_remote(manifest_remote)
    manifest: dict = json.loads(manifest_local.read_text(encoding="utf-8"))

    # Find dataset entry
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir), None)
    if dataset_entry is None:
        raise ValueError(f"Dataset {dataset_dir} not found in manifest")
    max_train: int = int((dataset_entry.get("stats") or {}).get("files_train"))
    val_shards: int = int((dataset_entry.get("stats") or {}).get("files_val"))
    if args.train_shards > max_train:
        raise ValueError(f"{args.variant} only has {max_train} training shards, requested {args.train_shards}")

    # Download tokenizer
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if tokenizer_entry is None:
        raise ValueError(f"Tokenizer {tokenizer_name} not found in manifest")
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            print(f"Downloading tokenizer artifact: {value}")
            download_artifact(f"{REMOTE_ROOT_PREFIX}/{value}")

    # Download validation shards (always all)
    dataset_prefix = f"{REMOTE_ROOT_PREFIX}/datasets/{dataset_dir}"
    print(f"Downloading {val_shards} validation shard(s)...")
    for i in range(val_shards):
        download_artifact(f"{dataset_prefix}/fineweb_val_{i:06d}.bin")

    # Download training shards
    print(f"Downloading {args.train_shards} training shard(s)...")
    for i in range(args.train_shards):
        download_artifact(f"{dataset_prefix}/fineweb_train_{i:06d}.bin")

    print(f"Done. Data at: {DATASETS_DIR / dataset_dir}")
    print(f"Tokenizer at: {TOKENIZERS_DIR}")


if __name__ == "__main__":
    main()
