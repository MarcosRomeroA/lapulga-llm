"""
BPE Tokenizer Trainer for La Pulga.
Trains a custom 8192-token BPE tokenizer on TinyStories,
so La Pulga can understand every word without truncation.

Usage:
    python src/data/train_tokenizer.py
"""
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB_SIZE: int = 8192
OUTPUT_PATH: str = "tokenizer.json"


def train_tokenizer() -> None:
    """
    Trains a Byte-Pair Encoding tokenizer from scratch on TinyStories.

    Why BPE from scratch instead of tiktoken?
    tiktoken's cl100k_base has ~100k tokens. Clamping it to 8192 corrupts 92%
    of the corpus. Training our own BPE ensures every merge pair is optimized
    for the specific distribution of children's stories.
    """
    print("Loading TinyStories dataset for tokenizer training...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    # We sample a meaningful subset to train the tokenizer quickly
    # 100k stories is more than enough to learn solid merge rules
    sample_size: int = min(100_000, len(dataset))
    texts: list[str] = [dataset[i]["text"] for i in range(sample_size)]
    print(f"Sampled {sample_size:,} stories for tokenizer training.")

    # Build a BPE tokenizer from scratch
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )

    print(f"Training BPE tokenizer with vocab_size={VOCAB_SIZE}...")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.save(OUTPUT_PATH)
    print(f"Tokenizer saved to {OUTPUT_PATH}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Quick sanity check
    test_text: str = "Once upon a time, there was a little dog named Max."
    encoded = tokenizer.encode(test_text)
    decoded: str = tokenizer.decode(encoded.ids)
    print(f"Test encode: '{test_text}' -> {encoded.ids[:20]}...")
    print(f"Test decode: '{decoded}'")


if __name__ == "__main__":
    train_tokenizer()
