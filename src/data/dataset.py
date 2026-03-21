"""
Dataset Interface (Clean Architecture).
Responsible for fetching and caching public datasets.
"""
from datasets import load_dataset
from typing import Generator

def fetch_tinystories_stream(split: str = "train") -> Generator[str, None, None]:
    """
    Downloads the TinyStories dataset to local cache (~/.cache/huggingface) on the first run.
    Subsequent runs will load instantly from disk (memory-mapped Arrow files) without using the internet.
    We iterate over the memory-mapped dataset to avoid loading the entire corpus into RAM at once.
    """
    print("Checking local cache for TinyStories (will download if missing)...")
    dataset = load_dataset("roneneldan/TinyStories", split=split)
    
    for item in dataset:
        yield item["text"]
