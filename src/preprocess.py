"""
preprocess.py – data preparation utilities.
The module exposes
    1. TextDataset – a thin wrapper that tokenises on-the-fly and can
       provide a PyTorch DataLoader
    2. preprocess(config) – downloads the dataset and returns the
       TextDataset instance (also caches it on disk).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, token_ids: List[List[int]]):
        self.data = token_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

    # -------------------------------------------------------------------
    # custom DataLoader with padding to the longest sequence in the batch
    # -------------------------------------------------------------------
    @staticmethod
    def _make_collate_fn(pad_id: int):
        def _collate(batch):
            max_len = max(len(seq) for seq in batch)
            out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            for i, seq in enumerate(batch):
                out[i, : len(seq)] = seq
            return out

        return _collate

    def dataloader(
        self,
        batch_size: int = 4,
        shuffle: bool = False,
        pad_id: int = 0,
        drop_last: bool = False,
    ):
        collate_fn = self._make_collate_fn(pad_id)
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )

# ---------------------------------------------------------------------------

_CACHE_FILE = DATA_DIR / "wikitext2_tokens.pkl"


def preprocess(config: Dict) -> TextDataset:
    """Download *wikitext-2* (validation split, first N samples) and tokenize.
    The tokenised ids are cached to `data/wikitext2_tokens.pkl` so that
    subsequent runs are instant.
    """
    if _CACHE_FILE.exists():
        with _CACHE_FILE.open("rb") as f:
            token_ids = pickle.load(f)
        return TextDataset(token_ids)

    model_name = config.get("model_name", "sshleifer/tiny-gpt2")
    max_samples = config.get("max_samples", 512)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="validation[:{}]".format(max_samples)
    )
    token_ids = []
    for item in dataset:
        ids = tokenizer(item["text"], truncation=True, max_length=128)["input_ids"]
        if len(ids) > 0:
            token_ids.append(ids)

    # cache
    with _CACHE_FILE.open("wb") as f:
        pickle.dump(token_ids, f)
    return TextDataset(token_ids)
