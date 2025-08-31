```python
"""src/preprocess.py
Tokenises a small slice of WikiText-2 and stores it as two uint16
mem-mapped binary files under `data/`:
    • train.bin
    • val.bin
The default slice is **very** small so that the whole end-to-end pipeline
can execute inside the evaluation time budget.  Adjust `max_train_tokens`
and `max_val_tokens` via function arguments or the CLI if you wish to run
a longer training job locally.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Low-level dataset class – reused by train/evaluate
# ---------------------------------------------------------------------------

class MemMapDataset:
    """Memory-mapped token sequence split into *fixed-length* blocks."""

    def __init__(self, bin_path: str | os.PathLike, seq_len: int):
        self.fp = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return len(self.fp) // self.seq_len

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        arr = self.fp[start : start + self.seq_len]
        return torch.as_tensor(arr, dtype=torch.long)


# We use torch only for type annotation above – import here to avoid a hard
# requirement for preprocess (which may run on CPU-only machines).
import torch  # noqa: E402  isort: skip


# ---------------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------------

def _tokenise_texts(texts: List[str], tok) -> List[int]:
    """Flatten tokenised texts into a single list of IDs."""
    out = []
    for t in tqdm(texts, desc="tokenising", leave=False):
        out.extend(tok(t).input_ids + [tok.eos_token_id])
    return out


def _save_memmap(arr: List[int], path: Path):
    mm = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(arr),))
    mm[:] = np.array(arr, dtype=np.uint16)
    mm.flush()


def run(seq_len: int = 128,
        max_train_tokens: int = 50_000,
        max_val_tokens: int = 10_000):
    """Create *tiny* WikiText-2 mem-mapped dataset for demo purposes."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"

    if train_bin.exists() and val_bin.exists():
        # nothing to do – reuse existing files
        return

    print("Downloading WikiText-2 (via 🤗 datasets)…")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token  # ensure pad_token exists

    # ------------------------------------------------------------------
    # Build train / val arrays – we slice only a tiny subset
    # ------------------------------------------------------------------
    train_ids = _tokenise_texts(ds["train"]["text"][:2000], tok)[:max_train_tokens]
    val_ids = _tokenise_texts(ds["validation"]["text"][:400], tok)[:max_val_tokens]

    print(f"train tokens: {len(train_ids):,} – val tokens: {len(val_ids):,}")

    _save_memmap(train_ids, train_bin)
    _save_memmap(val_ids, val_bin)
    print("✓ Tokenised dataset stored in 'data/'")


# ---------------------------------------------------------------------------
# CLI helper so that one can run:  python -m src.preprocess
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
``