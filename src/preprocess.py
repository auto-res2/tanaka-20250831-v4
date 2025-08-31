"""src/preprocess.py
----------------------------------
Very light-weight data preprocessing that downloads the required HuggingFace
`datasets` and stores a tokenised version to disk so that subsequent stages do
not have to repeat the (slow) tokenisation over and over again.

Given the constrained runtime of the reproduction we only process a couple of
hundred samples.
"""
from __future__ import annotations

import os
from typing import List, Dict

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def prepare_dataset(
    name: str = "wikitext2",
    split: str = "train",
    n_samples: int = 256,
    tokenizer_name: str = "sshleifer/tiny-gpt2",
    max_length: int = 512,
) -> str:
    """Download *name*, tokenise, and persist a `.pt` tensor file.
    Returns the path so callers can reuse the artefact."""
    save_path = os.path.join(DATA_DIR, f"{name}_{split}_{n_samples}.pt")
    if os.path.exists(save_path):
        return save_path

    ds = load_dataset(name, split=split).select(range(n_samples))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    encoded: List[Dict[str, torch.Tensor]] = []
    for sample in tqdm(ds, desc="tokenising"):
        ids = tokenizer(sample["text"], truncation=True, max_length=max_length, return_tensors="pt").input_ids.squeeze(0)
        encoded.append({"input_ids": ids})

    torch.save(encoded, save_path)
    return save_path
