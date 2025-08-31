"""src/evaluate.py
Evaluation helper – computes perplexity on the validation split created
by `src.preprocess` and produces a tiny memory/latency profile similar to
what is sketched in the (much larger) research code.

All images are now saved under `.research/iteration4/images` as required.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .preprocess import MemMapDataset

# ---------------------------------------------------------------------------
# Image output directory (updated as per specification)
# ---------------------------------------------------------------------------
IMAGES_DIR = Path(".research/iteration4/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 0):
    import random, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def max_vram_mb():
    return torch.cuda.max_memory_allocated() / 1024 ** 2


# ---------------------------------------------------------------------------
# main eval routine
# ---------------------------------------------------------------------------

def evaluate(
    model_dir: str | os.PathLike = "models/gpt2-wikitext2",
    seq_len: int = 128,
    batch_size: int = 8,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Run evaluation for a set of *seeds* and return a DataFrame."""
    if seeds is None:
        seeds = [0, 1, 2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
    model.to(device).eval()

    val_ds = MemMapDataset("data/val.bin", seq_len)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    records = []

    for seed in seeds:
        set_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        pbar = tqdm(loader, desc=f"eval s={seed}", leave=False, ncols=80)
        total_ppl, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in pbar:
                inp = batch.to(device)
                # Use built-in loss computation to avoid float16 overflow issues
                out = model(inp, labels=inp, use_cache=True)
                loss = out.loss.float()
                ppl = float(torch.exp(loss))
                total_ppl += ppl
                n_batches += 1
        avg_ppl = total_ppl / n_batches if n_batches else float("nan")
        mem = max_vram_mb()
        records.append({"seed": seed, "perplexity": avg_ppl, "vram_mb": mem})
        print(f"seed {seed}: ppl={avg_ppl:.2f}, peak VRAM={mem:.0f} MB")

    df = pd.DataFrame.from_records(records)

    # simple visualisation – violin plot of ppl
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(4, 3))
    sns.violinplot(data=df, y="perplexity", inner="points", color="skyblue")
    plt.title("Validation perplexity distribution (seeds)")
    plt.ylabel("perplexity")
    plt.tight_layout()
    out_path = IMAGES_DIR / "perplexity_violin.pdf"
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()

    # Robust printing – avoid ValueError when paths are on different mounts
    try:
        relative = out_path.relative_to(Path.cwd())
    except ValueError:
        relative = out_path
    print(f"✓ Figure saved to {relative}")

    return df


if __name__ == "__main__":
    evaluate()
