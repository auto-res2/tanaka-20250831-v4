
"""
evaluate.py – utilities to evaluate the (tiny) LM trained with and
without HQRC.  We compute two metrics:
    1. Per-token negative log-likelihood (≈ perplexity)
    2. Memory usage & latency during autoregressive decoding
The results are returned as a python dictionary and are *also* plotted
and saved to .research/iteration10/images as vector-pdf files.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

from .preprocess import TextDataset

# ---------------------------------------------------------------------------
#   image output directory (updated to iteration10)
# ---------------------------------------------------------------------------

IMG_DIR = Path(".research/iteration10/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#   helper
# ---------------------------------------------------------------------------

def _gpu_mem_gib() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 3


@torch.no_grad()
def _synthetic_decode(model, seq_len: int = 64, device="cpu") -> Tuple[float, float]:
    # create a random prompt of length 16 ----------------------------------
    vocab_size = int(model.config.vocab_size)
    prompt = torch.randint(0, vocab_size, (1, 16), device=device)

    # prefill --------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    _ = model(prompt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_t = time.perf_counter() - t0

    # decode ---------------------------------------------------------------
    gen = prompt
    decode_times = []
    for _ in range(seq_len):
        t1 = time.perf_counter()
        out = model(gen[:, -1:])
        next_tok = out.logits[:, -1].argmax(-1, keepdim=True)
        gen = torch.cat([gen, next_tok], dim=-1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - t1)
    return prefill_t / 16, float(np.mean(decode_times))


# ---------------------------------------------------------------------------
#   public interface
# ---------------------------------------------------------------------------

def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: TextDataset,
    cache_mode: str,
) -> Dict:
    device = next(model.parameters()).device

    # ---------- Perplexity ----------------------------------------------
    nll_total, n_tokens = 0.0, 0
    for batch in dataset.dataloader(batch_size=4, pad_id=tokenizer.pad_token_id):
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch, labels=batch)
        nll_total += out.loss.item() * batch.numel()
        n_tokens += batch.numel()
    ppl = np.exp(nll_total / n_tokens)

    # ---------- Memory & latency ----------------------------------------
    prefill_ms, decode_ms = _synthetic_decode(model, seq_len=64, device=device)
    peak_mem = _gpu_mem_gib()

    results = {
        "cache_mode": cache_mode,
        "ppl": ppl,
        "prefill_ms_tok": prefill_ms * 1e3,
        "decode_ms_tok": decode_ms * 1e3,
        "peak_mem_gib": peak_mem,
    }
    return results


# ---------------------------------------------------------------------------
#   plotting helpers (bar chart) – saved as PDF ---------------------------
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(data=df, x="cache_mode", y="ppl", ax=ax)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Perplexity")
    fig.tight_layout()
    out_path = IMG_DIR / "perplexity.pdf"
    fig.savefig(out_path, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    sns.barplot(data=df, x="cache_mode", y="peak_mem_gib", ax=ax2)
    for p in ax2.patches:
        ax2.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2.set_ylabel("Peak memory (GiB)")
    fig2.tight_layout()
    fig2.savefig(IMG_DIR / "memory.pdf", bbox_inches="tight")

    print("Figures saved to", IMG_DIR)
