"""src/main.py
Entry point for the reproduction of the paper's experiments.
Run from project root via

    python -m src.main [--stage preprocess|train|eval|exp] [other args]

The code purposefully keeps the default run *very* light weight so that it
finishes on a single T4 within a couple of minutes while still exercising all
modules (preprocess → train → evaluate → result plots).

Stages
------
1. preprocess : download + tokenise tiny subset of Wikitext-2
2. train      : run a brief calibration/fine-tune loop (see `src/train.py`)
3. eval       : measure perplexity, latency, memory (see `src/evaluate.py`)
4. exp        : run the full, self-contained experiment suite that mirrors the
                pseudo-code from the manuscript (scaled down version!)
"""

from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from . import preprocess as pp
from . import train as tr
from . import evaluate as ev
from .utils.plotting import save_pdf
from .utils.cache_modes import CACHE_MODES  # noqa: F401 (side-effect printing)

# directory for images dictated by the instructions
IMG_DIR = ".research/iteration14/images"  # Updated as per instructions
os.makedirs(IMG_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Helper functions for the small-scale experiment
# -----------------------------------------------------------------------------

def _tiny_experiment():
    """One-shot run that executes preprocess, train, evaluate and plots."""
    ds_path = pp.prepare_dataset()

    trainer = tr.ModelTrainer(use_hqrc=True, max_steps=20)
    ckpt = trainer.train()

    results: List[dict] = []
    for mode in ["fp16", "int8", "el", "hqrc"]:
        res = ev.run_single(ckpt, cache_mode=mode)
        results.append(res)

    df = pd.DataFrame(results)
    # ----- plotting ------------------------------------------------------
    sns.set(style="whitegrid")

    # Memory
    fig = sns.barplot(data=df, x="cache_mode", y="peak_mem_gib").get_figure()
    plt.ylabel("Peak GPU memory (GiB)")
    save_pdf(fig, os.path.join(IMG_DIR, "memory_small.pdf"))

    # Latency
    plt.clf()
    fig = sns.barplot(data=df, x="cache_mode", y="decode_ms_per_tok").get_figure()
    plt.ylabel("Decode latency (ms/token)")
    save_pdf(fig, os.path.join(IMG_DIR, "latency_small.pdf"))


# -----------------------------------------------------------------------------
# main CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["preprocess", "train", "eval", "exp"],
        default="exp",
        help="Which pipeline stage to run.",
    )
    args = parser.parse_args()

    if args.stage == "preprocess":
        pp.prepare_dataset()
    elif args.stage == "train":
        trainer = tr.ModelTrainer(use_hqrc=True)
        trainer.train()
    elif args.stage == "eval":
        # Requires that a checkpoint exists already
        ckpt = os.path.join("models", "tiny-gpt2_hqrc")
        if not os.path.isdir(ckpt):
            raise FileNotFoundError(
                "Checkpoint not found – please run with --stage train first."
            )
        ev.run_single(ckpt, cache_mode="hqrc")
    elif args.stage == "exp":
        _tiny_experiment()


if __name__ == "__main__":
    main()
