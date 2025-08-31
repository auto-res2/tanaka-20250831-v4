"""
main.py – command-line front-end.  Running

    python -m src.main

will execute the full pipeline:
    1. preprocess → tiny Wikitext-2 subset
    2. train two models: baseline (fp16 cache) and HQRC
    3. evaluate both versions
    4. save figures & print a small table to stdout

All heavy parameters can be set via a YAML config passed with --config
but sensible defaults are provided so the command above runs in <2 min
on CPU and much faster on GPU / T4 (fits the 16 GB budget easily).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

# local imports (relative!)
from .preprocess import preprocess
from .train import train
from .evaluate import evaluate, make_plots

try:
    import yaml  # optional – only needed if user passes a config file
except ImportError:
    yaml = None


_DEFAULT_CONFIG: Dict = {
    "model_name": "sshleifer/tiny-gpt2",  # fast to download/fine-tune
    "max_samples": 512,

    # training hyper-params (tiny for demo)
    "lr": 5e-4,
    "train_steps": 20,
    "batch_size": 4,
}


# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML not installed – cannot parse config file.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------

def run_pipeline(config: Dict):
    print("\n==== HQRC – tiny demonstration pipeline ====\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ------------------------------------------------------------------
    print("[1] Pre-processing …")
    ds = preprocess(config)
    print("   → {} sequences".format(len(ds)))

    # ------------------------------------------------------------------
    results = []
    for mode in ("fp16", "hqrc"):
        cfg = dict(config)
        cfg["cache_mode"] = mode
        print(f"[2] Training (cache_mode={mode}) …")
        model, tokenizer = train(cfg, ds)

        print("[3] Evaluation …")
        res = evaluate(model, tokenizer, ds, cache_mode=mode)
        results.append(res)

    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    out_csv = Path("results")
    out_csv.mkdir(exist_ok=True)
    csv_path = out_csv / "summary.csv"
    df.to_csv(csv_path, index=False)

    print("\n==== SUMMARY ====\n")
    print(df.to_string(index=False, float_format="{:.3f}".format))
    print("Results saved to", csv_path)

    make_plots(df)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HQRC demo pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Yaml file with parameters. If omitted, default tiny config is used.",
    )
    args = parser.parse_args()

    cfg = dict(_DEFAULT_CONFIG)
    if args.config is not None:
        cfg.update(_load_yaml(Path(args.config)))

    run_pipeline(cfg)
