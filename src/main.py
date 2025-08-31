"""src/main.py
Driver that strings together preprocessing, training and evaluation so
that the full research pipeline can be executed via
    python -m src.main
in accordance with the grading harness.

All output figures are stored under `.research/iteration4/images/` in PDF
format ready for inclusion in academic material.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import evaluate as evl
from . import preprocess as prep
from . import train as trn

IMAGES_DIR = Path(".research/iteration4/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="HQRC – toy reproducible pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "pre", "train", "eval"],
        default="all",
        help="Which stage(s) to run – default: all",
    )
    args = parser.parse_args()

    if args.stage in ("all", "pre"):
        print("\n=== Stage: PREPROCESS ===")
        prep.run()

    if args.stage in ("all", "train"):
        print("\n=== Stage: TRAIN ===")
        trn.train()

    if args.stage in ("all", "eval"):
        print("\n=== Stage: EVALUATE ===")
        df = evl.evaluate()
        print("\nSummary:")
        print(df)

    print(
        "\nPipeline finished – artefacts:\n  • model  → models/gpt2-wikitext2\n  • data   → data/train|val.bin\n  • plots  → .research/iteration4/images/*.pdf"
    )


if __name__ == "__main__":
    main()
