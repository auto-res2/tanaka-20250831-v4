"""src/train.py
----------------------------------
Minimal training / calibration utilities for the HQRC demonstration.
The intention of this module is **not** to carry out a full-blown fine-tune
of a language model – that would take many GPU-hours – but to provide a
hook such that the pipeline

   python -m src.main --stage train

executes without errors and produces an artefact that the subsequent
"evaluate" stage can load.

For the purposes of the open-source reproduction we therefore:
1. load a small, publicly available causal-LM (defaults to GPT-2 tiny).
2. run *at most* a handful of optimiser steps so runtime stays < 2 min.
3. optionally attach the (dummy) HQRC cache and train only these very few
   additional parameters (≈0.2 %).
4. save the resulting model weights into the `models/` directory so that
   evaluate.py can reload them.

NOTE:  In a real research setting you would replace the tiny training loop
by a proper Trainer/Accelerate routine and point `MODEL_NAME` to a checkpoint
that has already been quantised with SqueezeLLM etc.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW  # Fixed: AdamW comes from torch.optim
from tqdm import tqdm

# Relative import – keeps the public API of HQRC helpers inside this package
from .utils.cache_modes import attach_hqrc  # noqa: F401 (used via string)

# -------------------------------------------------------------------------
# Configuration helpers
# -------------------------------------------------------------------------

DEFAULT_MODEL = "sshleifer/tiny-gpt2"  # 12 M parameters, loads fast
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


class ModelTrainer:
    """Very small wrapper around a hand written training loop."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        lr: float = 2e-5,
        max_steps: int = 25,
        use_hqrc: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model_name = model_name
        self.lr = lr
        self.max_steps = max_steps
        self.use_hqrc = use_hqrc
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name).to(self.device).train()
        )
        if use_hqrc:
            from .utils.cache_modes import attach_hqrc  # lazy import to avoid cycles

            attach_hqrc(self.model)

        # Only train HQRC params (or the whole model if HQRC disabled)
        params = (
            [p for n, p in self.model.named_parameters() if p.requires_grad]
            if use_hqrc
            else self.model.parameters()
        )
        self.optim = AdamW(params, lr=lr)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def train(self, save_path: Optional[str] = None) -> str:
        text = (
            "Scientific progress on the GPU has been extraordinary. " * 128
        )  # ≈1 k tokens
        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        labels = inputs.clone()

        start = time.perf_counter()
        for step in tqdm(range(1, self.max_steps + 1), desc="training"):
            out = self.model(inputs, labels=labels, use_cache=True)
            out.loss.backward()
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)

            if step % 5 == 0:
                print(f"step {step:>3d} | loss {out.loss.item():.4f}")
        runtime = time.perf_counter() - start
        print(f"Finished {self.max_steps} steps in {runtime:.1f} s")

        if save_path is None:
            ckpt_name = (
                f"{self.model_name.split('/')[-1]}_hqrc.pt"
                if self.use_hqrc
                else f"{self.model_name.split('/')[-1]}_ft.pt"
            )
            save_path = os.path.join(MODELS_DIR, ckpt_name)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to: {save_path}")
        return save_path
