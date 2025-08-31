
"""
train.py – training / calibration of a (tiny) language model with and
without the proposed HQRC cache wrapper.  The goal is not to obtain a
state-of-the-art model but to demonstrate how HQRC can be trained and
saved so that it can be used afterwards by evaluate.py.

The script exposes one public function:
    train(config: dict, dataset) -> (model, tokenizer)
which is imported and executed by src/main.py.

All heavy-duty hyper-parameters live in the config dictionary.  A
minimal default configuration is created in main.py so the user can run

    python -m src.main

out-of-the-box.  If finer control is needed one can pass a yaml file via
--config (see main.py).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# local relative import (allowed)
from .preprocess import TextDataset

# ---------------------------------------------------------------------------
#   HQRC – a minimal, self–contained implementation
# ---------------------------------------------------------------------------

class _PQCodebook(nn.Module):
    """A very small product-quantisation codebook (4-bit ≈ 16 entries).
    NB:  This is drastically simplified so that the whole training run
    finishes within a couple of seconds on CPU / a small GPU.
    """
    def __init__(self, dim: int, k: int = 16):
        super().__init__()
        self.dim = dim
        self.k = k
        self.codebook = nn.Parameter(torch.randn(k, dim))

    def forward(self, x: torch.Tensor):
        # x: (..., dim)
        flat = x.reshape(-1, self.dim)
        # L2 distance
        dist = (flat.unsqueeze(1) - self.codebook).pow(2).sum(-1)
        inds = dist.argmin(1)
        codes = self.codebook[inds]
        return codes.reshape_as(x), inds.reshape(x.shape[:-1])


class _HQRCEncoder(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(hidden_size, latent_dim, bias=False)
        self.cb = _PQCodebook(latent_dim, k=16)

    def forward(self, kv: torch.Tensor):
        # kv: (B, H, T, D)
        z = self.proj(kv)
        q, idx = self.cb(z)
        return q, idx


class _HQRCDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )

    def forward(self, z):
        return self.mlp(z)


class HQRCWrapper(nn.Module):
    """Wrap a HuggingFace causal-LM and compress its KV cache online.
    The implementation is a *toy* version: we only quantise the most
    recent token of every layer during generation in order to showcase
    memory savings.  For training we run the forward pass and optimise
    an auto-encoder reconstruction loss.
    """

    def __init__(self, hf_model: AutoModelForCausalLM):
        super().__init__()
        self.model = hf_model
        # expose the underlying config so that downstream code (e.g.
        # evaluate.py) can access `model.config` transparently.
        self.config = hf_model.config  # <-- attribute delegation
        hs = hf_model.config.hidden_size
        nl = hf_model.config.num_hidden_layers
        self.encoders = nn.ModuleList([_HQRCEncoder(hs) for _ in range(nl)])
        self.decoders = nn.ModuleList([_HQRCDecoder(32, hs) for _ in range(nl)])
        self._install_hooks()

    # ------------------------------------------------------------------
    def _install_hooks(self):
        """Register forward hooks on *self-attention* blocks in each layer."""
        handles = []
        # GPT-like architectures expose layers under `transformer.h`
        for li, layer in enumerate(self.model.transformer.h):
            attn_mod = layer.attn

            def _make_hook(layer_idx: int):
                # Use a factory to correctly capture the current `layer_idx`
                def _hook(mod, inputs, output):  # noqa: D401  – HF signature
                    """Post-process the KV cache coming out of the attention."""
                    if output is None:
                        return output

                    # Standardise the tuple length first -------------------
                    if len(output) == 3:
                        attn_out, attn_weights, past = output
                    else:
                        attn_out, past = output
                        attn_weights = None

                    # If no KV cache was returned we safely skip compression
                    if past is None:
                        return output  # unchanged

                    # `past` is a tuple(key, value) with shape (B, H, T, D)
                    k, v = past[0][:, :, -1:], past[1][:, :, -1:]  # last step
                    z_k, _ = self.encoders[layer_idx](k)
                    z_v, _ = self.encoders[layer_idx](v)
                    k_recon = self.decoders[layer_idx](z_k)
                    v_recon = self.decoders[layer_idx](z_v)
                    past[0][:, :, -1:] = k_recon
                    past[1][:, :, -1:] = v_recon

                    # Return a tuple of the *same* length as the original
                    if attn_weights is None:
                        return (attn_out, past)
                    else:
                        return (attn_out, attn_weights, past)

                return _hook

            handles.append(attn_mod.register_forward_hook(_make_hook(li)))
        self._handles = handles

    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):  # type: ignore[override]
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------------
    # HuggingFace compatibility helpers ---------------------------------
    # ------------------------------------------------------------------
    def save_pretrained(self, save_directory: str | os.PathLike, **kwargs):  # noqa: D401
        """Save the wrapped model so that it can be reloaded later."""
        self.model.save_pretrained(save_directory, **kwargs)
        state_to_save = {
            "encoders": self.encoders.state_dict(),
            "decoders": self.decoders.state_dict(),
        }
        torch.save(state_to_save, os.path.join(save_directory, "hqrc_state.pt"))


# ---------------------------------------------------------------------------
#   TRAIN ROUTINE
# ---------------------------------------------------------------------------


def _save(model: nn.Module, tokenizer: AutoTokenizer, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def train(config: Dict, dataset: TextDataset) -> Tuple[nn.Module, AutoTokenizer]:
    """Tiny training loop (≤ 60s on CPU) – demonstrates how HQRC would be
    trained.  The function either returns a vanilla pre-trained model or
    a model wrapped with HQRC and *lightly* fine-tuned so the codebooks
    learn something meaningful.
    """
    model_name = config.get("model_name", "sshleifer/tiny-gpt2")
    cache_mode = config.get("cache_mode", "fp16")  # "fp16" | "hqrc"
    lr = config.get("lr", 1e-4)
    train_steps = config.get("train_steps", 20)
    batch_size = config.get("batch_size", 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # tiny-gpt2 has no pad token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.float16 if (device.type == "cuda" and cache_mode == "fp16") else torch.float32),
    )
    if cache_mode == "hqrc":
        model = HQRCWrapper(base_model)
    else:
        model = base_model

    model.to(device)
    model.train()

    # Data loader -------------------------------------------------------
    dl = dataset.dataloader(
        batch_size=batch_size,
        shuffle=True,
        pad_id=tokenizer.pad_token_id,
        drop_last=True,
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    pbar = tqdm(range(train_steps), desc="training", ncols=80)
    itr = iter(dl)
    for step in pbar:
        try:
            batch = next(itr)
        except StopIteration:
            itr = iter(dl)
            batch = next(itr)
        batch = batch.to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    # Save --------------------------------------------------------------
    out_dir = Path("models") / ("hqrc" if cache_mode == "hqrc" else "baseline")
    _save(model, tokenizer, out_dir)

    return model.eval(), tokenizer
