"""src/evaluate.py
----------------------------------
Utilities to evaluate a (possibly HQRC-augmented) language model.
We mainly measure perplexity on Wikitext-2, latency, and GPU memory.
All heavy lifting (dataset download, cache handling) is delegated to helpers
in `src.utils` so that this module stays readable.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .utils.data import get_dataset, iter_batches
from .utils.cache_modes import CACHE_FUNCS  # attach HQRC / int8 / EL-attention
from .utils.gpu import gpu_mem_gib


def perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    split: str = "validation",
    max_samples: int = 64,
    context_len: int = 512,
) -> float:
    """Compute (approximate) perplexity on a subset of Wikitext-2."""
    ds = get_dataset("wikitext2", split).select(range(max_samples))
    running_nll, running_ntokens = 0.0, 0
    for sample in ds:
        ids = (
            tokenizer(sample["text"], return_tensors="pt").input_ids[:, :context_len]
            .to(model.device)
        )
        out = model(ids, labels=ids, use_cache=True)
        running_nll += out.loss.item() * ids.numel()
        running_ntokens += ids.numel()
    return float(np.exp(running_nll / running_ntokens))


@torch.inference_mode()
def latency_and_memory(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    decode_tokens: int = 128,
) -> Dict[str, float]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model(ids, use_cache=True)
    torch.cuda.synchronize()
    prefill = (time.perf_counter() - t0) * 1e3  # ms

    # autoregressive decode
    next_ids = ids[:, -1:]
    times = []
    for _ in range(decode_tokens):
        t1 = time.perf_counter()
        out = model(next_ids, use_cache=True)
        next_ids = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t1) * 1e3)
    return {
        "prefill_ms": prefill,
        "decode_ms_per_tok": float(np.mean(times)),
        "peak_mem_gib": gpu_mem_gib(),
    }


# -----------------------------------------------------------------------------
# public façade
# -----------------------------------------------------------------------------

def run_single(
    model_ckpt: str,
    cache_mode: str = "fp16",
    context_len: int = 512,
):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = (
        AutoModelForCausalLM.from_pretrained(model_ckpt).eval().cuda()
    )
    model = CACHE_FUNCS[cache_mode](model)

    ppl = perplexity(model, tokenizer, context_len=context_len)
    lm_stats = latency_and_memory(model, tokenizer)

    result = {"ppl": ppl, **lm_stats, "cache_mode": cache_mode}
    print(json.dumps(result, indent=2))
    return result
