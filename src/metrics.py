"""Metrics for evaluating steered generation quality.

Updated to handle both decoder (causal LM) and encoder (masked LM) models:
  - perplexity()           : causal LM loss — decoders only
  - pseudo_perplexity()    : masked LM pseudo-perplexity — encoders only
  - compute_all_metrics()  : dispatches based on model_type kwarg
"""

import re
import math
import torch
import numpy as np
from collections import Counter


HEDGING_TERMS = [
    "may", "might", "could", "possibly", "possible", "probably", "probable",
    "suggest", "suggests", "suggestive", "likely", "unclear", "uncertain",
    "appears", "appear", "seems", "seem", "would", "should",
    "consistent with", "compatible with", "cannot", "unknown", "perhaps",
    "indicate", "indicates", "indication",
]


def hedge_score(text: str) -> float:
    """Weighted count of hedging markers (unchanged from original paper)."""
    t = text.lower()
    return float(sum(t.count(h) for h in HEDGING_TERMS))


def lexical_diversity(text: str) -> float:
    """Type-token ratio: unique words / total words."""
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def token_validity(text: str, english_vocab=None) -> float:
    """
    Fraction of alphabetic tokens that are plausible English words.
    Uses exact lookup when *english_vocab* is provided; heuristic otherwise.
    """
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return 0.0
    if english_vocab is not None:
        valid = sum(1 for w in words if w in english_vocab or len(w) <= 2)
        return valid / len(words)

    def is_plausible(w):
        if len(w) > 20:
            return False
        for i in range(len(w) - 5):
            if w[i:i+3] == w[i+3:i+6]:
                return False
        return True

    return sum(1 for w in words if is_plausible(w)) / len(words)


def token_length(text: str) -> int:
    """Number of whitespace-separated tokens."""
    return len(text.split())


# --------------------------------------------------------------------------- #
# Perplexity — decoder (causal LM)                                            #
# --------------------------------------------------------------------------- #

def perplexity(text: str, tok, model, device: str = None) -> float:
    """
    Compute perplexity under a causal language model.
    Returns inf for empty or very short text.
    """
    from src.model_loader import get_device
    if device is None:
        device = get_device()

    if not text.strip():
        return float("inf")
    inputs = tok(
        text, return_tensors="pt", truncation=True, max_length=256
    ).to(device)
    if inputs["input_ids"].shape[1] < 2:
        return float("inf")
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return float(math.exp(out.loss.item()))


# --------------------------------------------------------------------------- #
# Pseudo-perplexity — encoder (masked LM)                                     #
# --------------------------------------------------------------------------- #

def pseudo_perplexity(text: str, tok, model, device: str = None) -> float:
    """
    Compute masked-LM pseudo-perplexity (Wang & Cho, 2019).

    For each token i, mask it, run a forward pass, collect log p(token_i | rest).
    PPL = exp( -1/N * sum log p(token_i) ).

    This is slower than causal PPL (N forward passes) but is the correct
    metric for encoder models.  Returns inf for very short sequences.
    """
    from src.model_loader import get_device
    if device is None:
        device = get_device()

    if not text.strip():
        return float("inf")

    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = enc["input_ids"][0]   # (seq,)
    N = input_ids.shape[0]
    if N < 2:
        return float("inf")

    log_probs = []
    for i in range(N):
        masked = input_ids.clone()
        masked[i] = tok.mask_token_id
        with torch.no_grad():
            out = model(
                input_ids=masked.unsqueeze(0).to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
        logits = out.logits[0, i]           # (vocab,)
        log_prob = torch.log_softmax(logits, dim=-1)[input_ids[i]].item()
        log_probs.append(log_prob)

    return float(math.exp(-np.mean(log_probs)))


# --------------------------------------------------------------------------- #
# Unified metric computation                                                   #
# --------------------------------------------------------------------------- #

def compute_all_metrics(text: str, tok, model,
                        model_type: str = "decoder",
                        device: str = None,
                        english_vocab=None) -> dict:
    """
    Compute all metrics for *text*.

    model_type : "decoder" → causal perplexity
                 "encoder" → masked pseudo-perplexity
    """
    from src.model_loader import get_device
    if device is None:
        device = get_device()

    metrics = {
        "hedge_score": hedge_score(text),
        "lexical_diversity": lexical_diversity(text),
        "token_validity": token_validity(text, english_vocab),
        "length": token_length(text),
    }

    if model_type == "decoder":
        metrics["perplexity"] = perplexity(text, tok, model, device)
        metrics["pseudo_perplexity"] = None
    else:
        metrics["perplexity"] = None
        metrics["pseudo_perplexity"] = pseudo_perplexity(text, tok, model, device)

    return metrics


# --------------------------------------------------------------------------- #
# NLTK vocab loader (unchanged)                                                #
# --------------------------------------------------------------------------- #

def load_english_vocab():
    """Load a simple English vocab from nltk or fall back to None."""
    try:
        import nltk
        try:
            from nltk.corpus import words
            return set(w.lower() for w in words.words())
        except LookupError:
            nltk.download("words", quiet=True)
            from nltk.corpus import words
            return set(w.lower() for w in words.words())
    except ImportError:
        return None
