"""Metrics for evaluating steered generation quality."""
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


def hedge_score(text):
    """Weighted count of hedging markers."""
    t = text.lower()
    return float(sum(t.count(h) for h in HEDGING_TERMS))


def lexical_diversity(text):
    """Type-token ratio: unique words / total words."""
    words = re.findall(r"[A-Za-z]+", text.lower())
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)


def token_validity(text, english_vocab=None):
    """
    Fraction of alphabetic tokens that are plausible English words.
    Uses a heuristic: real words have only common letter patterns.
    If english_vocab is provided (a set), uses exact lookup.
    """
    words = re.findall(r"[A-Za-z]+", text.lower())
    if len(words) == 0:
        return 0.0
    if english_vocab is not None:
        valid = sum(1 for w in words if w in english_vocab or len(w) <= 2)
        return valid / len(words)
    # Heuristic fallback: a "word" is suspect if it contains
    # repeated consonant triples (like "neurneur") or is very long
    def is_plausible(w):
        if len(w) > 20:
            return False
        # Check for repeated trigrams (e.g., "neurneur")
        for i in range(len(w) - 5):
            if w[i:i+3] == w[i+3:i+6]:
                return False
        return True
    valid = sum(1 for w in words if is_plausible(w))
    return valid / len(words)


def token_length(text):
    """Number of whitespace-separated tokens."""
    return len(text.split())


def perplexity(text, tok, model, device="mps"):
    """Compute perplexity of text under vanilla (unsteered) BioGPT."""
    if len(text.strip()) == 0:
        return float("inf")
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    if inputs["input_ids"].shape[1] < 2:
        return float("inf")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return float(math.exp(outputs.loss.item()))


def compute_all_metrics(text, tok, model, device="mps", english_vocab=None):
    return {
        "hedge_score": hedge_score(text),
        "lexical_diversity": lexical_diversity(text),
        "token_validity": token_validity(text, english_vocab),
        "length": token_length(text),
        "perplexity": perplexity(text, tok, model, device),
    }


def load_english_vocab():
    """Load a simple English vocab from nltk or fall back to heuristic."""
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


if __name__ == "__main__":
    # Quick sanity check
    samples = [
        "The chest X-ray shows a large pleural effusion.",
        "a neurneurneur some or a misstate at certain some",
    ]
    for s in samples:
        print(f"Text: {s[:60]}")
        print(f"  hedge_score:       {hedge_score(s):.2f}")
        print(f"  lexical_diversity: {lexical_diversity(s):.2f}")
        print(f"  token_validity:    {token_validity(s):.2f}")
        print(f"  length:            {token_length(s)}")
        print()