"""
Model-agnostic probing pipeline.

Replaces the original probe.py and probe_vector.py with a single,
reusable module that works for any model in the registry.

All random seeds are preserved from the original paper (seed=42 for
data sampling and probe training, seeds 0-19 for generation).
"""

import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.model_loader import get_device


# --------------------------------------------------------------------------- #
# Hidden-state extraction                                                      #
# --------------------------------------------------------------------------- #

def get_mean_hidden_state(text: str, tok, model, layer: int,
                          device: str = None) -> np.ndarray:
    """
    Mean-pool the hidden states of *layer* over all non-padding tokens.

    Layer indexing follows HuggingFace convention:
      index 0 → embedding output
      index i → output of the i-th transformer block

    Returns a 1-D numpy array of shape (hidden_dim,).
    """
    if device is None:
        device = get_device()

    inputs = tok(
        text, return_tensors="pt",
        truncation=True, max_length=256,
        padding=False,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple of (n_layers+1) tensors, each (1, seq_len, H)
    hs = outputs.hidden_states[layer]          # (1, seq_len, H)
    mask = inputs["attention_mask"].unsqueeze(-1).float()  # (1, seq_len, 1)
    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1)     # (1, H)
    return pooled[0].cpu().float().numpy()


def extract_features(sentences: list, tok, model, layer: int,
                     device: str = None) -> np.ndarray:
    """
    Extract mean-pooled hidden states for a list of sentences.

    Returns shape (N, hidden_dim).
    """
    return np.stack([
        get_mean_hidden_state(s, tok, model, layer, device)
        for s in sentences
    ])


# --------------------------------------------------------------------------- #
# Cross-validated probing                                                      #
# --------------------------------------------------------------------------- #

def probe_at_layer(X_uncertain: np.ndarray, X_certain: np.ndarray,
                   n_splits: int = 5, seed: int = 42) -> dict:
    """
    Train a logistic regression probe and return CV accuracy statistics.

    Parameters match the original paper (C=0.1, 5-fold stratified CV, seed=42).
    """
    X = np.vstack([X_uncertain, X_certain])
    y = np.array([1] * len(X_uncertain) + [0] * len(X_certain))

    clf = LogisticRegression(max_iter=2000, C=0.1, random_state=seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=1)

    mean_acc = float(scores.mean())
    std_acc = float(scores.std(ddof=1))
    ci_margin = 1.96 * std_acc / np.sqrt(len(scores))

    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "ci_low": mean_acc - ci_margin,
        "ci_high": mean_acc + ci_margin,
        "fold_scores": scores.tolist(),
    }


def run_probing_sweep(uncertain: list, certain: list,
                      tok, model, layers: list,
                      device: str = None,
                      n_splits: int = 5,
                      seed: int = 42,
                      verbose: bool = True) -> dict:
    """
    Run the full probing sweep across all specified layers.

    Returns a dict keyed by layer index with probe accuracy stats.
    """
    if verbose:
        print(f"{'Layer':>6} {'CV acc':>10} {'std':>8}  {'95% CI':>18}")
        print("-" * 50)

    results = {}
    for layer in layers:
        t0 = time.time()
        if verbose:
            print(f"Extracting layer {layer}...", end=" ", flush=True)

        X_u = extract_features(uncertain, tok, model, layer, device)
        X_c = extract_features(certain, tok, model, layer, device)
        stats = probe_at_layer(X_u, X_c, n_splits=n_splits, seed=seed)
        elapsed = time.time() - t0

        results[layer] = stats
        if verbose:
            print(
                f"done [{elapsed:.1f}s]   "
                f"layer {layer:>3}  "
                f"{stats['mean_acc']:>9.2%}  "
                f"{stats['std_acc']:>7.2%}  "
                f"[{stats['ci_low']:.2%}, {stats['ci_high']:.2%}]"
            )

    return results


def best_layer(probe_results: dict) -> int:
    """Return the layer index with the highest mean CV accuracy."""
    return max(probe_results, key=lambda l: probe_results[l]["mean_acc"])


def best_layer_near_anchor(probe_results: dict, anchor: int,
                           window: int = 2) -> int:
    """
    Select the steering layer within [anchor-window, anchor+window].

    Rationale: the 2/3-depth rule provides a theoretically motivated prior
    on where representations are semantically mature enough to steer but
    far enough from the output that the perturbation can propagate.  Within
    that neighbourhood the empirical probe accuracy breaks ties, preferring
    the layer where the uncertainty signal is strongest.  Ties in mean
    accuracy are resolved by minimum standard deviation (preferring the
    more reliable probe across CV folds).

    Layer 0 (embedding output) is excluded because it has no hookable
    transformer-block module.

    Parameters
    ----------
    probe_results : dict returned by run_probing_sweep (integer keys).
    anchor        : 2/3-depth layer index, e.g. from two_thirds_layer().
    window        : Number of layers to search on each side of anchor.

    Returns
    -------
    int : selected layer index.
    """
    candidates = [
        l for l in range(anchor - window, anchor + window + 1)
        if l >= 1 and l in probe_results
    ]
    if not candidates:
        raise ValueError(
            f"No probe results in window [{anchor - window}, {anchor + window}]. "
            f"Available layers: {sorted(probe_results.keys())}"
        )
    return max(candidates,
               key=lambda l: (probe_results[l]["mean_acc"],
                              -probe_results[l]["std_acc"]))


# --------------------------------------------------------------------------- #
# Probe vector (direction) extraction                                          #
# --------------------------------------------------------------------------- #

def build_probe_vector(X_uncertain: np.ndarray, X_certain: np.ndarray,
                       seed: int = 42):
    """
    Fit logistic regression on ALL data and return the weight vector.

    This is the "uncertainty direction" in activation space.
    Returns (weight_vector, train_accuracy).
    """
    X = np.vstack([X_uncertain, X_certain])
    y = np.array([1] * len(X_uncertain) + [0] * len(X_certain))
    clf = LogisticRegression(max_iter=2000, C=0.1, random_state=seed)
    clf.fit(X, y)
    return clf.coef_[0], float(clf.score(X, y))


def build_probe_vectors_at_layers(uncertain: list, certain: list,
                                  tok, model, layers: list,
                                  device: str = None,
                                  seed: int = 42,
                                  verbose: bool = True) -> dict:
    """
    Build probe vectors at multiple layers.

    Returns dict: {layer: {"vector": [...], "train_acc": float}}.
    """
    if verbose:
        print(f"{'Layer':>6} {'train_acc':>10} {'||vec||':>10}")
        print("-" * 30)

    out = {}
    for layer in layers:
        X_u = extract_features(uncertain, tok, model, layer, device)
        X_c = extract_features(certain, tok, model, layer, device)
        vec, acc = build_probe_vector(X_u, X_c, seed=seed)
        out[layer] = {"vector": vec.tolist(), "train_acc": acc}
        if verbose:
            print(f"{layer:>6} {acc:>9.2%} {np.linalg.norm(vec):>10.4f}")

    return out
