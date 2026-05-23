"""
Confound orthogonalization — model-agnostic.

Consolidates orthogonalize.py and orthogonalize_v2.py into one module.

The goal is to remove length and lexical-hedge confounds from the probe
direction so that the steering vector captures *genuine* epistemic
uncertainty rather than sentence length or surface hedging cues.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.probing import extract_features


# --------------------------------------------------------------------------- #
# Surface feature counting                                                     #
# --------------------------------------------------------------------------- #

HEDGING_TERMS = [
    "may", "might", "could", "possibly", "possible", "probably", "probable",
    "suggest", "suggests", "suggestive", "likely", "unclear", "uncertain",
    "appears", "appear", "seems", "seem", "would", "should",
    "consistent with", "compatible with", "cannot", "unknown", "perhaps",
    "indicate", "indicates", "indication",
]


def count_hedges(text: str) -> int:
    t = text.lower()
    return sum(t.count(h) for h in HEDGING_TERMS)


def count_tokens(text: str, tok) -> int:
    return len(tok.encode(text, add_special_tokens=False))


# --------------------------------------------------------------------------- #
# Core linear-algebra helpers                                                  #
# --------------------------------------------------------------------------- #

def orthogonalize(v_target: np.ndarray, v_confound: np.ndarray) -> np.ndarray:
    """Remove the v_confound component from v_target (Gram-Schmidt step)."""
    v_target = np.asarray(v_target, dtype=np.float64)
    v_confound = np.asarray(v_confound, dtype=np.float64)
    unit = v_confound / np.linalg.norm(v_confound)
    return v_target - np.dot(v_target, unit) * unit


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------- #
# Full orthogonalization pipeline                                              #
# --------------------------------------------------------------------------- #

def build_orthogonal_directions(uncertain: list, certain: list,
                                tok, model, layer: int,
                                device: str = None,
                                seed: int = 42,
                                verbose: bool = True) -> dict:
    """
    Build three directions at *layer*:

      v_probe         — raw logistic-regression uncertainty direction
      v_ortho_length  — probe orthogonalized against token-length confound
      v_ortho_lh      — further orthogonalized against lexical-hedge confound

    Returns a dict with all vectors, cosine similarities, and norms.
    Also evaluates whether the final direction still classifies uncertainty
    above chance using a simple median-threshold classifier.
    """
    all_sentences = uncertain + certain
    y = np.array([1] * len(uncertain) + [0] * len(certain))

    lengths = np.array([count_tokens(s, tok) for s in all_sentences])
    hedge_counts = np.array([count_hedges(s) for s in all_sentences])

    X = extract_features(all_sentences, tok, model, layer, device)

    # Probe direction
    v_probe = LogisticRegression(
        max_iter=2000, C=0.1, random_state=seed
    ).fit(X, y).coef_[0]

    # Confound directions (linear regression weights)
    v_length = LinearRegression().fit(X, lengths).coef_
    v_hedge = LinearRegression().fit(X, hedge_counts).coef_

    # Orthogonalize step-by-step
    v_ortho_L = orthogonalize(v_probe, v_length)
    v_ortho_LH = orthogonalize(v_ortho_L, v_hedge)

    # Residual classification accuracy of the final direction
    unit_lh = v_ortho_LH / np.linalg.norm(v_ortho_LH)
    proj = X @ unit_lh
    threshold = np.median(proj)
    preds = (proj > threshold).astype(int)
    # Flip if needed (direction may point toward "certain")
    acc_lh = max(
        float((preds == y).mean()),
        float(((1 - preds) == y).mean()),
    )

    result = {
        "v_probe": v_probe.tolist(),
        "v_ortho_length": v_ortho_L.tolist(),
        "v_ortho_length_hedge": v_ortho_LH.tolist(),
        "v_length_confound": v_length.tolist(),
        "v_hedge_confound": v_hedge.tolist(),
        "cos_probe_length": cosine_similarity(v_probe, v_length),
        "cos_probe_hedge": cosine_similarity(v_probe, v_hedge),
        "norm_probe": float(np.linalg.norm(v_probe)),
        "norm_ortho_L": float(np.linalg.norm(v_ortho_L)),
        "norm_ortho_LH": float(np.linalg.norm(v_ortho_LH)),
        "ortho_LH_classification_acc": acc_lh,
    }

    if verbose:
        print(
            f"  cos(probe, length)={result['cos_probe_length']:.4f}  "
            f"cos(probe, hedge)={result['cos_probe_hedge']:.4f}  "
            f"ortho_LH acc={acc_lh:.2%}"
        )

    return result


def build_orthogonal_directions_at_layers(uncertain: list, certain: list,
                                          tok, model, layers: list,
                                          device: str = None,
                                          seed: int = 42,
                                          verbose: bool = True) -> dict:
    """
    Run build_orthogonal_directions for each layer in *layers*.

    Returns dict: {layer: result_dict}.
    """
    if verbose:
        print(f"{'Layer':>6}  {'cos(probe,len)':>16}  "
              f"{'cos(probe,hdg)':>16}  {'ortho_LH acc':>13}")
        print("-" * 60)

    results = {}
    for layer in layers:
        if verbose:
            print(f"Layer {layer:>3}: ", end="", flush=True)
        results[layer] = build_orthogonal_directions(
            uncertain, certain, tok, model, layer,
            device=device, seed=seed, verbose=verbose,
        )
    return results
