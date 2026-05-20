import torch
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from src.hidden_states import load_model, get_mean_hidden_state
from src.orthogonalize import (
    orthogonalize, cosine_similarity, get_features_and_lengths
)

HEDGING_TERMS = [
    "may", "might", "could", "possibly", "possible", "probably", "probable",
    "suggest", "suggests", "suggestive", "likely", "unclear", "uncertain",
    "appears", "appear", "seems", "seem", "would", "should",
    "consistent with", "compatible with", "cannot", "unknown", "perhaps",
    "indicate", "indicates", "indication",
]

def count_hedges(text):
    t = text.lower()
    return sum(t.count(h) for h in HEDGING_TERMS)

def count_tokens(text, tok):
    return len(tok.encode(text, add_special_tokens=False))


if __name__ == "__main__":
    with open("data/processed/bioscope_contrast.json") as f:
        data = json.load(f)
    U, C = data["uncertain"], data["certain"]
    all_sentences = U + C
    y = np.array([1] * len(U) + [0] * len(C))

    tok, model = load_model()

    # Count hedging words per sentence
    hedge_counts = np.array([count_hedges(s) for s in all_sentences])
    print(f"Mean hedge count — uncertain: {hedge_counts[:len(U)].mean():.2f}, "
          f"certain: {hedge_counts[len(U):].mean():.2f}\n")

    LAYERS = [12, 16, 20]

    print(f"{'Layer':>6} {'cos(probe,hedge)':>18} {'cos(probe,length)':>19} "
          f"{'||ortho_L||':>13} {'||ortho_LH||':>14}")
    print("-" * 80)

    results = {}
    for layer in LAYERS:
        lengths = np.array([count_tokens(s, tok) for s in all_sentences])
        X, _ = get_features_and_lengths(all_sentences, tok, model, layer)

        # Probe direction
        probe = LogisticRegression(max_iter=2000, C=0.1, random_state=42).fit(X, y)
        v_probe = probe.coef_[0]

        # Length and hedge-count confound directions
        v_length = LinearRegression().fit(X, lengths).coef_
        v_hedge = LinearRegression().fit(X, hedge_counts).coef_

        # Orthogonalize step by step
        v_ortho_L = orthogonalize(v_probe, v_length)
        v_ortho_LH = orthogonalize(v_ortho_L, v_hedge)

        cos_hedge = cosine_similarity(v_probe, v_hedge)
        cos_length = cosine_similarity(v_probe, v_length)

        print(f"{layer:>6} {cos_hedge:>18.4f} {cos_length:>19.4f} "
              f"{np.linalg.norm(v_ortho_L):>13.4f} "
              f"{np.linalg.norm(v_ortho_LH):>14.4f}")

        results[layer] = {
            "v_probe": v_probe.tolist(),
            "v_ortho_length": v_ortho_L.tolist(),
            "v_ortho_length_hedge": v_ortho_LH.tolist(),
            "cos_probe_hedge": float(cos_hedge),
            "cos_probe_length": float(cos_length),
            "norm_probe": float(np.linalg.norm(v_probe)),
            "norm_ortho_L": float(np.linalg.norm(v_ortho_L)),
            "norm_ortho_LH": float(np.linalg.norm(v_ortho_LH)),
        }

        # Also: does the ortho_LH direction still classify uncertainty?
        # Project and check accuracy
        v_lh_unit = v_ortho_LH / np.linalg.norm(v_ortho_LH)
        projections = X @ v_lh_unit
        # Simple threshold classifier
        threshold = np.median(projections)
        preds = (projections > threshold).astype(int)
        if preds.mean() > 0.5:
            acc = (preds == y).mean()
        else:
            acc = ((1 - preds) == y).mean()
        print(f"        → ortho_LH direction classification acc: {acc:.2%}")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/orthogonal_vectors_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to data/processed/orthogonal_vectors_v2.json")