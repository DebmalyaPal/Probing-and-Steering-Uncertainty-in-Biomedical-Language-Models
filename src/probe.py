import torch
import numpy as np
import json
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.hidden_states import load_model, get_mean_hidden_state

def get_features(sentences, tok, model, layer, device="mps"):
    """Extract mean-pooled hidden states for all sentences at a given layer."""
    feats = []
    for s in sentences:
        h = get_mean_hidden_state(s, tok, model, layer, device)
        feats.append(h.numpy())
    return np.stack(feats)


def run_probe_at_layer(X_u, X_c, layer_name, n_splits=5, seed=42):
    """Train logistic regression probe, return mean/std CV accuracy."""
    X = np.vstack([X_u, X_c])
    y = np.array([1] * len(X_u) + [0] * len(X_c))

    clf = LogisticRegression(max_iter=2000, C=0.1, random_state=seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=1)
    return scores


if __name__ == "__main__":
    # Load BioScope contrast set
    with open("data/processed/bioscope_contrast.json") as f:
        data = json.load(f)
    U, C = data["uncertain"], data["certain"]
    print(f"Loaded {len(U)} uncertain, {len(C)} certain sentences")

    tok, model = load_model()

    LAYERS = [0, 4, 8, 12, 16, 20, 22, 23, 24]

    print(f"\n{'Layer':>6} {'CV accuracy':>15} {'std':>8}  {'95% CI':>15}")
    print("-" * 50)

    results = {}
    for layer in LAYERS:
        t0 = time.time()
        print(f"Extracting features at layer {layer}...", end=" ", flush=True)
        X_u = get_features(U, tok, model, layer)
        X_c = get_features(C, tok, model, layer)
        print(f"done [{time.time() - t0:.1f}s]")

        scores = run_probe_at_layer(X_u, X_c, layer_name=f"layer_{layer}")
        mean_acc = scores.mean()
        std_acc = scores.std(ddof=1)
        # 95% CI: mean ± 1.96 * (std / sqrt(n))
        ci_margin = 1.96 * std_acc / np.sqrt(len(scores))
        ci_low = mean_acc - ci_margin
        ci_high = mean_acc + ci_margin

        results[layer] = {
            "mean_acc": float(mean_acc),
            "std_acc": float(std_acc),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "fold_scores": [float(s) for s in scores],
        }
        print(f"{layer:>6} {mean_acc:>14.2%}  {std_acc:>7.2%}  "
              f"[{ci_low:>5.2%}, {ci_high:>5.2%}]")

    Path("results").mkdir(exist_ok=True)
    with open("results/probe_accuracy_per_layer.json", "w") as f:
        json.dump({
            "n_uncertain": len(U),
            "n_certain": len(C),
            "n_splits": 5,
            "results": results,
        }, f, indent=2)
    print("\nSaved to results/probe_accuracy_per_layer.json")