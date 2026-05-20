# src/sample_size_convergence.py
import numpy as np
import json
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pathlib import Path
from src.hidden_states import load_model, get_mean_hidden_state

# Load the full pool of filtered BioScope sentences, not just 200 per class
from src.bioscope_parser import build_balanced_contrast_set

if __name__ == "__main__":
    tok, model = load_model()
    LAYER = 16
    SAMPLE_SIZES = [50, 100, 200, 400, 800]

    print(f"{'n_per_class':>12} {'cv_acc':>10} {'std':>8}")
    print("-" * 35)

    results = {}
    for n in SAMPLE_SIZES:
        try:
            U, C = build_balanced_contrast_set(
                "data/bioscope", max_per_class=n, seed=42
            )
        except ValueError:
            print(f"Not enough data for n={n}, skipping")
            continue

        X_u = np.stack([
            get_mean_hidden_state(s, tok, model, LAYER).numpy() for s in U
        ])
        X_c = np.stack([
            get_mean_hidden_state(s, tok, model, LAYER).numpy() for s in C
        ])
        X = np.vstack([X_u, X_c])
        y = np.array([1] * len(U) + [0] * len(C))

        clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")

        results[n] = {
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=1)),
        }
        print(f"{n:>12} {scores.mean():>9.2%} {scores.std(ddof=1):>7.2%}")

    Path("results").mkdir(exist_ok=True)
    with open("results/sample_size_convergence.json", "w") as f:
        json.dump(results, f, indent=2)