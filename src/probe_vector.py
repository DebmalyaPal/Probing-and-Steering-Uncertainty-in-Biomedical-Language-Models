import torch
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from src.hidden_states import load_model, get_mean_hidden_state

def get_features(sentences, tok, model, layer, device="mps"):
    return np.stack([
        get_mean_hidden_state(s, tok, model, layer, device).numpy()
        for s in sentences
    ])


def build_probe_vector(X_u, X_c, seed=42):
    """
    Train logistic regression on all data, return the weight vector.
    This is the direction that best separates uncertain from certain.
    """
    X = np.vstack([X_u, X_c])
    y = np.array([1] * len(X_u) + [0] * len(X_c))
    clf = LogisticRegression(max_iter=2000, C=0.1, random_state=seed)
    clf.fit(X, y)
    # coef_ shape is (1, hidden_dim); squeeze to (hidden_dim,)
    return clf.coef_[0], clf.score(X, y)


if __name__ == "__main__":
    with open("data/processed/bioscope_contrast.json") as f:
        data = json.load(f)
    U, C = data["uncertain"], data["certain"]

    tok, model = load_model()
    LAYERS = [8, 12, 16, 20, 23]

    print(f"{'Layer':>6} {'train_acc':>10} {'vec_norm':>10}")
    print("-" * 30)

    probe_vectors = {}
    for layer in LAYERS:
        X_u = get_features(U, tok, model, layer)
        X_c = get_features(C, tok, model, layer)
        vec, acc = build_probe_vector(X_u, X_c)
        probe_vectors[layer] = vec.tolist()
        print(f"{layer:>6} {acc:>9.2%} {np.linalg.norm(vec):>10.4f}")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/probe_vectors.json", "w") as f:
        json.dump({
            "layers": LAYERS,
            "vectors": probe_vectors,
            "n_uncertain": len(U),
            "n_certain": len(C),
            "classifier": "LogisticRegression(C=0.1)",
        }, f, indent=2)
    print(f"\nSaved probe vectors to data/processed/probe_vectors.json")