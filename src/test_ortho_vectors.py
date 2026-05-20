import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.hidden_states import load_model, get_mean_hidden_state


def project(X, v):
    v_unit = v / np.linalg.norm(v)
    return X @ v_unit


if __name__ == "__main__":
    with open("data/processed/bioscope_contrast.json") as f:
        data = json.load(f)
    U, C = data["uncertain"], data["certain"]
    all_sentences = U + C
    y = np.array([1] * len(U) + [0] * len(C))

    with open("data/processed/orthogonal_vectors_v2.json") as f:
        ortho_data = json.load(f)

    tok, model = load_model()

    print(f"{'Layer':>6} {'probe_cv_acc':>14} {'ortho_L_cv_acc':>16} "
          f"{'ortho_LH_cv_acc':>17}")
    print("-" * 60)

    for layer_str, vecs in ortho_data.items():
        layer = int(layer_str)

        # Extract features at this layer
        X = np.stack([
            get_mean_hidden_state(s, tok, model, layer).numpy()
            for s in all_sentences
        ])

        v_probe = np.array(vecs["v_probe"])
        v_ortho_L = np.array(vecs["v_ortho_length"])
        v_ortho_LH = np.array(vecs["v_ortho_length_hedge"])

        # 1D projections
        proj_probe = project(X, v_probe).reshape(-1, 1)
        proj_L = project(X, v_ortho_L).reshape(-1, 1)
        proj_LH = project(X, v_ortho_LH).reshape(-1, 1)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Train a 1D logistic regression on each projection
        def cv_acc(proj):
            clf = LogisticRegression(max_iter=2000)
            return cross_val_score(clf, proj, y, cv=skf, scoring="accuracy").mean()

        print(f"{layer:>6} {cv_acc(proj_probe):>13.2%} "
              f"{cv_acc(proj_L):>15.2%} {cv_acc(proj_LH):>16.2%}")