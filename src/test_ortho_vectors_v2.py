import numpy as np
import json
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from src.hidden_states import load_model, get_mean_hidden_state


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


def orthogonalize(v_target, v_confound):
    v_conf_unit = v_confound / np.linalg.norm(v_confound)
    return v_target - np.dot(v_target, v_conf_unit) * v_conf_unit


def classify_via_projection(proj_train, y_train, proj_test, y_test):
    """Fit 1D logistic on train projection, score on test projection."""
    clf = LogisticRegression(max_iter=2000)
    clf.fit(proj_train.reshape(-1, 1), y_train)
    return clf.score(proj_test.reshape(-1, 1), y_test)


if __name__ == "__main__":
    with open("data/processed/bioscope_contrast.json") as f:
        data = json.load(f)
    U, C = data["uncertain"], data["certain"]
    all_sentences = U + C
    y = np.array([1] * len(U) + [0] * len(C))

    tok, model = load_model()

    lengths = np.array([count_tokens(s, tok) for s in all_sentences])
    hedges = np.array([count_hedges(s) for s in all_sentences])

    LAYERS = [12, 16, 20]

    print(f"{'Layer':>6} {'full_probe':>12} {'proj_probe':>12} "
          f"{'proj_ortho_L':>14} {'proj_ortho_LH':>15}")
    print("-" * 65)

    for layer in LAYERS:
        # Extract features once
        X = np.stack([
            get_mean_hidden_state(s, tok, model, layer).numpy()
            for s in all_sentences
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        full_probe_accs = []
        proj_probe_accs = []
        proj_L_accs = []
        proj_LH_accs = []

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            len_tr = lengths[train_idx]
            hedge_tr = hedges[train_idx]

            # Step 1: full-dim probe (standard CV accuracy)
            full_clf = LogisticRegression(max_iter=2000, C=0.1)
            full_clf.fit(X_tr, y_tr)
            full_probe_accs.append(full_clf.score(X_te, y_te))

            # Step 2: probe direction from TRAIN only
            v_probe = full_clf.coef_[0]

            # Step 3: confound directions from TRAIN only
            v_length = LinearRegression().fit(X_tr, len_tr).coef_
            v_hedge = LinearRegression().fit(X_tr, hedge_tr).coef_

            # Step 4: orthogonalize
            v_ortho_L = orthogonalize(v_probe, v_length)
            v_ortho_LH = orthogonalize(v_ortho_L, v_hedge)

            # Step 5: project TEST fold onto each direction, classify
            def fold_acc(v):
                p_tr = X_tr @ (v / np.linalg.norm(v))
                p_te = X_te @ (v / np.linalg.norm(v))
                return classify_via_projection(p_tr, y_tr, p_te, y_te)

            proj_probe_accs.append(fold_acc(v_probe))
            proj_L_accs.append(fold_acc(v_ortho_L))
            proj_LH_accs.append(fold_acc(v_ortho_LH))

        print(f"{layer:>6} "
              f"{np.mean(full_probe_accs):>11.2%} "
              f"{np.mean(proj_probe_accs):>11.2%} "
              f"{np.mean(proj_L_accs):>13.2%} "
              f"{np.mean(proj_LH_accs):>14.2%}")