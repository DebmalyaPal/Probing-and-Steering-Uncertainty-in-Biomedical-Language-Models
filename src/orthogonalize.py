import torch
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from src.hidden_states import load_model, get_mean_hidden_state


def get_features_and_lengths(sentences, tok, model, layer, device="mps"):
    """Extract mean-pooled features AND token-length for each sentence."""
    feats = []
    lengths = []
    for s in sentences:
        h = get_mean_hidden_state(s, tok, model, layer, device)
        feats.append(h.numpy())
        # Token length as tokenized by BioGPT
        n_tokens = len(tok.encode(s, add_special_tokens=False))
        lengths.append(n_tokens)
    return np.stack(feats), np.array(lengths)


def build_length_direction(X, lengths):
    """
    Find the direction in activation space most correlated with length.
    Use linear regression: length ~ X @ w. The weight vector w points in
    the direction of activation space that best predicts length.
    """
    reg = LinearRegression()
    reg.fit(X, lengths)
    return reg.coef_  # shape: (hidden_dim,)


def orthogonalize(v_target, v_confound):
    """
    Project v_confound out of v_target.
    Returns the component of v_target orthogonal to v_confound.
    """
    v_target = np.asarray(v_target, dtype=np.float64)
    v_confound = np.asarray(v_confound, dtype=np.float64)
    v_conf_unit = v_confound / np.linalg.norm(v_confound)
    projection = np.dot(v_target, v_conf_unit) * v_conf_unit
    return v_target - projection


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    with open("data/processed/bioscope_contrast.json") as f:
        data = json.load(f)
    U, C = data["uncertain"], data["certain"]

    tok, model = load_model()
    LAYERS = [12, 16, 20]

    print(f"{'Layer':>6} {'mean_len_U':>11} {'mean_len_C':>11} "
          f"{'cos(probe,len)':>15} {'||probe||':>10} {'||ortho||':>10}")
    print("-" * 75)

    # First show that length differs between classes
    tok_u_lens = [len(tok.encode(s, add_special_tokens=False)) for s in U]
    tok_c_lens = [len(tok.encode(s, add_special_tokens=False)) for s in C]
    print(f"\nSanity check — uncertain mean tokens: {np.mean(tok_u_lens):.1f}, "
          f"certain mean tokens: {np.mean(tok_c_lens):.1f}")
    print(f"Difference: {np.mean(tok_u_lens) - np.mean(tok_c_lens):+.2f} tokens\n")

    orthogonal_vectors = {}
    for layer in LAYERS:
        # Extract features + lengths for combined set
        all_sentences = U + C
        all_lengths = np.array(tok_u_lens + tok_c_lens)

        X, _ = get_features_and_lengths(all_sentences, tok, model, layer)
        y = np.array([1] * len(U) + [0] * len(C))

        # Step 1: probe direction (uncertainty)
        probe = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        probe.fit(X, y)
        v_probe = probe.coef_[0]

        # Step 2: length direction
        v_length = build_length_direction(X, all_lengths)

        # Step 3: orthogonalize
        v_ortho = orthogonalize(v_probe, v_length)

        cos_sim = cosine_similarity(v_probe, v_length)

        print(f"{layer:>6} {np.mean(tok_u_lens):>11.1f} {np.mean(tok_c_lens):>11.1f} "
              f"{cos_sim:>15.4f} {np.linalg.norm(v_probe):>10.4f} "
              f"{np.linalg.norm(v_ortho):>10.4f}")

        orthogonal_vectors[layer] = {
            "v_probe": v_probe.tolist(),
            "v_length": v_length.tolist(),
            "v_ortho": v_ortho.tolist(),
            "cos_probe_length": float(cos_sim),
            "norm_probe": float(np.linalg.norm(v_probe)),
            "norm_ortho": float(np.linalg.norm(v_ortho)),
        }

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/orthogonal_vectors.json", "w") as f:
        json.dump(orthogonal_vectors, f, indent=2)
    print("\nSaved to data/processed/orthogonal_vectors.json")