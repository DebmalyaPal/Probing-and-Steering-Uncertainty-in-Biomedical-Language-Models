import torch
import numpy as np
import random
import json
from src.hidden_states import load_model, get_mean_hidden_state

# Load the full BioScope contrast set we built
with open("data/processed/bioscope_contrast.json") as f:
    data = json.load(f)

ALL_UNCERTAIN = data["uncertain"]
ALL_CERTAIN = data["certain"]

# Train/test split — vector built from TRAIN, evaluated on TEST
# Use a fixed seed so the split is reproducible
SPLIT_SEED = 42
TEST_FRACTION = 0.25  # 25% held out

random.seed(SPLIT_SEED)
u_indices = list(range(len(ALL_UNCERTAIN)))
c_indices = list(range(len(ALL_CERTAIN)))
random.shuffle(u_indices)
random.shuffle(c_indices)

n_test_u = int(len(ALL_UNCERTAIN) * TEST_FRACTION)
n_test_c = int(len(ALL_CERTAIN) * TEST_FRACTION)

TEST_UNCERTAIN = [ALL_UNCERTAIN[i] for i in u_indices[:n_test_u]]
TRAIN_UNCERTAIN = [ALL_UNCERTAIN[i] for i in u_indices[n_test_u:]]
TEST_CERTAIN = [ALL_CERTAIN[i] for i in c_indices[:n_test_c]]
TRAIN_CERTAIN = [ALL_CERTAIN[i] for i in c_indices[n_test_c:]]

print(f"Train: {len(TRAIN_UNCERTAIN)} uncertain, {len(TRAIN_CERTAIN)} certain")
print(f"Test:  {len(TEST_UNCERTAIN)} uncertain, {len(TEST_CERTAIN)} certain\n")


def build_vector_from_train(tok, model, layer, device="mps"):
    """Build uncertainty vector from the TRAIN split only."""
    u_vecs = torch.stack([
        get_mean_hidden_state(s, tok, model, layer, device)
        for s in TRAIN_UNCERTAIN
    ])
    c_vecs = torch.stack([
        get_mean_hidden_state(s, tok, model, layer, device)
        for s in TRAIN_CERTAIN
    ])
    return u_vecs.mean(dim=0) - c_vecs.mean(dim=0)


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                         / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def classification_accuracy(u_projs, c_projs):
    """
    Simple accuracy: project onto vector, pick a threshold at the midpoint
    of the two means, classify each point.
    """
    u = np.array(u_projs)
    c = np.array(c_projs)
    threshold = (u.mean() + c.mean()) / 2
    # If u_mean > c_mean, uncertain should project higher than threshold
    if u.mean() > c.mean():
        correct = (u > threshold).sum() + (c <= threshold).sum()
    else:
        correct = (u <= threshold).sum() + (c > threshold).sum()
    return correct / (len(u) + len(c))


if __name__ == "__main__":
    tok, model = load_model()

    print(f"{'Layer':>6} {'u_mean':>10} {'c_mean':>10} {'diff':>8} "
          f"{'u_std':>8} {'c_std':>8} {'cohens_d':>10} {'acc':>8}")
    print("-" * 80)

    for layer in [4, 8, 12, 16, 20, 23]:
        # Build vector from TRAIN split only
        v = build_vector_from_train(tok, model, layer)
        v_unit = v / v.norm()

        # Evaluate on held-out TEST split
        u_proj = [get_mean_hidden_state(s, tok, model, layer=layer) @ v_unit
                  for s in TEST_UNCERTAIN]
        c_proj = [get_mean_hidden_state(s, tok, model, layer=layer) @ v_unit
                  for s in TEST_CERTAIN]

        u = [p.item() for p in u_proj]
        c = [p.item() for p in c_proj]

        d = cohens_d(u, c)
        acc = classification_accuracy(u, c)

        print(f"{layer:>6} {np.mean(u):>10.2f} {np.mean(c):>10.2f} "
              f"{np.mean(u)-np.mean(c):>8.3f} "
              f"{np.std(u, ddof=1):>8.3f} {np.std(c, ddof=1):>8.3f} "
              f"{d:>10.3f} {acc:>7.2%}")