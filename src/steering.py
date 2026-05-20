import json
import torch
from pathlib import Path
from src.hidden_states import load_model, get_mean_hidden_state

def load_contrast_set(path="data/processed/bioscope_contrast.json"):
    with open(path) as f:
        data = json.load(f)
    return data["uncertain"], data["certain"]

UNCERTAIN, CERTAIN = load_contrast_set()

def build_uncertainty_vector(tok, model, layer=23, device="mps"):
    """Build mean-difference vector using mean-pooled hidden states."""
    u_vecs = torch.stack([
        get_mean_hidden_state(s, tok, model, layer, device) for s in UNCERTAIN
    ])
    c_vecs = torch.stack([
        get_mean_hidden_state(s, tok, model, layer, device) for s in CERTAIN
    ])
    return u_vecs.mean(dim=0) - c_vecs.mean(dim=0)

if __name__ == "__main__":
    tok, model = load_model()
    v = build_uncertainty_vector(tok, model, layer=23)
    print(f"Vector at layer 23: norm={v.norm().item():.3f}")