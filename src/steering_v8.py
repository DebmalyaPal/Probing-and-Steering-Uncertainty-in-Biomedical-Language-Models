import torch
import numpy as np
import random
import json
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from src.hidden_states import load_model, get_mean_hidden_state
from src.steering_v3 import hedge_score, PROMPTS


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


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def generate_steered(prompts_list, tok, model, vec_unit, alpha_frac,
                     layer_idx, hidden_norm, device="mps",
                     max_new_tokens=60, seed=42):
    v = vec_unit.to(device)
    effective = alpha_frac * hidden_norm

    def steer_hook(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
            return (hs + effective * v,) + output[1:]
        return output + effective * v

    handle = None
    if alpha_frac != 0.0:
        handle = model.biogpt.layers[layer_idx].register_forward_hook(steer_hook)

    try:
        set_all_seeds(seed)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        inputs = tok(prompts_list, return_tensors="pt", padding=True,
                     truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, min_new_tokens=20,
                do_sample=True, top_p=0.9, temperature=0.8,
                repetition_penalty=1.2, pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
        return [tok.decode(seq, skip_special_tokens=True) for seq in out]
    finally:
        if handle is not None:
            handle.remove()


HIDDEN_NORMS = {8: 401.0, 12: 552.0, 16: 616.0, 20: 624.0, 23: 518.0}


def build_directions(tok, model, layer):
    """Build raw probe and length+hedge orthogonalized directions."""
    with open("data/processed/bioscope_contrast.json") as f:
        data = json.load(f)
    U, C = data["uncertain"], data["certain"]
    all_sentences = U + C
    y = np.array([1] * len(U) + [0] * len(C))

    lengths = np.array([count_tokens(s, tok) for s in all_sentences])
    hedges = np.array([count_hedges(s) for s in all_sentences])

    X = np.stack([
        get_mean_hidden_state(s, tok, model, layer).numpy()
        for s in all_sentences
    ])

    v_probe = LogisticRegression(max_iter=2000, C=0.1, random_state=42).fit(X, y).coef_[0]
    v_length = LinearRegression().fit(X, lengths).coef_
    v_hedge = LinearRegression().fit(X, hedges).coef_

    v_ortho = orthogonalize(orthogonalize(v_probe, v_length), v_hedge)

    return torch.tensor(v_probe, dtype=torch.float32), torch.tensor(v_ortho, dtype=torch.float32)


if __name__ == "__main__":
    tok, model = load_model()
    LAYER = 16
    ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    SEEDS = list(range(10))

    print(f"Building directions at layer {LAYER}...")
    v_probe, v_ortho = build_directions(tok, model, LAYER)
    print(f"  probe norm: {v_probe.norm().item():.4f}")
    print(f"  ortho norm: {v_ortho.norm().item():.4f}\n")

    results = {}
    for name, vec in [("probe", v_probe), ("ortho", v_ortho)]:
        print(f"\n{'='*60}\n{name.upper()} direction\n{'='*60}")
        vec_unit = vec / vec.norm()
        hidden_norm = HIDDEN_NORMS[LAYER]

        direction_results = {}
        for af in ALPHAS:
            t0 = time.time()
            scores = []
            for seed in SEEDS:
                outs = generate_steered(
                    PROMPTS, tok, model, vec_unit,
                    alpha_frac=af, layer_idx=LAYER,
                    hidden_norm=hidden_norm, seed=seed,
                )
                for prompt, full_out in zip(PROMPTS, outs):
                    gen = full_out.replace(prompt, "").strip()
                    scores.append(hedge_score(gen))
            scores = np.array(scores)
            direction_results[af] = {
                "mean": float(scores.mean()),
                "std": float(scores.std(ddof=1)),
                "frac_hedged": float((scores > 0).mean()),
            }
            print(f"  α_frac={af:.2f}  mean={scores.mean():.3f}  "
                  f"frac_hedged={(scores > 0).mean():.2f}  "
                  f"[{time.time()-t0:.1f}s]")
        results[name] = direction_results

    Path("results").mkdir(exist_ok=True)
    with open("results/steering_v8_probe_vs_ortho.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/steering_v8_probe_vs_ortho.json")