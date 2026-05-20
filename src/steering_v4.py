import torch
import numpy as np
from src.hidden_states import load_model
from src.steering import build_uncertainty_vector
from src.steering_v2 import generate_with_steering
from src.steering_v3 import hedge_score, PROMPTS

if __name__ == "__main__":
    tok, model = load_model()
    LAYER = 12
    ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    SEEDS = list(range(20))   # 20 seeds × 3 prompts = 60 samples per alpha

    v = build_uncertainty_vector(tok, model, layer=LAYER)
    print(f"Layer {LAYER}, vector norm: {v.norm().item():.2f}")
    print(f"Samples per alpha: {len(SEEDS) * len(PROMPTS)}\n")

    results = {}
    for alpha in ALPHAS:
        scores = []
        for prompt in PROMPTS:
            for seed in SEEDS:
                out, _ = generate_with_steering(
                    prompt, tok, model, v,
                    alpha=alpha, layer_idx=LAYER, seed=seed,
                )
                generated = out.replace(prompt, "").strip()
                scores.append(hedge_score(generated))
        scores = np.array(scores)
        results[alpha] = scores
        print(f"α={alpha:>4}  mean={scores.mean():.3f}  "
              f"std={scores.std():.3f}  "
              f"median={np.median(scores):.3f}  "
              f"frac_hedged={(scores > 0).mean():.2f}")