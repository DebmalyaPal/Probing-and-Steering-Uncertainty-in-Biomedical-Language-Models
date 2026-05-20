import torch
import json
from src.hidden_states import load_model
from src.steering_v3 import PROMPTS, hedge_score
from src.steering_v7 import (
    load_probe_vector, generate_steered, HIDDEN_NORMS, set_all_seeds
)


if __name__ == "__main__":
    tok, model = load_model()
    LAYER = 16
    ALPHAS = [0.0, 0.10, 0.15, 0.20, 0.25]
    SEEDS = [42, 123, 777]

    raw_vec = load_probe_vector(LAYER)
    vec_unit = raw_vec / raw_vec.norm()
    hidden_norm = HIDDEN_NORMS[LAYER]

    print(f"=== LAYER {LAYER} STEERING SAMPLES ===\n")

    for af in ALPHAS:
        print(f"\n{'#'*70}")
        print(f"# α_frac = {af}")
        print(f"{'#'*70}")
        for seed in SEEDS:
            outs = generate_steered(
                PROMPTS, tok, model, vec_unit,
                alpha_frac=af, layer_idx=LAYER,
                hidden_norm=hidden_norm, seed=seed,
            )
            for prompt, full_out in zip(PROMPTS, outs):
                gen = full_out.replace(prompt, "").strip()
                score = hedge_score(gen)
                print(f"\n[seed={seed}, prompt='{prompt[:30]}...']")
                print(f"  hedge_score={score:.2f}")
                print(f"  output: {gen[:300]}")