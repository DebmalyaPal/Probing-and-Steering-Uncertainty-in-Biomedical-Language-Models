import torch
import json
from src.hidden_states import load_model
from src.steering_v3 import PROMPTS, hedge_score
from src.steering_v8 import build_directions, generate_steered, HIDDEN_NORMS


if __name__ == "__main__":
    tok, model = load_model()
    LAYER = 16
    ALPHAS = [0.0, 0.10, 0.20, 0.25]
    SEEDS = [42, 123, 777]

    v_probe, v_ortho = build_directions(tok, model, LAYER)

    hidden_norm = HIDDEN_NORMS[LAYER]

    for direction_name, vec in [("PROBE", v_probe), ("ORTHO", v_ortho)]:
        print(f"\n\n{'#'*75}")
        print(f"# DIRECTION: {direction_name}")
        print(f"{'#'*75}")
        vec_unit = vec / vec.norm()

        for af in ALPHAS:
            print(f"\n{'='*70}")
            print(f"α_frac = {af}")
            print(f"{'='*70}")
            for seed in SEEDS:
                outs = generate_steered(
                    PROMPTS, tok, model, vec_unit,
                    alpha_frac=af, layer_idx=LAYER,
                    hidden_norm=hidden_norm, seed=seed,
                )
                for prompt, full_out in zip(PROMPTS, outs):
                    gen = full_out.replace(prompt, "").strip()
                    score = hedge_score(gen)
                    print(f"\n[seed={seed}, prompt={prompt[:25]}...]")
                    print(f"  hedge_score={score:.2f}")
                    print(f"  output: {gen[:250]}")