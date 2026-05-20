"""Final experiment for the paper. Runs full sweep with all metrics."""
import torch
import numpy as np
import random
import json
import time
import platform
from pathlib import Path
from datetime import datetime
import transformers

from src.hidden_states import load_model
from src.steering_v8 import build_directions, generate_steered, HIDDEN_NORMS
from src.metrics import compute_all_metrics, load_english_vocab


PROMPTS = [
    "Findings: the patient's chest imaging demonstrates",
    "Impression: based on the radiographic findings,",
    "On examination of the chest radiograph, there is",
    "The radiologist's interpretation of the chest X-ray:",
    "Clinical assessment of the thoracic imaging indicates",
]


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_full_experiment():
    tok, model = load_model()
    english_vocab = load_english_vocab()
    if english_vocab is None:
        print("Warning: using heuristic token validity (nltk not installed)")

    LAYER = 16
    ALPHAS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]
    SEEDS = list(range(20))
    DIRECTIONS = ["probe", "ortho"]

    print(f"Building steering directions at layer {LAYER}...")
    v_probe, v_ortho = build_directions(tok, model, LAYER)
    direction_vectors = {"probe": v_probe, "ortho": v_ortho}

    hidden_norm = HIDDEN_NORMS[LAYER]
    total_gens = len(ALPHAS) * len(SEEDS) * len(PROMPTS) * len(DIRECTIONS)
    print(f"Total generations: {total_gens}\n")

    all_records = []
    summary = {}

    for direction_name in DIRECTIONS:
        vec = direction_vectors[direction_name]
        vec_unit = vec / vec.norm()
        summary[direction_name] = {}

        for af in ALPHAS:
            t0 = time.time()
            samples_this_config = []

            for seed in SEEDS:
                outs = generate_steered(
                    PROMPTS, tok, model, vec_unit,
                    alpha_frac=af, layer_idx=LAYER,
                    hidden_norm=hidden_norm, seed=seed,
                )
                for prompt, full_out in zip(PROMPTS, outs):
                    gen = full_out.replace(prompt, "").strip()
                    m = compute_all_metrics(gen, tok, model,
                                             english_vocab=english_vocab)
                    record = {
                        "direction": direction_name,
                        "alpha_frac": af,
                        "layer": LAYER,
                        "seed": seed,
                        "prompt": prompt,
                        "generation": gen,
                        **m,
                    }
                    all_records.append(record)
                    samples_this_config.append(m)

            # Aggregate
            agg = {}
            for key in samples_this_config[0].keys():
                vals = [s[key] for s in samples_this_config
                        if not (isinstance(s[key], float)
                                and (np.isinf(s[key]) or np.isnan(s[key])))]
                if vals:
                    agg[f"{key}_mean"] = float(np.mean(vals))
                    agg[f"{key}_std"] = float(np.std(vals, ddof=1))
                    agg[f"{key}_median"] = float(np.median(vals))
                agg[f"{key}_n_valid"] = len(vals)
            summary[direction_name][str(af)] = agg

            elapsed = time.time() - t0
            print(f"{direction_name:>6}  α={af:.3f}  "
                  f"hedge={agg.get('hedge_score_mean', 0):.3f}  "
                  f"lex_div={agg.get('lexical_diversity_mean', 0):.3f}  "
                  f"tok_val={agg.get('token_validity_mean', 0):.3f}  "
                  f"ppl={agg.get('perplexity_mean', 0):.1f}  "
                  f"[{elapsed:.0f}s]")

    env = {
        "timestamp": datetime.now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
    }
    config = {
        "layer": LAYER, "alphas": ALPHAS, "seeds": SEEDS,
        "prompts": PROMPTS, "directions": DIRECTIONS,
        "hidden_norm_estimate": hidden_norm,
        "n_contrast_uncertain": 200, "n_contrast_certain": 200,
        "model": "microsoft/biogpt",
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/final_experiment.json", "w") as f:
        json.dump({
            "config": config, "environment": env,
            "summary": summary, "records": all_records,
        }, f, indent=2)

    print(f"\nSaved {len(all_records)} per-sample records to results/final_experiment.json")


if __name__ == "__main__":
    run_full_experiment()