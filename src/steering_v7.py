import torch
import numpy as np
import random
import json
import time
from pathlib import Path
from src.hidden_states import load_model
from src.steering_v3 import hedge_score, PROMPTS


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_probe_vector(layer):
    with open("data/processed/probe_vectors.json") as f:
        data = json.load(f)
    return torch.tensor(data["vectors"][str(layer)], dtype=torch.float32)


def generate_steered(prompts_list, tok, model, vector_unit, alpha_frac,
                     layer_idx, hidden_norm, device="mps",
                     max_new_tokens=60, seed=42):
    v = vector_unit.to(device)
    effective_alpha = alpha_frac * hidden_norm

    def steer_hook(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
            return (hs + effective_alpha * v,) + output[1:]
        return output + effective_alpha * v

    handle = None
    if alpha_frac != 0.0:
        handle = model.biogpt.layers[layer_idx].register_forward_hook(steer_hook)

    try:
        set_all_seeds(seed)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        inputs = tok(prompts_list, return_tensors="pt",
                     padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=20,
                do_sample=True, top_p=0.9, temperature=0.8,
                repetition_penalty=1.2,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
        return [tok.decode(seq, skip_special_tokens=True) for seq in out]
    finally:
        if handle is not None:
            handle.remove()


HIDDEN_NORMS = {
    8: 401.0,
    12: 552.0,
    16: 616.0,
    20: 624.0,
    23: 518.0,
}


if __name__ == "__main__":
    tok, model = load_model()
    LAYERS = [12, 16, 20]
    ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    SEEDS = list(range(10))

    results = {}
    for layer in LAYERS:
        print(f"\n{'='*60}\nLAYER {layer}\n{'='*60}")
        raw_vec = load_probe_vector(layer)
        vec_unit = raw_vec / raw_vec.norm()
        hidden_norm = HIDDEN_NORMS[layer]

        layer_results = {}
        for af in ALPHAS:
            t0 = time.time()
            scores = []
            for seed in SEEDS:
                outs = generate_steered(
                    PROMPTS, tok, model, vec_unit,
                    alpha_frac=af, layer_idx=layer,
                    hidden_norm=hidden_norm, seed=seed,
                )
                for prompt, full_out in zip(PROMPTS, outs):
                    gen = full_out.replace(prompt, "").strip()
                    scores.append(hedge_score(gen))
            scores = np.array(scores)
            layer_results[af] = {
                "mean": float(scores.mean()),
                "std": float(scores.std(ddof=1)),
                "frac_hedged": float((scores > 0).mean()),
            }
            print(f"  α_frac={af:.2f}  mean={scores.mean():.3f}  "
                  f"frac_hedged={(scores > 0).mean():.2f}  "
                  f"[{time.time()-t0:.1f}s]")
        results[layer] = layer_results

    Path("results").mkdir(exist_ok=True)
    with open("results/probe_steering_coarse.json", "w") as f:
        json.dump(results, f, indent=2)