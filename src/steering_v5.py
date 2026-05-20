import torch
import numpy as np
import random
import time
import json
import platform
from pathlib import Path
from datetime import datetime
import transformers

from src.hidden_states import load_model
from src.steering import build_uncertainty_vector, UNCERTAIN, CERTAIN
from src.steering_v3 import hedge_score, HEDGING_LEXICON, PROMPTS


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def generate_batched_scaled(prompts_list, tok, model, vector, alpha_frac,
                            layer_idx, hidden_norm_estimate,
                            device="mps", max_new_tokens=60, seed=42,
                            temperature=0.8, top_p=0.9,
                            repetition_penalty=1.2, min_new_tokens=20):
    """
    alpha_frac is the FRACTION of hidden-state norm to inject.
    alpha_frac=0.2 means perturbation magnitude = 20% of hidden state norm.
    """
    v_unit = (vector / vector.norm()).to(device)
    effective_alpha = alpha_frac * hidden_norm_estimate

    def steer_hook(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
            return (hs + effective_alpha * v_unit,) + output[1:]
        return output + effective_alpha * v_unit

    hook_handle = None
    if alpha_frac != 0.0:
        target = model.biogpt.layers[layer_idx]
        hook_handle = target.register_forward_hook(steer_hook)

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
                min_new_tokens=min_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
        return [tok.decode(seq, skip_special_tokens=True) for seq in out]
    finally:
        if hook_handle is not None:
            hook_handle.remove()


if __name__ == "__main__":
    tok, model = load_model()

    LAYER = 12
    # From your diagnostic, layer 12 hidden states have norm ~550
    HIDDEN_NORM = 550.0
    # alpha_frac = fraction of hidden state norm to perturb by
    ALPHA_FRACS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    SEEDS = list(range(20))

    v = build_uncertainty_vector(tok, model, layer=LAYER)
    print(f"Layer {LAYER}, vector norm: {v.norm().item():.2f}")
    print(f"Hidden state norm estimate: {HIDDEN_NORM}")
    print(f"Samples per alpha_frac: {len(SEEDS) * len(PROMPTS)}\n")

    results = {}
    for af in ALPHA_FRACS:
        t0 = time.time()
        scores = []
        for seed in SEEDS:
            outputs = generate_batched_scaled(
                PROMPTS, tok, model, v,
                alpha_frac=af, layer_idx=LAYER,
                hidden_norm_estimate=HIDDEN_NORM, seed=seed,
            )
            for prompt, full_out in zip(PROMPTS, outputs):
                gen = full_out.replace(prompt, "").strip()
                scores.append(hedge_score(gen))
        scores = np.array(scores)
        results[af] = {
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=1)),
            "frac_hedged": float((scores > 0).mean()),
        }
        effective = af * HIDDEN_NORM
        elapsed = time.time() - t0
        print(f"α_frac={af:.2f}  (eff α={effective:>6.1f})  "
              f"mean={scores.mean():.3f}  std={scores.std(ddof=1):.3f}  "
              f"frac_hedged={(scores > 0).mean():.2f}  [{elapsed:.1f}s]")

    Path("results").mkdir(exist_ok=True)
    with open(f"results/alpha_frac_sweep_layer{LAYER}.json", "w") as f:
        json.dump({"layer": LAYER, "hidden_norm": HIDDEN_NORM,
                   "results": results}, f, indent=2)