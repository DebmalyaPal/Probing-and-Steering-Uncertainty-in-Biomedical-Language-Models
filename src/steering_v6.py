import torch
import numpy as np
import random
import time
import json
from pathlib import Path

from src.hidden_states import load_model
from src.steering import build_uncertainty_vector
from src.steering_v3 import hedge_score, PROMPTS


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def generate_multilayer_steered(prompts_list, tok, model, vectors_by_layer,
                                alpha_frac, hidden_norm_estimate,
                                device="mps", max_new_tokens=60, seed=42):
    """
    Inject steering at multiple layers simultaneously.
    vectors_by_layer: dict of {layer_idx: vector}
    """
    handles = []

    def make_hook(v_unit, effective_alpha):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
                return (hs + effective_alpha * v_unit,) + output[1:]
            return output + effective_alpha * v_unit
        return hook

    if alpha_frac != 0.0:
        for layer_idx, v in vectors_by_layer.items():
            v_unit = (v / v.norm()).to(device)
            effective = alpha_frac * hidden_norm_estimate
            h = model.biogpt.layers[layer_idx].register_forward_hook(
                make_hook(v_unit, effective)
            )
            handles.append(h)

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
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.2,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
        return [tok.decode(seq, skip_special_tokens=True) for seq in out]
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    tok, model = load_model()

    TARGET_LAYERS = [8, 12, 16, 20]
    HIDDEN_NORM = 550.0
    # Multi-layer means each layer's effective alpha is smaller
    ALPHA_FRACS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    SEEDS = list(range(15))

    print("Building vectors at each target layer...")
    vectors = {L: build_uncertainty_vector(tok, model, layer=L)
               for L in TARGET_LAYERS}
    for L, v in vectors.items():
        print(f"  Layer {L}: norm={v.norm().item():.2f}")

    print(f"\nSteering at layers: {TARGET_LAYERS}")
    print(f"Samples per alpha: {len(SEEDS) * len(PROMPTS)}\n")

    results = {}
    for af in ALPHA_FRACS:
        t0 = time.time()
        scores = []
        for seed in SEEDS:
            outputs = generate_multilayer_steered(
                PROMPTS, tok, model, vectors,
                alpha_frac=af, hidden_norm_estimate=HIDDEN_NORM,
                seed=seed,
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
        elapsed = time.time() - t0
        print(f"α_frac={af:.2f}  "
              f"mean={scores.mean():.3f}  "
              f"std={scores.std(ddof=1):.3f}  "
              f"frac_hedged={(scores > 0).mean():.2f}  "
              f"[{elapsed:.1f}s]")

    Path("results").mkdir(exist_ok=True)
    with open("results/multilayer_sweep.json", "w") as f:
        json.dump({
            "target_layers": TARGET_LAYERS,
            "hidden_norm": HIDDEN_NORM,
            "results": results,
        }, f, indent=2)