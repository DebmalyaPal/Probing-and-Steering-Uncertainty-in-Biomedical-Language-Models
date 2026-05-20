import torch
from src.hidden_states import load_model
from src.steering import build_uncertainty_vector
from src.steering_v2 import generate_with_steering

PROMPTS = [
    "Findings: the patient's chest imaging demonstrates",
    "Impression: based on the radiographic findings,",
    "On examination of the chest radiograph, there is",
]

HEDGING_LEXICON = {
    "may": 1.0, "might": 1.0, "could": 1.0, "possibly": 1.0, "possible": 1.0,
    "probably": 1.0, "probable": 1.0, "likely": 0.7,
    "suggest": 1.0, "suggestive": 1.0, "suggests": 1.0,
    "consistent with": 1.0, "compatible with": 1.0,
    "cannot exclude": 1.5, "cannot rule out": 1.5,
    "appears": 0.8, "appear": 0.8, "appearing": 0.8,
    "uncertain": 1.2, "unclear": 1.2, "equivocal": 1.5,
    "indeterminate": 1.5, "nonspecific": 1.0,
    "no obvious": 1.0, "no definite": 1.2, "no evidence of": 1.0,
    "without clear": 1.0, "without definite": 1.2,
    "further evaluation": 1.0, "clinical correlation": 1.2,
    "differential": 0.8,
}

def hedge_score(text):
    t = text.lower()
    return sum(w * t.count(h) for h, w in HEDGING_LEXICON.items())

if __name__ == "__main__":
    tok, model = load_model()
    SEEDS = [42, 123, 777]
    LAYER = 12  # best candidate from your data
    ALPHAS = [0.0, 1.0, 2.0, 2.5, 3.0]

    v = build_uncertainty_vector(tok, model, layer=LAYER)
    print(f"Layer {LAYER}, vector norm: {v.norm().item():.2f}\n")

    for alpha in ALPHAS:
        scores = []
        samples = []
        for prompt in PROMPTS:
            for seed in SEEDS:
                out, _ = generate_with_steering(
                    prompt, tok, model, v,
                    alpha=alpha, layer_idx=LAYER, seed=seed,
                )
                generated = out.replace(prompt, "").strip()
                scores.append(hedge_score(generated))
                samples.append((prompt[:30], seed, generated[:100]))

        avg = sum(scores) / len(scores)
        print(f"α={alpha}  avg hedge score={avg:.2f}")
        # Show one example
        ex = samples[0]
        print(f"    example: ...{ex[2]}\n")