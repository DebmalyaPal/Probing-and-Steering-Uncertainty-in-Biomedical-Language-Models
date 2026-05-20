import torch
from src.hidden_states import load_model, get_last_token_hidden_state
from src.steering import UNCERTAIN, CERTAIN, build_uncertainty_vector

def generate_with_steering(prompt, tok, model, vector, alpha=0.0, layer_idx=16,
                           device="mps", max_new_tokens=60, seed=42):
    vector = vector.to(device)
    hook_handle = None
    hook_fire_count = [0]

    def steer_hook(module, input, output):
        hook_fire_count[0] += 1
        if isinstance(output, tuple):
            hs = output[0]
            return (hs + alpha * vector,) + output[1:]
        return output + alpha * vector

    if alpha != 0.0:
        target = model.biogpt.layers[layer_idx]
        hook_handle = target.register_forward_hook(steer_hook)

    try:
        # Fix seed so alpha is the only variable
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                min_new_tokens=20,  # prevent early EOS collapse
                repetition_penalty=1.2,
            )
        return tok.decode(out[0], skip_special_tokens=True), hook_fire_count[0]
    finally:
        if hook_handle is not None:
            hook_handle.remove()

HEDGING = ["may", "might", "could", "possibly", "possible", "suggest",
           "suggestive", "consistent with", "cannot exclude", "appears",
           "likely", "probable", "uncertain"]

def hedge_count(text):
    text = text.lower()
    return sum(text.count(h) for h in HEDGING)

if __name__ == "__main__":
    tok, model = load_model()
    prompt = "The chest X-ray shows"

    # Finer alpha grid, focused middle layers
    for layer in [12, 16, 20]:
        print(f"\n{'='*60}\nLAYER {layer}\n{'='*60}")
        v = build_uncertainty_vector(tok, model, layer=layer)
        print(f"Vector norm: {v.norm().item():.2f}")

        for alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            out, fires = generate_with_steering(
                prompt, tok, model, v, alpha=alpha, layer_idx=layer
            )
            hedges = hedge_count(out)
            # Strip prompt for cleaner display
            generated = out.replace(prompt, "").strip()
            print(f"\n  α={alpha:>4}  hedges={hedges}  fires={fires}")
            print(f"  → {generated[:140]}")