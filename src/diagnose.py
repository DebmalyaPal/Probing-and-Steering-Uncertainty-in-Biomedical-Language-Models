import torch
from src.hidden_states import load_model
from src.steering import build_uncertainty_vector, UNCERTAIN, CERTAIN
from src.hidden_states import get_last_token_hidden_state

device = "mps"
tok, model = load_model(device)

# 1. What's the actual module structure?
print("=== Model structure (top level) ===")
for name, _ in model.named_children():
    print(f"  {name}")

print("\n=== BioGPT submodule structure ===")
# Try common attribute paths
for path in ["biogpt", "model", "transformer"]:
    if hasattr(model, path):
        submod = getattr(model, path)
        print(f"Found: model.{path}")
        for name, _ in submod.named_children():
            print(f"    {name}")
        break

# 2. How many layers does BioGPT have?
# Adjust this based on what you found above
try:
    layers = model.biogpt.layers
    print(f"\nNumber of layers: {len(layers)}")
except AttributeError:
    print("\nCould not find .biogpt.layers — check structure above")

# 3. Hidden state magnitudes
print("\n=== Hidden state magnitudes across layers ===")
text = "The patient may have pneumonia."
inputs = tok(text, return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

for i, h in enumerate(out.hidden_states):
    last_tok = h[0, -1, :]
    print(f"  Layer {i:2d}: norm={last_tok.norm().item():.2f}")

# 4. Uncertainty vector at different layers
print("\n=== Uncertainty vector norms by layer ===")
num_layers = len(out.hidden_states)
for layer_idx in range(num_layers):
    u = torch.stack([
        get_last_token_hidden_state(s, tok, model, layer=layer_idx, device=device)
        for s in UNCERTAIN
    ])
    c = torch.stack([
        get_last_token_hidden_state(s, tok, model, layer=layer_idx, device=device)
        for s in CERTAIN
    ])
    raw_diff = u.mean(0) - c.mean(0)
    print(f"  Layer {layer_idx:2d}: raw diff norm={raw_diff.norm().item():.3f}")