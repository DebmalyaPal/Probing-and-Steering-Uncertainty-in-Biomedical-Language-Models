import torch
import numpy as np
from src.hidden_states import load_model, get_last_token_hidden_state
from src.steering import UNCERTAIN, CERTAIN, build_uncertainty_vector

tok, model = load_model()

# Held-out test sentences the vector has NEVER seen
TEST_UNCERTAIN = [
    "The lesion may represent a benign process.",
    "Findings are suggestive of pulmonary edema.",
    "Possible early changes of emphysema.",
    "Cannot rule out small pleural effusion.",
]
TEST_CERTAIN = [
    "The lesion is a benign cyst.",
    "Findings show pulmonary edema.",
    "Early changes of emphysema are present.",
    "There is a small pleural effusion.",
]

for layer in [8, 12, 16, 20, 23]:
    v = build_uncertainty_vector(tok, model, layer=layer)
    v_norm = v / v.norm()

    # Project held-out sentences onto the vector
    u_proj = [get_last_token_hidden_state(s, tok, model, layer=layer) @ v_norm
              for s in TEST_UNCERTAIN]
    c_proj = [get_last_token_hidden_state(s, tok, model, layer=layer) @ v_norm
              for s in TEST_CERTAIN]

    u_mean = np.mean([p.item() for p in u_proj])
    c_mean = np.mean([p.item() for p in c_proj])
    separation = u_mean - c_mean

    print(f"Layer {layer:2d}: uncertain_proj={u_mean:+.3f}  "
          f"certain_proj={c_mean:+.3f}  "
          f"separation={separation:+.3f}")