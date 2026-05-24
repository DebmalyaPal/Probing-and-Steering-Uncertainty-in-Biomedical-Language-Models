"""
Model-agnostic activation-steering via PyTorch forward hooks.

Replaces steering_v8.py with a version that works for any decoder model
in the registry.  Encoder models cannot use this module for text generation;
see experiment_runner.py for encoder-specific representation-steering analysis.

Steering mechanics
------------------
At inference time a forward hook is registered on layer *layer_idx*.
For every forward pass the hook adds  α * hidden_norm * unit_vec  to the
layer's output hidden states.  α_frac controls the fraction of the layer's
typical activation magnitude used as the steering offset.

Seeds are identical to the original paper (do not modify).
"""

import random
import numpy as np
import torch

from src.model_registry import get_config, get_layers
from src.model_loader import get_device


# --------------------------------------------------------------------------- #
# Seed handling (preserved from original paper)                               #
# --------------------------------------------------------------------------- #

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Hidden-norm calibration                                                      #
# --------------------------------------------------------------------------- #

def calibrate_hidden_norms(tok, model, model_name: str,
                            sample_texts: list = None,
                            layers: list = None,
                            device: str = None) -> dict:
    """
    Estimate the mean L2 norm of hidden states at each layer.

    This replaces the hard-coded HIDDEN_NORMS dict from steering_v8.py with
    a value that is computed dynamically for any model.

    Parameters
    ----------
    sample_texts : A small list of sentences used for calibration.
                   Defaults to a set of generic biomedical sentences.
    layers       : Layer indices to calibrate.  Defaults to probe_layers
                   from the model registry.

    Returns dict: {layer: mean_norm}.
    """
    if device is None:
        device = get_device()
    cfg = get_config(model_name)
    if layers is None:
        layers = cfg["probe_layers"]
    if sample_texts is None:
        sample_texts = [
            "The patient presents with symptoms consistent with pneumonia.",
            "Findings suggest possible pleural effusion.",
            "The diagnosis is confirmed by laboratory results.",
            "There may be a small pericardial effusion present.",
            "The imaging demonstrates bilateral infiltrates.",
            "No acute cardiopulmonary process is identified.",
            "The etiology remains unclear at this time.",
            "Clinical assessment indicates a possible malignancy.",
        ]

    norms = {layer: [] for layer in layers}
    for text in sample_texts:
        inputs = tok(
            text, return_tensors="pt",
            truncation=True, max_length=128,
        ).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for layer in layers:
            hs = out.hidden_states[layer]   # (1, seq, H)
            # Per-token L2 norms, then mean
            token_norms = hs[0].norm(dim=-1).float()   # (seq,)
            norms[layer].append(token_norms.mean().item())

    return {layer: float(np.mean(vals)) for layer, vals in norms.items()}


# --------------------------------------------------------------------------- #
# Steered generation (decoder models only)                                    #
# --------------------------------------------------------------------------- #

def _probe_layer_to_hook_index(layer_idx: int) -> int:
    """
    Convert a probe-layer index (hidden_states[i] convention) to the
    0-based index into the model's layer module list.

    Indexing conventions:
      hidden_states[0]  → embedding output       (no hookable layer module)
      hidden_states[i]  → output of transformer block i   (i ≥ 1)
      layers_list[j]    → transformer block j+1  (0-based list index)

    So to intercept hidden_states[i] we hook layers_list[i-1].

    Layer 0 (embedding) cannot be intercepted via the transformer layer list;
    callers must ensure layer_idx >= 1.
    """
    if layer_idx < 1:
        raise ValueError(
            f"layer_idx={layer_idx} is the embedding output and cannot be "
            f"hooked via the transformer layer list.  Use layer_idx >= 1."
        )
    return layer_idx - 1


def generate_steered(prompts: list, tok, model, model_name: str,
                     vec_unit: torch.Tensor,
                     alpha_frac: float,
                     layer_idx: int,
                     hidden_norm: float,
                     device: str = None,
                     max_new_tokens: int = 60,
                     seed: int = 42) -> list:
    """
    Generate text from *prompts* with an activation-steering hook at *layer_idx*.

    Parameters
    ----------
    vec_unit    : Unit-norm steering direction (torch.Tensor, shape hidden_dim).
    alpha_frac  : Fraction of hidden_norm added to activations.
    layer_idx   : Probe-layer index (hidden_states[layer_idx] convention).
                  Must be >= 1.  The hook is registered on layers_list[layer_idx-1]
                  so it intercepts exactly the output that becomes hidden_states[layer_idx].
    hidden_norm : Estimated mean activation norm at *layer_idx*.
    seed        : Random seed.  Do NOT change — preserved from paper.

    Returns list of decoded strings (one per prompt, full sequence).
    """
    assert get_config(model_name)["model_type"] == "decoder", (
        f"generate_steered is only available for decoder models; "
        f"'{model_name}' is an encoder."
    )

    if device is None:
        device = get_device()

    v = vec_unit.to(device)
    effective_alpha = alpha_frac * hidden_norm

    def steer_hook(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
            return (hs + effective_alpha * v,) + output[1:]
        return output + effective_alpha * v

    layers_list = get_layers(model, model_name)
    hook_idx = _probe_layer_to_hook_index(layer_idx)
    handle = None
    if alpha_frac != 0.0:
        handle = layers_list[hook_idx].register_forward_hook(steer_hook)

    try:
        set_all_seeds(seed)
        tok.padding_side = "left"
        inputs = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=20,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.2,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
                return_dict_in_generate=False,
            )
        sequences = outputs if isinstance(outputs, torch.Tensor) else outputs.sequences
        return tok.batch_decode(sequences, skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()


# --------------------------------------------------------------------------- #
# Direction builder (consolidated from steering_v8.build_directions)          #
# --------------------------------------------------------------------------- #

def build_directions_from_data(uncertain: list, certain: list,
                                tok, model, layer: int,
                                device: str = None,
                                seed: int = 42):
    """
    Build raw probe and orthogonalized steering directions.

    Returns (v_probe, v_ortho) as torch.Tensor (float32).
    This mirrors build_directions() from steering_v8.py but calls the
    refactored orthogonalization module.
    """
    from src.orthogonalization import build_orthogonal_directions
    result = build_orthogonal_directions(
        uncertain, certain, tok, model, layer,
        device=device, seed=seed, verbose=False,
    )
    v_probe = torch.tensor(result["v_probe"], dtype=torch.float32)
    v_ortho = torch.tensor(result["v_ortho_length_hedge"], dtype=torch.float32)
    return v_probe, v_ortho


# --------------------------------------------------------------------------- #
# Encoder representation steering (no generation)                             #
# --------------------------------------------------------------------------- #

def steer_representation(text: str, tok, model, model_name: str,
                          vec_unit: torch.Tensor,
                          alpha_frac: float,
                          layer_idx: int,
                          hidden_norm: float,
                          device: str = None) -> np.ndarray:
    """
    Hook into *layer_idx*, inject the steering vector, and return the
    modified and original mean-pooled hidden states at that layer.

    Works for both decoder and encoder models.  Used for encoder
    representation-steering analysis.

    layer_idx : Probe-layer index (hidden_states[i] convention, must be >= 1).
                The hook is placed on layers_list[layer_idx-1] so the captured
                output is exactly hidden_states[layer_idx].

    Returns (modified_hidden_state, original_hidden_state) as numpy arrays.
    """
    if device is None:
        device = get_device()

    v = vec_unit.to(device)
    effective_alpha = alpha_frac * hidden_norm

    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        captured["original"] = hs.detach().clone()
        modified = hs + effective_alpha * v if alpha_frac != 0.0 else hs
        captured["modified"] = modified.detach().clone()
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    layers_list = get_layers(model, model_name)
    hook_idx = _probe_layer_to_hook_index(layer_idx)
    handle = layers_list[hook_idx].register_forward_hook(capture_hook)

    try:
        inputs = tok(
            text, return_tensors="pt",
            truncation=True, max_length=256,
        ).to(device)
        with torch.no_grad():
            model(**inputs, output_hidden_states=True)

        mask = inputs["attention_mask"][0].float().unsqueeze(-1)  # (seq, 1)
        orig = captured["original"][0]   # (seq, H)
        mod = captured["modified"][0]

        h_orig = ((orig * mask).sum(0) / mask.sum()).cpu().float().numpy()
        h_mod = ((mod * mask).sum(0) / mask.sum()).cpu().float().numpy()
    finally:
        handle.remove()

    return h_mod, h_orig
