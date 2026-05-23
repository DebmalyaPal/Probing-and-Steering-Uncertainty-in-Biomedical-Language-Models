"""
Unified experiment runner for both decoder and encoder models.

Decoder pipeline  (e.g. BioGPT, BioMistral):
  probing → orthogonalization → calibrate norms → steering sweep → metrics

Encoder pipeline  (e.g. BioBERT, ClinicalBERT):
  probing → orthogonalization → representation-steering analysis
  (no text generation; steer hidden states and measure probe projection shift)

Seeds are preserved exactly from the original paper.  Do not modify.
  data seed       = 42
  probe seed      = 42
  generation seeds = range(20)
"""

import json
import time
import platform
import numpy as np
import torch
import transformers
from datetime import datetime
from pathlib import Path

from src.model_registry import get_config
from src.model_loader import get_device
from src.probing import (
    run_probing_sweep, best_layer,
    build_probe_vectors_at_layers, extract_features,
)
from src.orthogonalization import build_orthogonal_directions
from src.steering_core import (
    calibrate_hidden_norms, generate_steered,
    steer_representation, set_all_seeds,
)
from src.metrics import compute_all_metrics, load_english_vocab


# Preserved from original paper
PROMPTS = [
    "Findings: the patient's chest imaging demonstrates",
    "Impression: based on the radiographic findings,",
    "On examination of the chest radiograph, there is",
    "The radiologist's interpretation of the chest X-ray:",
    "Clinical assessment of the thoracic imaging indicates",
]

ALPHAS        = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]
SEEDS         = list(range(20))   # DO NOT CHANGE — matches accepted paper
DATA_SEED     = 42
PROBE_SEED    = 42


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #

def _env_block(model_name: str, device: str) -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "hf_id": get_config(model_name)["hf_id"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
    }


def _aggregate(samples: list) -> dict:
    """Compute mean/std/median over a list of metric dicts."""
    agg = {}
    if not samples:
        return agg
    for key in samples[0].keys():
        vals = [
            s[key] for s in samples
            if s[key] is not None
            and not (isinstance(s[key], float) and (
                np.isinf(s[key]) or np.isnan(s[key])
            ))
        ]
        if vals:
            agg[f"{key}_mean"]   = float(np.mean(vals))
            agg[f"{key}_std"]    = float(np.std(vals, ddof=1))
            agg[f"{key}_median"] = float(np.median(vals))
        agg[f"{key}_n_valid"] = len(vals)
    return agg


# --------------------------------------------------------------------------- #
# Decoder experiment                                                           #
# --------------------------------------------------------------------------- #

def run_decoder_experiment(model_name: str, tok, model,
                            uncertain: list, certain: list,
                            device: str = None,
                            steer_layer: int = None,
                            alphas: list = None,
                            seeds: list = None,
                            verbose: bool = True) -> dict:
    """
    Full probe + steer experiment for a causal-LM model.

    Returns a results dict that mirrors the structure of
    results/final_experiment.json from the original paper.
    """
    cfg = get_config(model_name)
    if device is None:
        device = get_device()
    if alphas is None:
        alphas = ALPHAS
    if seeds is None:
        seeds = SEEDS

    # 1. Probing sweep
    if verbose:
        print(f"\n{'='*60}\nProbing sweep — {cfg['display_name']}\n{'='*60}")
    probe_results = run_probing_sweep(
        uncertain, certain, tok, model,
        layers=cfg["probe_layers"], device=device, verbose=verbose,
    )

    # 2. Save per-layer probe accuracy as a standalone checkpoint
    from src.results_manager import save_results as _save
    _save(model_name, "probe_accuracy_per_layer", {
        "n_uncertain": len(uncertain),
        "n_certain": len(certain),
        "n_splits": 5,
        "results": {str(k): v for k, v in probe_results.items()},
    })

    # 3. Select best layer for steering.
    #    Layer 0 = embedding output — it cannot be intercepted via the
    #    transformer layer list, so fall back to the best non-embedding layer.
    layer = steer_layer if steer_layer is not None else best_layer(probe_results)
    if layer == 0:
        layer = max(
            (l for l in probe_results if l >= 1),
            key=lambda l: probe_results[l]["mean_acc"],
        )
        if verbose:
            print(f"  (layer 0 is embedding-only; selected layer {layer} instead)")
    if verbose:
        print(f"\nSelected steering layer: {layer} "
              f"(acc={probe_results[layer]['mean_acc']:.2%})")

    # 4. Orthogonalization
    if verbose:
        print(f"\nBuilding probe & orthogonal directions at layer {layer}...")
    ortho = build_orthogonal_directions(
        uncertain, certain, tok, model, layer,
        device=device, seed=PROBE_SEED, verbose=verbose,
    )
    v_probe = torch.tensor(ortho["v_probe"], dtype=torch.float32)
    v_ortho = torch.tensor(ortho["v_ortho_length_hedge"], dtype=torch.float32)

    # 4. Calibrate hidden norms
    if verbose:
        print("\nCalibrating hidden norms...")
    hidden_norms = calibrate_hidden_norms(
        tok, model, model_name,
        layers=[layer], device=device,
    )
    hidden_norm = hidden_norms[layer]
    if verbose:
        print(f"  Layer {layer} norm estimate: {hidden_norm:.1f}")

    # 5. Load English vocab for metrics
    english_vocab = load_english_vocab()

    # 6. Steering sweep
    direction_vectors = {
        "probe": v_probe / v_probe.norm(),
        "ortho": v_ortho / v_ortho.norm(),
    }
    all_records = []
    summary = {}
    total = len(alphas) * len(seeds) * len(PROMPTS) * len(direction_vectors)
    if verbose:
        print(f"\nSteering sweep — {total} generations total")

    for dir_name, vec_unit in direction_vectors.items():
        summary[dir_name] = {}
        for af in alphas:
            t0 = time.time()
            samples_this = []

            for seed in seeds:
                outs = generate_steered(
                    PROMPTS, tok, model, model_name, vec_unit,
                    alpha_frac=af, layer_idx=layer,
                    hidden_norm=hidden_norm, device=device,
                    seed=seed,
                )
                for prompt, full_out in zip(PROMPTS, outs):
                    gen = full_out.replace(prompt, "").strip()
                    m = compute_all_metrics(
                        gen, tok, model,
                        model_type="decoder",
                        device=device,
                        english_vocab=english_vocab,
                    )
                    record = {
                        "direction": dir_name,
                        "alpha_frac": af,
                        "layer": layer,
                        "seed": seed,
                        "prompt": prompt,
                        "generation": gen,
                        **m,
                    }
                    all_records.append(record)
                    samples_this.append(m)

            agg = _aggregate(samples_this)
            summary[dir_name][str(af)] = agg

            if verbose:
                elapsed = time.time() - t0
                print(
                    f"  {dir_name:>6}  α={af:.3f}  "
                    f"hedge={agg.get('hedge_score_mean', 0):.3f}  "
                    f"ppl={agg.get('perplexity_mean', 0):.1f}  "
                    f"[{elapsed:.0f}s]"
                )

    return {
        "config": {
            "model_name": model_name,
            "layer": layer,
            "alphas": alphas,
            "seeds": seeds,
            "prompts": PROMPTS,
            "directions": list(direction_vectors.keys()),
            "hidden_norm_estimate": hidden_norm,
            "n_contrast_uncertain": len(uncertain),
            "n_contrast_certain": len(certain),
        },
        "environment": _env_block(model_name, device),
        "probe_results": {str(k): v for k, v in probe_results.items()},
        "orthogonalization": ortho,
        "summary": summary,
        "records": all_records,
    }


# --------------------------------------------------------------------------- #
# Encoder experiment                                                           #
# --------------------------------------------------------------------------- #

def run_encoder_experiment(model_name: str, tok, model,
                            uncertain: list, certain: list,
                            device: str = None,
                            verbose: bool = True) -> dict:
    """
    Probe + representation-steering analysis for an encoder (masked LM) model.

    Steering metrics (per sentence, per alpha, per direction):
      delta_proj        : shift along the probe axis (was: only metric)
      prob_delta        : change in P(uncertain) from the fitted probe classifier
      flipped           : whether the probe prediction crossed the decision boundary
      cos_delta_centroid: change in cosine similarity to the uncertain centroid
      frac_on_axis      : fraction of the total hidden-state shift that lands on
                          the probe direction (1.0 = perfect, <1.0 = leakage)
    """
    from sklearn.linear_model import LogisticRegression
    from src.orthogonalization import cosine_similarity as _cos

    cfg = get_config(model_name)
    if device is None:
        device = get_device()

    # 1. Probing sweep
    if verbose:
        print(f"\n{'='*60}\nProbing sweep — {cfg['display_name']}\n{'='*60}")
    probe_results = run_probing_sweep(
        uncertain, certain, tok, model,
        layers=cfg["probe_layers"], device=device, verbose=verbose,
    )

    # 2. Save per-layer probe accuracy as a standalone checkpoint
    from src.results_manager import save_results as _save
    _save(model_name, "probe_accuracy_per_layer", {
        "n_uncertain": len(uncertain),
        "n_certain": len(certain),
        "n_splits": 5,
        "results": {str(k): v for k, v in probe_results.items()},
    })

    # 3. Best layer (must be >= 1; layer 0 = embedding, cannot be hooked)
    layer = best_layer(probe_results)
    if layer == 0:
        layer = max(
            (l for l in probe_results if l >= 1),
            key=lambda l: probe_results[l]["mean_acc"],
        )
        if verbose:
            print(f"  (layer 0 is embedding-only; selected layer {layer} instead)")
    if verbose:
        print(f"\nBest layer: {layer} "
              f"(acc={probe_results[layer]['mean_acc']:.2%})")

    # 4. Orthogonalization
    if verbose:
        print(f"\nBuilding probe & orthogonal directions at layer {layer}...")
    ortho = build_orthogonal_directions(
        uncertain, certain, tok, model, layer,
        device=device, seed=PROBE_SEED, verbose=verbose,
    )
    v_probe    = torch.tensor(ortho["v_probe"], dtype=torch.float32)
    v_ortho    = torch.tensor(ortho["v_ortho_length_hedge"], dtype=torch.float32)
    probe_unit = (v_probe / v_probe.norm()).numpy()   # unit vector

    # 5. Calibrate hidden norms
    hidden_norms = calibrate_hidden_norms(
        tok, model, model_name, layers=[layer], device=device,
    )
    hidden_norm = hidden_norms[layer]

    # 6. Extract full-dataset features at best layer for:
    #    (a) fitting the probe classifier for probability scoring
    #    (b) computing uncertain / certain centroids
    if verbose:
        print(f"\nExtracting features for probe classifier & centroids...")
    all_sentences = uncertain + certain
    y_all = np.array([1] * len(uncertain) + [0] * len(certain))
    X_all = extract_features(all_sentences, tok, model, layer, device)

    probe_clf = LogisticRegression(
        max_iter=2000, C=0.1, random_state=PROBE_SEED
    ).fit(X_all, y_all)

    centroid_unc = X_all[y_all == 1].mean(axis=0)   # mean uncertain hidden state
    centroid_cer = X_all[y_all == 0].mean(axis=0)   # mean certain hidden state
    centroid_unc_unit = centroid_unc / np.linalg.norm(centroid_unc)
    centroid_cer_unit = centroid_cer / np.linalg.norm(centroid_cer)

    # 7. Representation-steering analysis
    # Use the latter 40+40 sentences (not used to fit the probe) as test set
    test_sentences = uncertain[160:] + certain[160:]   # held-out 40+40
    test_labels    = [1] * len(uncertain[160:]) + [0] * len(certain[160:])

    direction_vectors = {
        "probe": v_probe / v_probe.norm(),
        "ortho": v_ortho / v_ortho.norm(),
    }

    steer_records = []
    summary = {}

    if verbose:
        print(f"\nRepresentation steering sweep (layer {layer})...")
        print(f"  {'dir':>6}  {'α':>6}  {'Δproj':>8}  {'Δprob':>8}  "
              f"{'flip%':>7}  {'Δcos_unc':>10}  {'on-axis':>8}")

    for dir_name, vec_unit in direction_vectors.items():
        vec_unit_np = vec_unit.numpy()
        summary[dir_name] = {}

        for af in ALPHAS:
            (deltas, prob_deltas, flips, flips_cer2unc, flips_unc2cer,
             cos_deltas, frac_on_axis_vals) = [], [], [], [], [], [], []

            for text, label in zip(test_sentences, test_labels):
                h_mod, h_orig = steer_representation(
                    text, tok, model, model_name, vec_unit,
                    alpha_frac=af, layer_idx=layer,
                    hidden_norm=hidden_norm, device=device,
                )

                # ── probe projection delta ──────────────────────────────
                orig_proj = float(h_orig @ probe_unit)
                mod_proj  = float(h_mod  @ probe_unit)
                delta     = mod_proj - orig_proj
                deltas.append(delta)

                # ── probe classifier probability ────────────────────────
                p_orig = float(probe_clf.predict_proba([h_orig])[0][1])
                p_mod  = float(probe_clf.predict_proba([h_mod])[0][1])
                prob_deltas.append(p_mod - p_orig)

                pred_orig = int(p_orig >= 0.5)
                pred_mod  = int(p_mod  >= 0.5)
                flipped   = pred_orig != pred_mod
                flips.append(int(flipped))
                if label == 0:
                    flips_cer2unc.append(int(flipped and pred_mod == 1))
                else:
                    flips_unc2cer.append(int(flipped and pred_mod == 0))

                # ── cosine similarity to uncertain centroid ─────────────
                cos_orig = float(
                    (h_orig / (np.linalg.norm(h_orig) + 1e-9)) @ centroid_unc_unit
                )
                cos_mod  = float(
                    (h_mod  / (np.linalg.norm(h_mod)  + 1e-9)) @ centroid_unc_unit
                )
                cos_deltas.append(cos_mod - cos_orig)

                # ── on-axis fraction (how much shift is on probe axis?) ─
                # delta_h = h_mod - h_orig = alpha * hidden_norm * vec_unit
                # component along probe = (delta_h · probe_unit) * probe_unit
                # leakage = ||delta_h - probe_component||
                delta_h      = h_mod - h_orig
                delta_h_norm = float(np.linalg.norm(delta_h))
                if delta_h_norm > 1e-9:
                    proj_scalar   = float(delta_h @ probe_unit)
                    leakage_norm  = float(
                        np.linalg.norm(delta_h - proj_scalar * probe_unit)
                    )
                    frac_on_axis  = abs(proj_scalar) / delta_h_norm
                else:
                    leakage_norm = 0.0
                    frac_on_axis = 1.0
                frac_on_axis_vals.append(frac_on_axis)

                steer_records.append({
                    "direction": dir_name, "alpha_frac": af,
                    "layer": layer, "label": label,
                    "orig_proj": orig_proj, "mod_proj": mod_proj,
                    "delta_proj": delta,
                    "prob_uncertain_orig": p_orig,
                    "prob_uncertain_mod":  p_mod,
                    "prob_delta": p_mod - p_orig,
                    "pred_orig": pred_orig, "pred_mod": pred_mod,
                    "flipped": flipped,
                    "cos_uncertain_centroid_orig": cos_orig,
                    "cos_uncertain_centroid_mod":  cos_mod,
                    "cos_delta_centroid": cos_mod - cos_orig,
                    "frac_shift_on_probe_axis": frac_on_axis,
                    "orthogonal_leakage_norm": leakage_norm,
                })

            flip_rate         = float(np.mean(flips))
            flip_rate_c2u     = float(np.mean(flips_cer2unc)) if flips_cer2unc else 0.0
            flip_rate_u2c     = float(np.mean(flips_unc2cer)) if flips_unc2cer else 0.0
            mean_frac_on_axis = float(np.mean(frac_on_axis_vals))

            summary[dir_name][str(af)] = {
                "delta_proj_mean":          float(np.mean(deltas)),
                "delta_proj_std":           float(np.std(deltas, ddof=1)),
                "delta_proj_median":        float(np.median(deltas)),
                "prob_delta_mean":          float(np.mean(prob_deltas)),
                "prob_delta_std":           float(np.std(prob_deltas, ddof=1)),
                "flip_rate":                flip_rate,
                "flip_rate_certain_to_uncertain": flip_rate_c2u,
                "flip_rate_uncertain_to_certain": flip_rate_u2c,
                "cos_delta_uncertain_centroid_mean": float(np.mean(cos_deltas)),
                "cos_delta_uncertain_centroid_std":  float(np.std(cos_deltas, ddof=1)),
                "frac_shift_on_probe_axis_mean": mean_frac_on_axis,
            }

            if verbose:
                print(
                    f"  {dir_name:>6}  {af:.3f}  "
                    f"{np.mean(deltas):>+8.3f}  "
                    f"{np.mean(prob_deltas):>+8.3f}  "
                    f"{flip_rate:>6.1%}  "
                    f"{np.mean(cos_deltas):>+10.4f}  "
                    f"{mean_frac_on_axis:>7.3f}"
                )

    return {
        "config": {
            "model_name": model_name,
            "layer": layer,
            "alphas": ALPHAS,
            "n_contrast_uncertain": len(uncertain),
            "n_contrast_certain": len(certain),
            "n_steer_test": len(test_sentences),
        },
        "environment": _env_block(model_name, device),
        "probe_results": {str(k): v for k, v in probe_results.items()},
        "orthogonalization": ortho,
        "summary": summary,
        "steer_records": steer_records,
    }


# --------------------------------------------------------------------------- #
# Unified entry point                                                          #
# --------------------------------------------------------------------------- #

def run_experiment(model_name: str, tok, model,
                   uncertain: list, certain: list,
                   device: str = None,
                   **kwargs) -> dict:
    """
    Dispatch to the correct experiment based on model type.
    Notebooks should call this function.
    """
    model_type = get_config(model_name)["model_type"]
    if model_type == "decoder":
        return run_decoder_experiment(
            model_name, tok, model, uncertain, certain,
            device=device, **kwargs,
        )
    else:
        return run_encoder_experiment(
            model_name, tok, model, uncertain, certain,
            device=device, **kwargs,
        )
