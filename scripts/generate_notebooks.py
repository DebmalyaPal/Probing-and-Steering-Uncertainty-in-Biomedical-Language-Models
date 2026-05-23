"""
Generate one Jupyter notebook per model.

Run from the project root:
    python scripts/generate_notebooks.py

Each notebook is self-contained and calls into the refactored src/ modules.
Results are saved to results/{model_name}/ for cross-model comparison.
"""

import json
from pathlib import Path

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
except ImportError:
    raise SystemExit("Install nbformat first:  pip install nbformat")


# --------------------------------------------------------------------------- #
# Model metadata used to customise each notebook                              #
# --------------------------------------------------------------------------- #

MODELS = [
    # (registry_key, display_name, model_type, notes)
    ("biogpt",       "BioGPT (original)",       "decoder",
     "Baseline from the original paper — 24 layers, 1024-dim."),
    ("biogpt_large", "BioGPT-Large",             "decoder",
     "48-layer, 1600-dim variant. Same architecture family as BioGPT."),
    ("biomedlm",     "BioMedLM (PubMedGPT)",     "decoder",
     "2.7 B GPT-2-style model trained on PubMed.  Hidden dim 2560."),
    ("biomistral",   "BioMistral-7B",             "decoder",
     "Mistral-7B fine-tuned on biomedical corpora. Loaded in 8-bit."),
    ("meditron",     "Meditron-7B",               "decoder",
     "LLaMA-2 7B fine-tuned on medical guidelines + PubMed. 8-bit."),
    ("biobert",      "BioBERT",                   "encoder",
     "BERT base trained on PubMed abstracts + PMC full-text."),
    ("clinicalbert", "ClinicalBERT",              "encoder",
     "BERT base trained on MIMIC-III clinical notes."),
    ("bluebert",     "BlueBERT",                  "encoder",
     "BERT base trained on PubMed + MIMIC-III."),
    ("scibert",      "SciBERT",                   "encoder",
     "BERT base trained on broad scientific literature (AllenAI)."),
]


# --------------------------------------------------------------------------- #
# Cell builders                                                                #
# --------------------------------------------------------------------------- #

def md(src: str):
    return new_markdown_cell(src)


def code(src: str):
    return new_code_cell(src)


# --------------------------------------------------------------------------- #
# Shared cell blocks                                                           #
# --------------------------------------------------------------------------- #

SETUP_IMPORTS = """\
import sys, json
from pathlib import Path

# ── ensure project root is on the path ──────────────────────────────────────
ROOT = Path().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.model_registry  import get_config, list_models
from src.model_loader    import load_model, get_device
from src.bioscope_parser import build_balanced_contrast_set
from src.probing         import run_probing_sweep, best_layer
from src.orthogonalization import build_orthogonal_directions
from src.steering_core   import calibrate_hidden_norms
from src.experiment_runner import run_experiment, PROMPTS, ALPHAS, SEEDS
from src.results_manager import save_results, load_results, results_exist
from src.metrics         import load_english_vocab

print(f"PyTorch   : {torch.__version__}")
print(f"Device    : {get_device()}")
"""

LOAD_DATA = """\
# ── BioScope contrast set ────────────────────────────────────────────────────
# seed=42 is fixed from the original paper — do not change
uncertain, certain = build_balanced_contrast_set(
    "data/bioscope", max_per_class=200, seed=42
)
print(f"Loaded {len(uncertain)} uncertain + {len(certain)} certain sentences")
print("\\nUncertain sample:")
for s in uncertain[:2]:
    print(f"  {s[:100]}")
print("\\nCertain sample:")
for s in certain[:2]:
    print(f"  {s[:100]}")
"""

SAVE_AND_SHOW = """\
# ── save experiment results ───────────────────────────────────────────────────
save_results(MODEL_NAME, "experiment", results)
print("\\nSaved files:", save_results.__module__)
"""


def probe_accuracy_plot(model_type: str) -> str:
    return """\
# ── probe accuracy plot ───────────────────────────────────────────────────────
probe_res = results["probe_results"]
layers_sorted = sorted(int(k) for k in probe_res.keys())
means  = [probe_res[str(l)]["mean_acc"] for l in layers_sorted]
cis_lo = [probe_res[str(l)]["ci_low"]   for l in layers_sorted]
cis_hi = [probe_res[str(l)]["ci_high"]  for l in layers_sorted]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(layers_sorted, means, marker="o", linewidth=2, label="CV accuracy")
ax.fill_between(layers_sorted, cis_lo, cis_hi, alpha=0.2, label="95% CI")
ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Chance")
ax.set_xlabel("Layer")
ax.set_ylabel("Accuracy")
ax.set_title(f"Probe accuracy per layer — {DISPLAY_NAME}")
ax.legend()
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.tight_layout()

fig_path = Path("results") / MODEL_NAME / "fig_probe_accuracy.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
plt.show()
print(f"Saved → {fig_path}")
"""


def steering_plot_decoder() -> str:
    return """\
# ── steering trade-off plot (decoder) ────────────────────────────────────────
summary = results["summary"]
directions = list(summary.keys())
metrics_to_plot = ["hedge_score", "perplexity", "lexical_diversity", "token_validity"]
metric_labels   = ["Hedge score", "Perplexity", "Lexical diversity", "Token validity"]

fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 4))
colors = {"probe": "#2196F3", "ortho": "#FF5722"}
markers = {"probe": "o", "ortho": "s"}

for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
    for dir_name in directions:
        dir_data = summary[dir_name]
        alphas_sorted = sorted(float(a) for a in dir_data.keys())
        y_means = [dir_data[str(a)].get(f"{metric}_mean", np.nan) for a in alphas_sorted]
        y_stds  = [dir_data[str(a)].get(f"{metric}_std",  0.0)    for a in alphas_sorted]
        ax.plot(alphas_sorted, y_means,
                marker=markers[dir_name], color=colors[dir_name],
                label=dir_name, linewidth=2)
        ax.fill_between(
            alphas_sorted,
            [m - s for m, s in zip(y_means, y_stds)],
            [m + s for m, s in zip(y_means, y_stds)],
            alpha=0.15, color=colors[dir_name],
        )
    ax.set_xlabel("α fraction")
    ax.set_title(label)
    ax.legend(fontsize=8)

plt.suptitle(f"Steering trade-offs — {DISPLAY_NAME}", fontsize=12)
plt.tight_layout()

fig_path = Path("results") / MODEL_NAME / "fig_steering_tradeoff.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
plt.show()
print(f"Saved → {fig_path}")
"""


def steering_plot_encoder() -> str:
    return """\
# ── representation steering plot (encoder) ───────────────────────────────────
summary = results["summary"]
directions = list(summary.keys())

fig, ax = plt.subplots(figsize=(7, 4))
colors  = {"probe": "#2196F3", "ortho": "#FF5722"}
markers = {"probe": "o",       "ortho": "s"}

for dir_name in directions:
    dir_data = summary[dir_name]
    alphas_sorted = sorted(float(a) for a in dir_data.keys())
    deltas = [dir_data[str(a)]["delta_mean"] for a in alphas_sorted]
    stds   = [dir_data[str(a)]["delta_std"]  for a in alphas_sorted]
    ax.plot(alphas_sorted, deltas,
            marker=markers[dir_name], color=colors[dir_name],
            label=dir_name, linewidth=2)
    ax.fill_between(
        alphas_sorted,
        [d - s for d, s in zip(deltas, stds)],
        [d + s for d, s in zip(deltas, stds)],
        alpha=0.15, color=colors[dir_name],
    )

ax.axhline(0, color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("α fraction")
ax.set_ylabel("Mean Δ probe projection")
ax.set_title(f"Representation steering — {DISPLAY_NAME}")
ax.legend()
plt.tight_layout()

fig_path = Path("results") / MODEL_NAME / "fig_representation_steering.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
plt.show()
print(f"Saved → {fig_path}")
"""


# --------------------------------------------------------------------------- #
# Notebook assembly                                                            #
# --------------------------------------------------------------------------- #

def build_decoder_notebook(key: str, display: str, notes: str) -> nbformat.NotebookNode:
    cells = [
        md(f"# {display}\n\n{notes}\n\n"
           f"**Model type:** Decoder (causal LM)  \n"
           f"**Pipeline:** Probing → Orthogonalization → Steering sweep  \n"
           f"**Results saved to:** `results/{key}/`"),

        md("## 1. Setup"),
        code(SETUP_IMPORTS),
        code(f'MODEL_NAME   = "{key}"\n'
             f'DISPLAY_NAME = "{display}"\n'
             f'cfg          = get_config(MODEL_NAME)\n'
             f'DEVICE       = get_device()\n'
             f'print(cfg)'),

        md("## 2. Load model"),
        code('tok, model = load_model(MODEL_NAME, device=DEVICE)\n'
             'print(f"Loaded: {cfg[\'hf_id\']}")'),

        md("## 3. Load BioScope data"),
        code(LOAD_DATA),

        md("## 4. Probing sweep — all layers\n\n"
           "Train a logistic regression at each layer and measure 5-fold CV accuracy."),
        code('from src.probing import run_probing_sweep, best_layer\n\n'
             'probe_results = run_probing_sweep(\n'
             '    uncertain, certain, tok, model,\n'
             '    layers=cfg["probe_layers"], device=DEVICE,\n'
             ')'),

        md("### Best layer"),
        code('selected_layer = cfg["steer_layer"] or best_layer(probe_results)\n'
             'print(f"Selected layer: {selected_layer}  "  \n'
             '      f"acc={probe_results[selected_layer][\'mean_acc\']:.2%}")'),

        md("## 5. Orthogonalization\n\n"
           "Remove length and lexical-hedge confounds from the probe direction."),
        code('from src.orthogonalization import build_orthogonal_directions\n\n'
             'ortho = build_orthogonal_directions(\n'
             '    uncertain, certain, tok, model, selected_layer,\n'
             '    device=DEVICE,\n'
             ')\n'
             'print(f"  cos(probe, length) = {ortho[\'cos_probe_length\']:.4f}")\n'
             'print(f"  cos(probe, hedge)  = {ortho[\'cos_probe_hedge\']:.4f}")\n'
             'print(f"  ortho_LH acc       = {ortho[\'ortho_LH_classification_acc\']:.2%}")'),

        md("## 6. Calibrate hidden norms"),
        code('hidden_norms = calibrate_hidden_norms(\n'
             '    tok, model, MODEL_NAME, layers=[selected_layer], device=DEVICE,\n'
             ')\n'
             'print(f"Hidden norm at layer {selected_layer}: {hidden_norms[selected_layer]:.1f}")'),

        md("## 7. Full steering experiment\n\n"
           "Sweeps 9 α values × 20 seeds × 5 prompts × 2 directions = 1 800 generations.\n\n"
           "> **Note:** This cell is compute-intensive. "
           "Skip and load from disk if already run (next cell checks)."),
        code('if results_exist(MODEL_NAME, "experiment"):\n'
             '    print("Loading cached results...")\n'
             '    results = load_results(MODEL_NAME, "experiment")\n'
             'else:\n'
             '    results = run_experiment(\n'
             '        MODEL_NAME, tok, model, uncertain, certain,\n'
             '        device=DEVICE,\n'
             '        steer_layer=selected_layer,\n'
             '    )\n'
             '    save_results(MODEL_NAME, "experiment", results)\n'
             'print(f"Records: {len(results[\'records\'])}")'),

        md("## 8. Visualizations"),
        code(probe_accuracy_plot("decoder")),
        code(steering_plot_decoder()),

        md("## 9. Sample steered outputs"),
        code('import random\n'
             'recs = [r for r in results["records"] if r["alpha_frac"] == 0.15\n'
             '        and r["direction"] == "probe"]\n'
             'for r in random.sample(recs, min(3, len(recs))):\n'
             '    print(f"[α=0.15 probe | seed={r[\'seed\']}]")\n'
             '    print(f"  {r[\'generation\'][:200]}")\n'
             '    print(f"  hedge={r[\'hedge_score\']:.2f}  ppl={r.get(\'perplexity\',\'N/A\')}")\n'
             '    print()'),

        md("## 10. Summary statistics"),
        code(
            'summary = results["summary"]\n'
            'header = "{:>8} {:>14} {:>14} {:>12} {:>12}".format("", "probe hedge", "ortho hedge", "probe ppl", "ortho ppl")\n'
            'print(header)\n'
            'for af in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:\n'
            '    ph = summary.get("probe", {}).get(str(af), {}).get("hedge_score_mean", float("nan"))\n'
            '    oh = summary.get("ortho", {}).get(str(af), {}).get("hedge_score_mean", float("nan"))\n'
            '    pp = summary.get("probe", {}).get(str(af), {}).get("perplexity_mean", float("nan"))\n'
            '    op = summary.get("ortho", {}).get(str(af), {}).get("perplexity_mean", float("nan"))\n'
            '    print(f"  alpha={af:.3f}  {ph:>14.3f} {oh:>14.3f} {pp:>12.1f} {op:>12.1f}")\n'
        ),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11.0"}
    return nb


def build_encoder_notebook(key: str, display: str, notes: str) -> nbformat.NotebookNode:
    cells = [
        md(f"# {display}\n\n{notes}\n\n"
           f"**Model type:** Encoder (masked LM)  \n"
           f"**Pipeline:** Probing → Orthogonalization → Representation steering  \n"
           f"**Note:** Text generation is not available for encoder models. "
           f"Steering is evaluated by measuring how injecting the probe direction "
           f"shifts hidden-state projections onto the uncertainty axis.  \n"
           f"**Results saved to:** `results/{key}/`"),

        md("## 1. Setup"),
        code(SETUP_IMPORTS),
        code(f'MODEL_NAME   = "{key}"\n'
             f'DISPLAY_NAME = "{display}"\n'
             f'cfg          = get_config(MODEL_NAME)\n'
             f'DEVICE       = get_device()\n'
             f'print(cfg)'),

        md("## 2. Load model"),
        code('tok, model = load_model(MODEL_NAME, device=DEVICE)\n'
             'print(f"Loaded: {cfg[\'hf_id\']}")'),

        md("## 3. Load BioScope data"),
        code(LOAD_DATA),

        md("## 4. Probing sweep — all layers"),
        code('probe_results = run_probing_sweep(\n'
             '    uncertain, certain, tok, model,\n'
             '    layers=cfg["probe_layers"], device=DEVICE,\n'
             ')'),

        md("### Best layer"),
        code('selected_layer = best_layer(probe_results)\n'
             'print(f"Selected layer: {selected_layer}  "  \n'
             '      f"acc={probe_results[selected_layer][\'mean_acc\']:.2%}")'),

        md("## 5. Orthogonalization"),
        code('ortho = build_orthogonal_directions(\n'
             '    uncertain, certain, tok, model, selected_layer,\n'
             '    device=DEVICE,\n'
             ')\n'
             'print(f"  cos(probe, length) = {ortho[\'cos_probe_length\']:.4f}")\n'
             'print(f"  cos(probe, hedge)  = {ortho[\'cos_probe_hedge\']:.4f}")\n'
             'print(f"  ortho_LH acc       = {ortho[\'ortho_LH_classification_acc\']:.2%}")'),

        md("## 6. Calibrate hidden norms"),
        code('hidden_norms = calibrate_hidden_norms(\n'
             '    tok, model, MODEL_NAME, layers=[selected_layer], device=DEVICE,\n'
             ')\n'
             'print(f"Hidden norm at layer {selected_layer}: "  \n'
             '      f"{hidden_norms[selected_layer]:.1f}")'),

        md("## 7. Representation-steering experiment\n\n"
           "For each α, inject the steering vector at the selected layer and measure\n"
           "the mean shift in probe-direction projection across 80 balanced test sentences.\n\n"
           "A positive Δ means the representation moves toward the 'uncertain' pole."),
        code('if results_exist(MODEL_NAME, "experiment"):\n'
             '    print("Loading cached results...")\n'
             '    results = load_results(MODEL_NAME, "experiment")\n'
             'else:\n'
             '    results = run_experiment(\n'
             '        MODEL_NAME, tok, model, uncertain, certain,\n'
             '        device=DEVICE,\n'
             '    )\n'
             '    save_results(MODEL_NAME, "experiment", results)'),

        md("## 8. Visualizations"),
        code(probe_accuracy_plot("encoder")),
        code(steering_plot_encoder()),

        md("## 9. Summary statistics"),
        code(
            'summary = results["summary"]\n'
            'header = "{:>8} {:>14} {:>14}".format("", "probe Dproj", "ortho Dproj")\n'
            'print(header)\n'
            'for af in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:\n'
            '    pd_ = summary.get("probe", {}).get(str(af), {}).get("delta_mean", float("nan"))\n'
            '    od_ = summary.get("ortho", {}).get(str(af), {}).get("delta_mean", float("nan"))\n'
            '    print(f"  alpha={af:.3f}  {pd_:>14.4f} {od_:>14.4f}")\n'
        ),

        md("## 10. CLS-token projection distribution\n\n"
           "Visualise where uncertain vs. certain sentences fall on the probe axis "
           "before and after steering."),
        code('import numpy as np\n'
             'from src.probing import extract_features\n\n'
             'v_probe = np.array(ortho["v_probe"])\n'
             'v_unit  = v_probe / np.linalg.norm(v_probe)\n'
             '\n'
             'X_u = extract_features(uncertain[:50], tok, model, selected_layer, DEVICE)\n'
             'X_c = extract_features(certain[:50],   tok, model, selected_layer, DEVICE)\n'
             '\n'
             'proj_u = X_u @ v_unit\n'
             'proj_c = X_c @ v_unit\n'
             '\n'
             'fig, ax = plt.subplots(figsize=(7, 4))\n'
             'ax.hist(proj_u, bins=20, alpha=0.6, label="Uncertain", color="#E53935")\n'
             'ax.hist(proj_c, bins=20, alpha=0.6, label="Certain",   color="#1E88E5")\n'
             'ax.set_xlabel("Probe projection")\n'
             'ax.set_title(f"Uncertainty axis projection — {DISPLAY_NAME}")\n'
             'ax.legend()\n'
             'plt.tight_layout()\n'
             'fig_path = Path("results") / MODEL_NAME / "fig_probe_projection.png"\n'
             'fig.savefig(fig_path, dpi=150, bbox_inches="tight")\n'
             'plt.show()'),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11.0"}
    return nb


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    out_dir = Path("notebooks")
    out_dir.mkdir(exist_ok=True)

    for key, display, model_type, notes in MODELS:
        if model_type == "decoder":
            nb = build_decoder_notebook(key, display, notes)
        else:
            nb = build_encoder_notebook(key, display, notes)

        nb_path = out_dir / f"{key}.ipynb"
        nbformat.write(nb, str(nb_path))
        print(f"Generated {nb_path}")

    print(f"\nAll notebooks written to {out_dir}/")


if __name__ == "__main__":
    main()
