"""
Quick end-to-end encoder probing run.

Usage (from project root):
    python scripts/run_encoder_probe.py --model biobert
    python scripts/run_encoder_probe.py --model scibert
    python scripts/run_encoder_probe.py --model clinicalbert
    python scripts/run_encoder_probe.py --model bluebert

Runs the full encoder pipeline:
  1. Load BioScope contrast set
  2. Load model on MPS / CUDA / CPU (auto-detected)
  3. Probe every layer (0–12) — saves probe_accuracy_per_layer.json
  4. Orthogonalize at best layer
  5. Representation-steering sweep
  6. Save full results + print summary table
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np

from src.model_registry  import get_config, list_models
from src.model_loader    import load_model, get_device
from src.bioscope_parser import build_balanced_contrast_set
from src.experiment_runner import run_encoder_experiment
from src.results_manager import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="biobert",
        choices=list_models("encoder"),
        help="Encoder model key from the registry",
    )
    parser.add_argument(
        "--max_per_class", type=int, default=200,
        help="Sentences per class (default 200, matches paper)",
    )
    args = parser.parse_args()

    cfg    = get_config(args.model)
    device = get_device()

    print(f"\n{'='*60}")
    print(f"Model      : {cfg['display_name']}")
    print(f"HF ID      : {cfg['hf_id']}")
    print(f"Device     : {device}")
    print(f"Hidden dim : {cfg['hidden_dim']}  |  Layers: {cfg['num_layers']}")
    print(f"{'='*60}\n")

    # ── data ──────────────────────────────────────────────────────────────
    print("Loading BioScope contrast set...")
    uncertain, certain = build_balanced_contrast_set(
        str(ROOT / "data" / "bioscope"),
        max_per_class=args.max_per_class,
        seed=42,
    )
    print(f"  {len(uncertain)} uncertain  |  {len(certain)} certain\n")

    # ── model ─────────────────────────────────────────────────────────────
    print(f"Loading {cfg['display_name']}...")
    tok, model = load_model(args.model, device=device)
    print(f"  Loaded.\n")

    # ── experiment ────────────────────────────────────────────────────────
    results = run_encoder_experiment(
        args.model, tok, model,
        uncertain, certain,
        device=device,
        verbose=True,
    )

    # ── save ──────────────────────────────────────────────────────────────
    save_results(args.model, "experiment", results)

    # ── summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Probe accuracy by layer — {cfg['display_name']}")
    print(f"{'='*60}")
    probe = results["probe_results"]
    for layer in sorted(int(k) for k in probe.keys()):
        r = probe[str(layer)]
        bar = "█" * int(r["mean_acc"] * 20)
        print(
            f"  Layer {layer:>2}  {r['mean_acc']:>6.2%}  "
            f"±{r['std_acc']:.2%}  {bar}"
        )

    best = max(probe, key=lambda k: probe[k]["mean_acc"])
    print(f"\n  Best layer: {best}  ({probe[best]['mean_acc']:.2%})")

    print(f"\nRepresentation steering summary (best layer)")
    print(f"{'alpha':>7}  {'probe Δproj':>13}  {'ortho Δproj':>13}")
    print("-" * 38)
    summary = results["summary"]
    for af in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        pd = summary.get("probe", {}).get(str(af), {}).get("delta_mean", float("nan"))
        od = summary.get("ortho", {}).get(str(af), {}).get("delta_mean", float("nan"))
        print(f"  {af:.3f}  {pd:>13.4f}  {od:>13.4f}")

    print(f"\nResults saved → results/{args.model}/")


if __name__ == "__main__":
    main()
