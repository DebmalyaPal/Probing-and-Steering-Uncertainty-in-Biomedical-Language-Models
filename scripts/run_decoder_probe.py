"""
End-to-end decoder probing + steering run.

Usage (from project root):
    python scripts/run_decoder_probe.py --model biogpt
    python scripts/run_decoder_probe.py --model biogpt_large
    python scripts/run_decoder_probe.py --model biomedlm
    python scripts/run_decoder_probe.py --model biomistral
    python scripts/run_decoder_probe.py --model meditron

Runs the full decoder pipeline:
  1. Load BioScope contrast set
  2. Load model on CUDA / MPS / CPU (auto-detected)
  3. Probe every layer (0–N) — saves probe_accuracy_per_layer.json
  4. Orthogonalize at best layer
  5. Calibrate hidden norms
  6. Steering sweep: generate text at each alpha × seed × prompt × direction
  7. Compute hedge score, perplexity, type-token ratio, English word fraction
  8. Save full results + print summary table
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.model_registry    import get_config, list_models
from src.model_loader      import load_model, get_device
from src.bioscope_parser   import build_balanced_contrast_set
from src.experiment_runner import run_decoder_experiment, ALPHAS, SEEDS
from src.results_manager   import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="biogpt",
        choices=list_models("decoder"),
        help="Decoder model key from the registry",
    )
    parser.add_argument(
        "--max_per_class", type=int, default=200,
        help="Sentences per class (default 200, matches paper)",
    )
    parser.add_argument(
        "--steer_layer", type=int, default=None,
        help="Override steering layer (default: auto from probing)",
    )
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
        help="Numeric precision used for the run (default: float32)",
    )
    args = parser.parse_args()

    cfg    = get_config(args.model)
    device = get_device()

    print(f"\n{'='*60}")
    print(f"Model      : {cfg['display_name']}")
    print(f"HF ID      : {cfg['hf_id']}")
    print(f"Device     : {device}")
    print(f"Hidden dim : {cfg['hidden_dim']}  |  Layers: {cfg['num_layers']}")
    print(f"Quantize   : {cfg['quantize']}")
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
    # Steer layer priority: CLI flag > registry value > auto (best from probing)
    steer_layer = args.steer_layer if args.steer_layer is not None else cfg["steer_layer"]
    results = run_decoder_experiment(
        args.model, tok, model,
        uncertain, certain,
        device=device,
        steer_layer=steer_layer,
        dtype=args.dtype,
        verbose=True,
    )

    # ── save ──────────────────────────────────────────────────────────────
    save_results(args.model, "experiment", results, device=device, dtype=args.dtype)

    # ── summary: probe accuracy by layer ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Probe accuracy by layer — {cfg['display_name']}")
    print(f"{'='*60}")
    probe = results["probe_results"]
    for layer in sorted(int(k) for k in probe.keys()):
        r   = probe[str(layer)]
        bar = "█" * int(r["mean_acc"] * 20)
        print(
            f"  Layer {layer:>3}  {r['mean_acc']:>6.2%}  "
            f"±{r['std_acc']:.2%}  {bar}"
        )

    best = max(probe, key=lambda k: probe[k]["mean_acc"])
    print(f"\n  Best layer: {best}  ({probe[best]['mean_acc']:.2%})")
    print(f"  Steering at layer: {results['config']['layer']}")

    # ── summary: steering metrics ──────────────────────────────────────────
    for dir_name in ("probe", "ortho"):
        dir_summary = results["summary"].get(dir_name, {})
        print(f"\nSteering summary — direction: {dir_name}")
        hdr = "{:>7}  {:>8}  {:>8}  {:>8}  {:>10}".format(
            "alpha", "hedge", "ppl", "lex_div", "tok_valid"
        )
        print(hdr)
        print("-" * len(hdr))
        for af in ALPHAS:
            s  = dir_summary.get(str(af), {})
            hg = s.get("hedge_score_mean",        float("nan"))
            pp = s.get("perplexity_mean",          float("nan"))
            ld = s.get("lexical_diversity_mean",   float("nan"))
            tv = s.get("token_validity_mean",      float("nan"))
            print("{:>7.3f}  {:>8.4f}  {:>8.2f}  {:>8.4f}  {:>10.4f}".format(
                af, hg, pp, ld, tv
            ))

    print(f"\nResults saved → results/{args.model}/{device}/{args.dtype}/")


if __name__ == "__main__":
    main()
