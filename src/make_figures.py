"""Generate the paper's figures from final_experiment.json and probe results."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def figure_probing_accuracy():
    with open("results/probe_accuracy_per_layer.json") as f:
        data = json.load(f)["results"]
    layers = sorted(int(k) for k in data.keys())
    means = [data[str(l)]["mean_acc"] for l in layers]
    ci_low = [data[str(l)]["ci_low"] for l in layers]
    ci_high = [data[str(l)]["ci_high"] for l in layers]
    yerr_low = [m - l for m, l in zip(means, ci_low)]
    yerr_high = [h - m for m, h in zip(means, ci_high)]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.errorbar(layers, means, yerr=[yerr_low, yerr_high],
                marker="o", capsize=3, linewidth=1.5)
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Chance")
    ax.set_xlabel("BioGPT layer")
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Linear probe accuracy for uncertainty classification")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/fig_probe_accuracy.pdf", bbox_inches="tight")
    plt.savefig("results/fig_probe_accuracy.png", dpi=150, bbox_inches="tight")
    print("Saved fig_probe_accuracy.{pdf,png}")


def figure_steering_tradeoff():
    with open("results/final_experiment.json") as f:
        data = json.load(f)
    summary = data["summary"]
    alphas = sorted([float(k) for k in summary["probe"].keys()])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    metrics = [
        ("hedge_score_mean", "Hedge score", axes[0, 0]),
        ("lexical_diversity_mean", "Lexical diversity", axes[0, 1]),
        ("token_validity_mean", "Token validity", axes[1, 0]),
        ("perplexity_mean", "Perplexity (log scale)", axes[1, 1]),
    ]
    for metric_key, label, ax in metrics:
        for direction, style in [("probe", "o-"), ("ortho", "s--")]:
            vals = [summary[direction][str(a)].get(metric_key, np.nan)
                    for a in alphas]
            ax.plot(alphas, vals, style, label=direction, linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if "perplexity" in metric_key:
            ax.set_yscale("log")
    axes[1, 0].set_xlabel(r"Steering strength $\alpha_{\mathrm{frac}}$")
    axes[1, 1].set_xlabel(r"Steering strength $\alpha_{\mathrm{frac}}$")
    plt.suptitle("Steering tradeoff: probe vs orthogonalized direction (layer 16)")
    plt.tight_layout()
    plt.savefig("results/fig_steering_tradeoff.pdf", bbox_inches="tight")
    plt.savefig("results/fig_steering_tradeoff.png", dpi=150, bbox_inches="tight")
    print("Saved fig_steering_tradeoff.{pdf,png}")


if __name__ == "__main__":
    figure_probing_accuracy()
    figure_steering_tradeoff()