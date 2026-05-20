def figure_steering_tradeoff():
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    with open("results/final_experiment.json") as f:
        data = json.load(f)

    summary = data["summary"]
    alphas = sorted([float(k) for k in summary["probe"].keys()])

    metrics = [
        ("hedge_score_mean", "Hedge score"),
        ("lexical_diversity_mean", "Lexical diversity"),
        ("token_validity_mean", "Token validity"),
        ("perplexity_mean", "Perplexity"),
    ]

    # Publication-style figure setup
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 8,
    })

    fig, axes = plt.subplots(
        1, 4,
        figsize=(16, 3.8),
        sharex=True,
        constrained_layout=True
    )

    # Consistent style mapping
    styles = {
        "probe": dict(color="#1f77b4", marker="o", linestyle="-", linewidth=1.8),
        "ortho": dict(color="#ff7f0e", marker="s", linestyle="--", linewidth=1.8),
    }

    for ax, (metric_key, label) in zip(axes, metrics):
        for direction in ["probe", "ortho"]:
            vals = [
                summary[direction][str(a)].get(metric_key, np.nan)
                for a in alphas
            ]
            ax.plot(alphas, vals, label=direction, **styles[direction])

        ax.set_title(label)
        ax.grid(True, alpha=0.25)

        if metric_key == "perplexity_mean":
            ax.set_yscale("log")

    # Shared x-labels
    for ax in axes:
        ax.set_xlabel(r"Steering strength $\alpha_{\mathrm{frac}}$")

    # Single global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle(
        "Steering tradeoff: probe vs orthogonalized direction (layer 16)",
        fontsize=13
    )

    plt.savefig("results/fig_steering_tradeoff_v2.pdf", bbox_inches="tight")
    plt.savefig("results/fig_steering_tradeoff_v2.png", dpi=200, bbox_inches="tight")

    print("Saved fig_steering_tradeoff_v2.{pdf,png}")


def main():
    figure_steering_tradeoff()

if __name__ == "__main__":
    main()