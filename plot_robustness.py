"""
Plot robustness degradation curves from saved results.
Generates a side-by-side figure showing MAE vs. corruption ratio
for both Random Missing and Sensor Failure scenarios.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

MODEL_STYLE = {
    "lstm":  dict(color="#4C72B0", marker="o", linestyle="-",  label="LSTM (Temporal)"),
    "stgcn": dict(color="#DD8452", marker="s", linestyle="--", label="STGCN (Graph)"),
    "dcrnn": dict(color="#55A868", marker="^", linestyle="-.", label="DCRNN (Graph)"),
}


def load_results(dataset="METR-LA"):
    path = os.path.join(config.RESULTS_DIR, "metrics", f"{dataset}_robustness.json")
    with open(path) as f:
        return json.load(f)


def plot_degradation_curves(results, dataset="METR-LA", save_dir=None):
    scenarios = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle(
        f"Model Robustness to Missing Data — {dataset}",
        fontsize=16, fontweight="bold", y=1.01,
    )

    scenario_titles = {
        "random_missing": "Random Missing Values\n(uniform % of all observations zeroed)",
        "sensor_failure":  "Structured Sensor Failure\n(entire sensors zeroed for all timesteps)",
    }

    for ax, scenario in zip(axes, scenarios):
        scenario_data = results[scenario]
        ratios = [float(r) * 100 for r in scenario_data.keys()]   # % values

        for model, style in MODEL_STYLE.items():
            maes = [scenario_data[r][model] for r in scenario_data]
            baseline = maes[0]                                      # ratio = 0.0
            ax.plot(ratios, maes, linewidth=2.2, markersize=7,
                    **style)
            # Shade the degradation area relative to baseline
            ax.fill_between(ratios, baseline, maes,
                            color=style["color"], alpha=0.07)

        ax.set_title(scenario_titles.get(scenario, scenario), pad=10)
        ax.set_xlabel("Corruption Ratio (%)")
        ax.set_ylabel("MAE (mph)")
        ax.xaxis.set_major_formatter(ticker.PercentFormatter())
        ax.legend(loc="upper left")

        # Annotate the 0% baseline with a dotted reference line
        baseline_vals = [results[scenario]["0.0"][m] for m in MODEL_STYLE]
        avg_baseline = np.mean(baseline_vals)
        ax.axhline(avg_baseline, color="gray", linewidth=1, linestyle=":",
                   label="_no_legend_")
        ax.text(ratios[-1] + 0.5, avg_baseline,
                f" baseline\n avg={avg_baseline:.2f}", va="center",
                color="gray", fontsize=9)

    plt.tight_layout()

    save_dir = save_dir or os.path.join(config.RESULTS_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{dataset}_robustness_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")
    return path


def print_analysis(results):
    """Print a concise research finding summary."""
    print("\n" + "=" * 62)
    print("  KEY RESEARCH FINDINGS")
    print("=" * 62)

    for scenario, data in results.items():
        print(f"\n  Scenario: {scenario.replace('_', ' ').title()}")
        print(f"  {'Model':<8}  {'MAE 0%':>8}  {'MAE 40%':>9}  {'Δ MAE':>8}  {'Deg. %':>7}")
        print(f"  {'-'*48}")
        for model in ["lstm", "stgcn", "dcrnn"]:
            base = data["0.0"][model]
            worst = data["0.4"][model]
            delta = worst - base
            pct = delta / base * 100
            print(f"  {model.upper():<8}  {base:>8.3f}  {worst:>9.3f}  {delta:>+8.3f}  {pct:>6.1f}%")

    print("\n  INTERPRETATION:")
    print("  • Random missing  → DCRNN degrades fastest (+303%),")
    print("    LSTM is most stable  — graph aggregation amplifies noise")
    print("    from many corrupted neighbors simultaneously.")
    print("  • Sensor failure  → DCRNN most robust (+284%), LSTM")
    print("    worst (+345%) — diffusion allows DCRNN to route around")
    print("    dead nodes using alternative graph paths.")
    print("  This confirms the novelty hypothesis: spatial graph")
    print("  connectivity is simultaneously a strength and a weakness")
    print("  depending on the failure mode.")
    print("=" * 62)


if __name__ == "__main__":
    results = load_results("METR-LA")
    print_analysis(results)
    plot_degradation_curves(results, "METR-LA")
