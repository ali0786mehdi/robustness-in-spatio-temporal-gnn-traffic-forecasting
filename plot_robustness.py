"""
Plot robustness degradation curves from saved results.
Generates a side-by-side figure for Random Missing and Sensor Failure
showing MAE vs. corruption ratio for all available models.
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
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# Defines display order and styling for every possible model
MODEL_STYLE = {
    # ── Sanity baselines ──────────────────────────────────────────────────────
    "Persistence":       dict(color="#BBBBBB", marker="x",  linestyle=":",   label="Persistence",      zorder=2),
    "HistoricalAverage": dict(color="#888888", marker="+",  linestyle=":",   label="Hist. Average",    zorder=2),
    # ── Classical baselines ───────────────────────────────────────────────────
    "ARIMA":             dict(color="#E377C2", marker="v",  linestyle="-.",  label="ARIMA",            zorder=3),
    "RandomForest":      dict(color="#9467BD", marker="D",  linestyle="-.",  label="Random Forest",    zorder=3),
    # ── Deep temporal ─────────────────────────────────────────────────────────
    "LSTM":              dict(color="#4C72B0", marker="o",  linestyle="-",   label="LSTM",             zorder=4),
    # ── Graph models ──────────────────────────────────────────────────────────
    "STGCN":             dict(color="#DD8452", marker="s",  linestyle="--",  label="STGCN",            zorder=5),
    "DCRNN":             dict(color="#55A868", marker="^",  linestyle="-.",  label="DCRNN",            zorder=5),
}


def load_results(dataset="METR-LA"):
    path = os.path.join(config.RESULTS_DIR, "metrics", f"{dataset}_robustness.json")
    with open(path) as f:
        return json.load(f)


def plot_degradation_curves(results, dataset="METR-LA", save_dir=None):
    scenarios = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    fig.suptitle(
        f"Model Robustness to Missing Data — {dataset}",
        fontsize=16, fontweight="bold", y=1.02,
    )

    scenario_titles = {
        "random_missing": "Random Missing Values\n(uniform % of all observations zeroed)",
        "sensor_failure": "Structured Sensor Failure\n(entire sensors zeroed for all timesteps)",
    }

    for ax, scenario in zip(axes, scenarios):
        scenario_data = results[scenario]
        ratio_keys    = list(scenario_data.keys())
        ratios_pct    = [float(r) * 100 for r in ratio_keys]

        available = set(scenario_data[ratio_keys[0]].keys())

        def _mean_std(entry, model):
            """Support both old float and new {mean, std} JSON formats."""
            v = entry.get(model)
            if v is None:
                return None, 0.0
            if isinstance(v, dict):
                return v['mean'], v.get('std', 0.0)
            return float(v), 0.0

        for model, style in MODEL_STYLE.items():
            if model not in available:
                continue
            means, stds = [], []
            for r in ratio_keys:
                m, s = _mean_std(scenario_data[r], model)
                means.append(m); stds.append(s)

            means = np.array(means, dtype=float)
            stds  = np.array(stds,  dtype=float)

            ax.plot(ratios_pct, means, linewidth=2.2, markersize=7, **style)

            # ±1σ shaded band (only where std > 0)
            if stds.max() > 0:
                ax.fill_between(ratios_pct,
                                means - stds, means + stds,
                                color=style["color"], alpha=0.18)

        ax.set_title(scenario_titles.get(scenario, scenario), pad=10)
        ax.set_xlabel("Corruption Ratio (%)")
        ax.set_ylabel("MAE (mph)")
        ax.xaxis.set_major_formatter(ticker.PercentFormatter())
        ax.legend(loc="upper left", framealpha=0.8)

    plt.tight_layout()

    save_dir = save_dir or os.path.join(config.RESULTS_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{dataset}_robustness_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")
    return path



def print_analysis(results):
    """Dynamically compute and print a research finding summary."""
    print("\n" + "=" * 68)
    print("  ROBUSTNESS ANALYSIS  (mean ± std over corruption seeds)")
    print("=" * 68)

    def _get(entry, model):
        v = entry.get(model)
        if v is None:
            return None, None
        if isinstance(v, dict):
            return v['mean'], v.get('std', 0.0)
        return float(v), 0.0

    for scenario, data in results.items():
        ratio_keys = list(data.keys())
        worst_key  = ratio_keys[-1]
        worst_pct  = int(float(worst_key) * 100)

        available_models = list(data[ratio_keys[0]].keys())

        print(f"\n  Scenario: {scenario.replace('_', ' ').title()}")
        col_w = max(len(m) for m in available_models) + 2
        print(f"  {'Model':<{col_w}}  {'MAE 0%':>10}  {'MAE {:.0f}% (μ±σ)'.format(worst_pct):>16}  {'Δ MAE':>8}  {'Deg.%':>7}")
        print(f"  {'-'*(col_w + 46)}")

        degradations = {}
        for model in available_models:
            base_m,  _     = _get(data["0.0"],    model)
            worst_m, worst_s = _get(data[worst_key], model)
            if base_m is None or worst_m is None:
                continue
            delta = worst_m - base_m
            pct   = delta / base_m * 100
            degradations[model] = pct
            worst_str = f"{worst_m:.3f}±{worst_s:.3f}"
            print(f"  {model:<{col_w}}  {base_m:>10.4f}  {worst_str:>16}  {delta:>+8.3f}  {pct:>6.1f}%")

        deep_models = [m for m in available_models if m in ("LSTM", "STGCN", "DCRNN")]
        if deep_models:
            most_robust  = min(deep_models, key=lambda m: degradations.get(m, 999))
            least_robust = max(deep_models, key=lambda m: degradations.get(m, 0))
            print(f"\n  → Most robust deep model:  {most_robust} ({degradations[most_robust]:.1f}%)")
            print(f"  → Least robust deep model: {least_robust} ({degradations[least_robust]:.1f}%)")

    print("\n" + "=" * 68)


if __name__ == "__main__":
    results = load_results("METR-LA")
    print_analysis(results)
    plot_degradation_curves(results, "METR-LA")
