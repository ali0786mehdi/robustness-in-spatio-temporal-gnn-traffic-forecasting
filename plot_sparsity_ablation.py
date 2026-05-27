"""
Plot the Graph Sparsity Ablation results.

Generates a 2x2 figure:
  Top-left:    Overall MAE vs Avg Connections (STGCN vs DCRNN)
  Top-right:   Robustness MAE @20% vs Avg Connections
  Bottom-left: Per-horizon MAE breakdown (15/30/60 min) at each epsilon
  Bottom-right: Performance degradation ratio (robust/clean) vs graph density
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
})

MODEL_COLOR = {
    "STGCN": "#DD8452",
    "DCRNN": "#55A868",
}
HORIZON_COLOR = {
    "15min": "#4C72B0",
    "30min": "#DD8452",
    "60min": "#C44E52",
}


def load_results(dataset="METR-LA"):
    path = os.path.join(config.RESULTS_DIR, "metrics",
                        f"{dataset}_sparsity_ablation.json")
    with open(path) as f:
        data = json.load(f)

    # Handle list format (older versions saved as list instead of dict)
    if isinstance(data, list):
        converted = {}
        for item in data:
            if isinstance(item, dict):
                # Try to find epsilon key
                eps = item.get("epsilon", item.get("eps", None))
                if eps is not None:
                    converted[f"eps={eps}"] = item
                elif "graph_stats" in item:
                    # Guess epsilon from graph stats or use index
                    converted[f"item_{len(converted)}"] = item
        data = converted if converted else {f"eps={i}": item for i, item in enumerate(data)}

    return data


def plot_sparsity_ablation(results, dataset="METR-LA", save_dir=None):
    labels      = list(results.keys())                # ["eps=0.1", ...]
    epsilons    = [float(l.split("=")[1]) for l in labels if "=" in l]
    avg_conns   = [results[l]["graph_stats"]["avg_conn"] for l in labels]
    densities   = [results[l]["graph_stats"]["density_pct"] for l in labels]

    # Check if any models have data
    has_models = any(len(results[l].get("models", {})) > 0 for l in labels)
    if not has_models:
        print("  ⚠ No model results found in sparsity ablation data.")
        print("    The JSON has graph_stats but 'models' is empty for all epsilons.")
        print("    Re-run: python -u run_sparsity_ablation.py")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Graph Sparsity Ablation — {dataset}",
                 fontsize=16, fontweight="bold", y=1.01)

    # ── Panel 1: Overall MAE vs avg connections ──────────────────────────────
    ax = axes[0, 0]
    for m, col in MODEL_COLOR.items():
        if m not in results[labels[0]]["models"]:
            continue
        maes = [results[l]["models"][m]["MAE_overall"] for l in labels]
        ax.plot(avg_conns, maes, "o-", color=col, linewidth=2.2,
                markersize=8, label=m)
        # Annotate epsilon values
        for x, y, eps in zip(avg_conns, maes, epsilons):
            ax.annotate(f"ε={eps}", (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=8, color=col)
    ax.set_title("Overall MAE vs Graph Density")
    ax.set_xlabel("Avg Connections per Node")
    ax.set_ylabel("MAE (mph)")
    ax.legend()

    # ── Panel 2: Robustness @20% vs avg connections ──────────────────────────
    ax = axes[0, 1]
    for m, col in MODEL_COLOR.items():
        if m not in results[labels[0]]["models"]:
            continue
        clean_maes = [results[l]["models"][m]["MAE_overall"]    for l in labels]
        rob_maes   = [results[l]["models"][m]["robustness_20pct_MAE"] for l in labels]
        ax.plot(avg_conns, rob_maes, "s--", color=col, linewidth=2.2,
                markersize=8, label=f"{m} (robust)")
        ax.plot(avg_conns, clean_maes, "o:", color=col, linewidth=1.2,
                markersize=5, alpha=0.5, label=f"{m} (clean)")
    ax.set_title("MAE Under 20% Random Missing vs Graph Density")
    ax.set_xlabel("Avg Connections per Node")
    ax.set_ylabel("MAE (mph)")
    ax.legend(fontsize=8, ncol=2)

    # ── Panel 3: Per-horizon breakdown across epsilon values ─────────────────
    ax = axes[1, 0]
    x = np.arange(len(labels))
    width = 0.13
    offsets = {"STGCN": {"15min": -0.2, "30min": -0.07, "60min": 0.07},
               "DCRNN": {"15min": 0.2,  "30min": 0.33,  "60min": 0.46}}
    hatches = {"STGCN": "", "DCRNN": "//"}

    for m in MODEL_COLOR:
        if m not in results[labels[0]]["models"]:
            continue
        for horizon, hcol in HORIZON_COLOR.items():
            key = f"MAE_{horizon}"
            vals = [results[l]["models"][m][key] for l in labels]
            offset = offsets[m][horizon]
            bars = ax.bar(x + offset, vals, width, color=hcol, alpha=0.75,
                          hatch=hatches[m],
                          label=f"{m} {horizon}" if m == "STGCN" else None)

    ax.set_title("Per-Horizon MAE by Epsilon")
    ax.set_xlabel("Epsilon (graph threshold)")
    ax.set_ylabel("MAE (mph)")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("eps=", "ε=") for l in labels])
    # Custom legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=h) for h, c in HORIZON_COLOR.items()]
    legend_els += [Patch(facecolor="gray", label="STGCN (solid)"),
                   Patch(facecolor="gray", hatch="//", label="DCRNN (hatched)")]
    ax.legend(handles=legend_els, fontsize=8, ncol=2)

    # ── Panel 4: Noise amplification ratio ───────────────────────────────────
    ax = axes[1, 1]
    for m, col in MODEL_COLOR.items():
        if m not in results[labels[0]]["models"]:
            continue
        clean = np.array([results[l]["models"][m]["MAE_overall"] for l in labels])
        rob   = np.array([results[l]["models"][m]["robustness_20pct_MAE"] for l in labels])
        ratio = rob / clean     # >1 means degraded; closer to 1 = more robust
        ax.plot(avg_conns, ratio, "^-", color=col, linewidth=2.2,
                markersize=8, label=m)
        for x_val, y_val, eps in zip(avg_conns, ratio, epsilons):
            ax.annotate(f"ε={eps}", (x_val, y_val),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=8, color=col)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2)
    ax.set_title("Noise Amplification Ratio (Robust MAE / Clean MAE)")
    ax.set_xlabel("Avg Connections per Node")
    ax.set_ylabel("Ratio (higher = more degraded)")
    ax.legend()

    plt.tight_layout()

    save_dir = save_dir or os.path.join(config.RESULTS_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset}_sparsity_ablation.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    return out_path


def print_key_findings(results):
    if not isinstance(results, dict):
        print("  ⚠ Results format unexpected — skipping key findings.")
        return
    labels   = list(results.keys())
    has_models = any(len(results[l].get("models", {})) > 0 for l in labels)
    if not has_models:
        print("\n  ⚠ No model results to summarize — 'models' is empty in all entries.")
        print("    Re-run: python -u run_sparsity_ablation.py")
        return
    print(f"\n{'='*65}")
    print("  KEY FINDINGS: GRAPH SPARSITY ABLATION")
    print(f"{'='*65}")

    for m in ["STGCN", "DCRNN"]:
        if m not in results[labels[0]]["models"]:
            continue
        print(f"\n  {m}:")
        best_label = min(
            labels, key=lambda l: results[l]["models"][m]["MAE_overall"]
        )
        worst_label = max(
            labels, key=lambda l: results[l]["models"][m]["MAE_overall"]
        )
        best_conn  = results[best_label]["graph_stats"]["avg_conn"]
        worst_conn = results[worst_label]["graph_stats"]["avg_conn"]
        best_mae   = results[best_label]["models"][m]["MAE_overall"]
        worst_mae  = results[worst_label]["models"][m]["MAE_overall"]

        print(f"  Best:  ε={best_label.split('=')[1]}"
              f" (avg_conn={best_conn})  MAE={best_mae:.4f}")
        print(f"  Worst: ε={worst_label.split('=')[1]}"
              f" (avg_conn={worst_conn})  MAE={worst_mae:.4f}")

        # Noise amplification at each epsilon
        print(f"  {'Epsilon':<8} {'AvgConn':>8} {'CleanMAE':>10} "
              f"{'Rob@20%':>10} {'Amp. Ratio':>12}")
        print(f"  {'-'*52}")
        for l in labels:
            c = results[l]["models"][m]["MAE_overall"]
            r = results[l]["models"][m]["robustness_20pct_MAE"]
            conn = results[l]["graph_stats"]["avg_conn"]
            eps  = l.split("=")[1]
            print(f"  {eps:<8} {conn:>8.1f} {c:>10.4f} {r:>10.4f} {r/c:>12.4f}")


if __name__ == "__main__":
    results = load_results("METR-LA")
    print_key_findings(results)
    plot_sparsity_ablation(results, "METR-LA")
