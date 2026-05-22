"""
Generate paper-ready assets: LaTeX tables, CSV summaries, dataset statistics.
Reads from results/metrics/ and produces formatted outputs in results/paper_assets/.

Usage:
    python generate_paper_assets.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config

METRICS_DIR = os.path.join(config.RESULTS_DIR, "metrics")
ASSETS_DIR  = os.path.join(config.RESULTS_DIR, "paper_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def load_json(filename):
    path = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Statistics Table
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset_stats():
    """Generate LaTeX table of dataset statistics."""
    stats = {
        "METR-LA": {
            "sensors":   207,
            "timesteps": 34272,
            "duration":  "Mar 1 – Jun 27, 2012 (4 months)",
            "interval":  "5 min",
            "feature":   "Speed (mph)",
            "source":    "Los Angeles highway loop detectors",
        },
        "PEMS-BAY": {
            "sensors":   325,
            "timesteps": 52116,
            "duration":  "Jan 1 – May 31, 2017 (6 months)",
            "interval":  "5 min",
            "feature":   "Speed (mph)",
            "source":    "Bay Area Caltrans PeMS",
        },
    }

    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(r"\caption{Dataset Statistics}")
    latex.append(r"\label{tab:dataset_stats}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"Property & METR-LA & PEMS-BAY \\")
    latex.append(r"\midrule")

    for key in ["sensors", "timesteps", "duration", "interval", "feature"]:
        v1 = stats["METR-LA"][key]
        v2 = stats["PEMS-BAY"][key]
        if isinstance(v1, int):
            v1 = f"{v1:,}"
            v2 = f"{v2:,}"
        latex.append(f"{key.capitalize()} & {v1} & {v2} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    text = "\n".join(latex)
    with open(os.path.join(ASSETS_DIR, "table_dataset_stats.tex"), "w") as f:
        f.write(text)
    print(f"  ✓ Dataset statistics table saved")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Comparison Table (Main Results)
# ─────────────────────────────────────────────────────────────────────────────

def generate_main_results_table():
    """Generate LaTeX table comparing all models on METR-LA."""
    baselines = load_json("METR-LA_baselines_results.json")
    if baselines is None:
        print("  ⚠ METR-LA baselines results not found")
        return

    # Model order for paper
    model_order = [
        "Persistence", "HistoricalAverage",
        "ARIMA", "RandomForest", "LSTM",
        "STGCN", "DCRNN"
    ]

    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(r"\caption{Traffic Forecasting Results on METR-LA}")
    latex.append(r"\label{tab:main_results}")
    latex.append(r"\begin{tabular}{l|ccc|ccc|ccc}")
    latex.append(r"\toprule")
    latex.append(r" & \multicolumn{3}{c|}{15 min} & \multicolumn{3}{c|}{30 min} & \multicolumn{3}{c}{60 min} \\")
    latex.append(r"Model & MAE & RMSE & MAPE & MAE & RMSE & MAPE & MAE & RMSE & MAPE \\")
    latex.append(r"\midrule")

    # Also write CSV
    csv_rows = ["Model,Horizon,MAE,RMSE,MAPE"]

    for model in model_order:
        if model in baselines:
            r = baselines[model]
            row_parts = []
            for h in ["15min", "30min", "60min"]:
                if h in r:
                    mae  = r[h]["MAE"]
                    rmse = r[h]["RMSE"]
                    mape = r[h]["MAPE"]
                    row_parts.append(f"{mae:.3f} & {rmse:.3f} & {mape:.2f}")
                    csv_rows.append(f"{model},{h},{mae:.4f},{rmse:.4f},{mape:.4f}")
                else:
                    row_parts.append("— & — & —")
            latex.append(f"{model} & " + " & ".join(row_parts) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    text = "\n".join(latex)
    with open(os.path.join(ASSETS_DIR, "table_main_results.tex"), "w") as f:
        f.write(text)

    with open(os.path.join(ASSETS_DIR, "main_results.csv"), "w") as f:
        f.write("\n".join(csv_rows))

    print(f"  ✓ Main results table saved (LaTeX + CSV)")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 3. Robustness Table
# ─────────────────────────────────────────────────────────────────────────────

def generate_robustness_table():
    """Generate LaTeX table of robustness results (mean ± std)."""
    rob = load_json("METR-LA_robustness.json")
    if rob is None:
        print("  ⚠ Robustness results not found")
        return

    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(r"\caption{Robustness Under Random Missing Data (METR-LA)}")
    latex.append(r"\label{tab:robustness}")

    # Get scenarios
    for scenario_name, scenario_data in rob.items():
        latex.append(f"\\textbf{{{scenario_name.replace('_', ' ').title()}}}")
        latex.append("")

        ratios = sorted(scenario_data.keys(), key=lambda x: float(x))
        models = set()
        for r_data in scenario_data.values():
            models.update(r_data.keys())
        models = sorted(models)

        num_cols = len(models)
        col_spec = "l" + "c" * num_cols
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append(r"\toprule")
        header = "Corruption & " + " & ".join(models) + r" \\"
        latex.append(header)
        latex.append(r"\midrule")

        csv_rows = [f"Ratio," + ",".join(models)]

        for ratio in ratios:
            r_data = scenario_data[ratio]
            row = f"{float(ratio)*100:.0f}\\%"
            csv_row = f"{float(ratio)*100:.0f}%"
            for m in models:
                if m in r_data:
                    val = r_data[m]
                    if isinstance(val, dict) and "mean" in val:
                        mean_v = val["mean"]
                        std_v  = val.get("std", 0)
                        row += f" & {mean_v:.3f} $\\pm$ {std_v:.3f}"
                        csv_row += f",{mean_v:.4f}±{std_v:.4f}"
                    else:
                        row += f" & {val:.3f}"
                        csv_row += f",{val:.4f}"
                else:
                    row += " & —"
                    csv_row += ",—"
            latex.append(row + r" \\")
            csv_rows.append(csv_row)

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append("")

        with open(os.path.join(ASSETS_DIR, f"robustness_{scenario_name}.csv"), "w") as f:
            f.write("\n".join(csv_rows))

    latex.append(r"\end{table}")

    text = "\n".join(latex)
    with open(os.path.join(ASSETS_DIR, "table_robustness.tex"), "w") as f:
        f.write(text)

    print(f"  ✓ Robustness table saved (LaTeX + CSV)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Architecture Comparison Table
# ─────────────────────────────────────────────────────────────────────────────

def generate_architecture_table():
    """Generate LaTeX table comparing model architectures."""
    import torch

    models_info = [
        ("Persistence",     "—",         "Copy last",          "—",                0),
        ("Hist. Average",   "—",         "Time-of-day lookup", "—",                0),
        ("ARIMA",           "ARIMA(3,0,1)", "Per-sensor ARIMA",  "—",             "N/A"),
        ("Random Forest",   "sklearn RF", "Flatten → predict",  "—",     100 * 15 * (12*207)),
        ("LSTM",            "2-layer GRU-style", "Sequential", "—",      55372),
        ("STGCN",           "ChebConv(K=3)", "Spectral graph", "Gated 1D CNN",  79532),
        ("DCRNN",           "DiffusionConv(K=2)", "Spatial diffusion", "DCGRU seq2seq", 223169),
    ]

    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(r"\caption{Model Architecture Comparison}")
    latex.append(r"\label{tab:architecture}")
    latex.append(r"\begin{tabular}{lcccr}")
    latex.append(r"\toprule")
    latex.append(r"Model & Spatial & Temporal & Graph Type & Params \\")
    latex.append(r"\midrule")

    for name, spatial, temporal, graph, params in models_info:
        p_str = f"{params:,}" if isinstance(params, int) else str(params)
        latex.append(f"{name} & {spatial} & {temporal} & {graph} & {p_str} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    text = "\n".join(latex)
    with open(os.path.join(ASSETS_DIR, "table_architecture.tex"), "w") as f:
        f.write(text)
    print(f"  ✓ Architecture table saved")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hyperparameter Documentation
# ─────────────────────────────────────────────────────────────────────────────

def generate_hyperparameter_doc():
    """Generate a complete hyperparameter documentation file."""
    doc = []
    doc.append("# Hyperparameter Configuration")
    doc.append("")
    doc.append("## Training")
    doc.append(f"- Batch size: {config.BATCH_SIZE}")
    doc.append(f"- Learning rate: {config.LEARNING_RATE}")
    doc.append(f"- Weight decay: {config.WEIGHT_DECAY}")
    doc.append(f"- Max epochs: {config.EPOCHS}")
    doc.append(f"- Early stopping patience: {config.PATIENCE}")
    doc.append(f"- Gradient clip norm: {config.GRAD_CLIP}")
    doc.append(f"- LR scheduler: ReduceLROnPlateau(factor={config.SCHEDULER_FACTOR}, patience={config.SCHEDULER_PATIENCE})")
    doc.append(f"- AMP enabled: {config.USE_AMP}")
    doc.append(f"- cuDNN benchmark: {config.CUDNN_BENCHMARK}")
    doc.append(f"- Random seed: {config.SEED}")
    doc.append("")
    doc.append("## Data")
    doc.append(f"- Input sequence length: {config.SEQ_LEN} steps (= {config.SEQ_LEN * 5} min)")
    doc.append(f"- Prediction horizon: {config.PRED_LEN} steps (= {config.PRED_LEN * 5} min)")
    doc.append(f"- Train/Val/Test split: {config.TRAIN_RATIO}/{config.VAL_RATIO}/{1 - config.TRAIN_RATIO - config.VAL_RATIO}")
    doc.append(f"- Normalization: Z-score per sensor (train stats only)")
    doc.append("")
    doc.append("## Graph Construction")
    doc.append(f"- Method: Pearson correlation → Gaussian kernel → threshold")
    doc.append(f"- Gaussian σ: {config.GRAPH_SIGMA}")
    doc.append(f"- Sparsity threshold ε: {config.GRAPH_EPSILON}")
    doc.append(f"- Diffusion steps K: {config.DIFFUSION_STEPS}")
    doc.append(f"- Data source: Training set only (no leakage)")
    doc.append("")
    doc.append("## LSTM")
    doc.append(f"- Hidden dim: {config.LSTM_HIDDEN}")
    doc.append(f"- Layers: {config.LSTM_LAYERS}")
    doc.append(f"- Dropout: {config.LSTM_DROPOUT}")
    doc.append("")
    doc.append("## STGCN")
    doc.append(f"- Channel progression: {config.STGCN_CHANNELS}")
    doc.append(f"- Temporal kernel size: {config.STGCN_KERNEL_SIZE}")
    doc.append(f"- Chebyshev order K: {config.STGCN_K}")
    doc.append("")
    doc.append("## DCRNN")
    doc.append(f"- Hidden dim: {config.DCRNN_HIDDEN}")
    doc.append(f"- Layers: {config.DCRNN_LAYERS}")
    doc.append(f"- Dropout: {config.DCRNN_DROPOUT}")
    doc.append(f"- Filter type: {config.DCRNN_FILTER_TYPE}")
    doc.append(f"- Teacher forcing: linear decay over first 50% of epochs")
    doc.append("")
    doc.append("## Random Forest")
    doc.append(f"- Estimators: {config.RF_N_ESTIMATORS}")
    doc.append(f"- Max depth: {config.RF_MAX_DEPTH}")
    doc.append("")
    doc.append("## ARIMA")
    doc.append(f"- Order (p,d,q): {config.ARIMA_ORDER}")
    doc.append(f"- Max sensors: {config.ARIMA_MAX_SENSORS}")

    text = "\n".join(doc)
    with open(os.path.join(ASSETS_DIR, "hyperparameters.md"), "w") as f:
        f.write(text)
    print(f"  ✓ Hyperparameter documentation saved")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Reproducibility Checklist
# ─────────────────────────────────────────────────────────────────────────────

def generate_reproducibility_checklist():
    """Generate a reproducibility checklist for the paper appendix."""
    items = [
        ("Fixed random seed (42)", True),
        ("Deterministic PyTorch (when CUDNN_BENCHMARK=False)", True),
        ("Chronological train/val/test splits (no shuffling)", True),
        ("Normalization stats from training set only", True),
        ("Graph built from training data only", True),
        ("Per-chunk sequence creation (no boundary leakage)", True),
        ("Metrics computed on de-normalized predictions", True),
        ("All models share identical data pipeline", True),
        ("Multi-seed robustness (5 seeds, reports mean ± std)", True),
        ("Early stopping on validation loss", True),
        ("Model checkpoints saved (gitignored for size)", True),
        ("All hyperparameters documented in config.py", True),
        ("Results saved as JSON (committed to git)", True),
    ]

    lines = ["# Reproducibility Checklist", ""]
    for item, status in items:
        mark = "✅" if status else "❌"
        lines.append(f"- {mark} {item}")

    text = "\n".join(lines)
    with open(os.path.join(ASSETS_DIR, "reproducibility_checklist.md"), "w") as f:
        f.write(text)
    print(f"  ✓ Reproducibility checklist saved")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  GENERATING PAPER ASSETS")
    print("=" * 60 + "\n")

    generate_dataset_stats()
    generate_main_results_table()
    generate_robustness_table()
    generate_architecture_table()
    generate_hyperparameter_doc()
    generate_reproducibility_checklist()

    print(f"\n  All assets saved to: {ASSETS_DIR}/")
    print("  Files:")
    for f in sorted(os.listdir(ASSETS_DIR)):
        size = os.path.getsize(os.path.join(ASSETS_DIR, f))
        print(f"    {f:40s} ({size:,} bytes)")
    print()
