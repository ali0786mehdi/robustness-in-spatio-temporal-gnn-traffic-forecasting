"""
Graph Sparsity Ablation Study.

Tests how the graph threshold (epsilon) affects:
  1. Graph density (avg connections per node)
  2. STGCN and DCRNN test performance (MAE at each horizon)
  3. Robustness at 20% random missing — does a denser graph amplify noise more?

Epsilon values tested: 0.1 (dense) → 0.2 → 0.3 (current) → 0.5 (very sparse)

Results are saved to:
  results/metrics/sparsity_ablation_results.json
  results/plots/sparsity_ablation.png
"""

import os
import sys
import json
import time
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.config import set_seed
from src.data_loader import prepare_dataset
from src.graph_builder import build_graph, compute_correlation_adj
from src.evaluate import evaluate_predictions, print_results
from src.train import train_model, predict_model
from src.robustness import inject_random_missing


# ─────────────────────────────────────────────────────────────────────────────
# Graph density stats
# ─────────────────────────────────────────────────────────────────────────────

def graph_density_stats(adj):
    n = adj.shape[0]
    nnz  = int(np.count_nonzero(adj))
    # Exclude self-loops from connection count
    off_diag = nnz - n
    avg_conn = off_diag / n
    density  = off_diag / (n * (n - 1))
    return {
        "num_nodes":    n,
        "num_edges":    off_diag,
        "avg_conn":     round(avg_conn, 2),
        "density_pct":  round(density * 100, 3),
        "sparsity_pct": round((1 - density) * 100, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train one model variant
# ─────────────────────────────────────────────────────────────────────────────

def train_eval_stgcn(data_prepared, graph_data, label, mean, std, epochs, dataset_name):
    """Train STGCN with a given graph and return test metrics."""
    from src.models.stgcn import STGCN

    num_sensors  = data_prepared['splits']['train'][0].shape[2]
    device       = config.DEVICE

    model = STGCN(
        num_sensors=num_sensors,
        seq_len=config.SEQ_LEN,
        pred_len=config.PRED_LEN,
        K=config.STGCN_K,
        channels=config.STGCN_CHANNELS,
    )

    print(f"\n  Training STGCN [{label}] for {epochs} epochs ...")
    _config = config
    _config.EPOCHS = epochs

    history = train_model(
        model, data_prepared['loaders']['train'], data_prepared['loaders']['val'],
        _config, 'stgcn', dataset_name, graph_data=graph_data,
    )

    preds, gt, _ = predict_model(
        model, data_prepared['loaders']['test'], _config, 'stgcn', graph_data=graph_data,
    )
    metrics = evaluate_predictions(preds, gt, mean, std)
    return metrics, preds, model


def train_eval_dcrnn(data_prepared, graph_data, label, mean, std, epochs, dataset_name):
    """Train DCRNN with a given graph and return test metrics."""
    from src.models.dcrnn import DCRNN

    num_sensors  = data_prepared['splits']['train'][0].shape[2]
    num_supports = len(graph_data['diffusion_supports'])
    device       = config.DEVICE

    model = DCRNN(
        num_sensors=num_sensors,
        num_supports=num_supports,
        seq_len=config.SEQ_LEN,
        pred_len=config.PRED_LEN,
        hidden_dim=config.DCRNN_HIDDEN,
        num_layers=config.DCRNN_LAYERS,
    )

    print(f"\n  Training DCRNN [{label}] for {epochs} epochs ...")
    _config = config
    _config.EPOCHS = epochs

    history = train_model(
        model, data_prepared['loaders']['train'], data_prepared['loaders']['val'],
        _config, 'dcrnn', dataset_name, graph_data=graph_data,
    )

    preds, gt, _ = predict_model(
        model, data_prepared['loaders']['test'], _config, 'dcrnn', graph_data=graph_data,
    )
    metrics = evaluate_predictions(preds, gt, mean, std)
    return metrics, preds, model


# ─────────────────────────────────────────────────────────────────────────────
# Robustness probe at a fixed corruption ratio
# ─────────────────────────────────────────────────────────────────────────────

def robustness_probe(model, model_type, data_prepared, graph_data, mean, std,
                     ratio=0.2, fill_value=0.0):
    """Evaluate a model on a corrupted test set and return MAE."""
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    test_X_orig, test_Y = data_prepared['splits']['test']
    cx = inject_random_missing(test_X_orig, ratio=ratio, fill_value=fill_value)

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(cx), torch.FloatTensor(test_Y)),
        batch_size=config.BATCH_SIZE, shuffle=False,
    )
    preds, gt, _ = predict_model(model, loader, config, model_type, graph_data=graph_data)
    return evaluate_predictions(preds, gt, mean, std)['overall']['MAE']


# ─────────────────────────────────────────────────────────────────────────────
# Main ablation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_sparsity_ablation(dataset_name="METR-LA", epochs=30,
                          epsilons=None, models=("stgcn", "dcrnn")):

    epsilons = epsilons or [0.1, 0.2, 0.3, 0.5]

    print(f"\n{'='*65}")
    print(f"  GRAPH SPARSITY ABLATION — {dataset_name}")
    print(f"  Epsilon values: {epsilons}")
    print(f"  Epochs per run: {epochs}")
    print(f"{'='*65}")

    set_seed()
    config.get_device()
    config.EPOCHS = epochs

    # Prepare data once
    data_prepared = prepare_dataset(
        config.DATASETS[dataset_name]['path'],
        seq_len=config.SEQ_LEN,
        pred_len=config.PRED_LEN,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        batch_size=config.BATCH_SIZE,
    )
    mean      = data_prepared['mean']
    std       = data_prepared['std']
    train_raw = data_prepared['train_raw']

    all_results = {}

    for eps in epsilons:
        label = f"eps={eps}"
        print(f"\n{'─'*65}")
        print(f"  ε = {eps}")
        print(f"{'─'*65}")

        # Build graph at this epsilon
        graph_data = build_graph(
            train_raw,
            sigma=config.GRAPH_SIGMA,
            epsilon=eps,
            K_cheb=config.STGCN_K,
            K_diff=config.DIFFUSION_STEPS,
        )

        stats = graph_density_stats(graph_data['adj'])
        print(f"  Graph: {stats['num_edges']} edges | "
              f"avg {stats['avg_conn']} conn/node | "
              f"density {stats['density_pct']:.3f}%")

        entry = {"graph_stats": stats, "models": {}}

        # ── STGCN ────────────────────────────────────────────────────────────
        if "stgcn" in models:
            try:
                t0 = time.time()
                metrics, preds, model = train_eval_stgcn(
                    data_prepared, graph_data, label, mean, std, epochs, dataset_name
                )
                train_time = round(time.time() - t0, 1)
                rob_mae = robustness_probe(model, 'stgcn', data_prepared,
                                           graph_data, mean, std, ratio=0.2)
                entry["models"]["STGCN"] = {
                    "MAE_15min":   metrics['15min']['MAE'],
                    "MAE_30min":   metrics['30min']['MAE'],
                    "MAE_60min":   metrics['60min']['MAE'],
                    "MAE_overall": metrics['overall']['MAE'],
                    "RMSE":        metrics['overall']['RMSE'],
                    "robustness_20pct_MAE": round(rob_mae, 4),
                    "train_time_s": train_time,
                }
                print(f"  STGCN → overall MAE: {metrics['overall']['MAE']:.4f} | "
                      f"robust @20%: {rob_mae:.4f}")
            except Exception as e:
                print(f"  STGCN failed: {e}")
                import traceback; traceback.print_exc()

        # ── DCRNN ────────────────────────────────────────────────────────────
        if "dcrnn" in models:
            try:
                t0 = time.time()
                metrics, preds, model = train_eval_dcrnn(
                    data_prepared, graph_data, label, mean, std, epochs, dataset_name
                )
                train_time = round(time.time() - t0, 1)
                rob_mae = robustness_probe(model, 'dcrnn', data_prepared,
                                           graph_data, mean, std, ratio=0.2)
                entry["models"]["DCRNN"] = {
                    "MAE_15min":   metrics['15min']['MAE'],
                    "MAE_30min":   metrics['30min']['MAE'],
                    "MAE_60min":   metrics['60min']['MAE'],
                    "MAE_overall": metrics['overall']['MAE'],
                    "RMSE":        metrics['overall']['RMSE'],
                    "robustness_20pct_MAE": round(rob_mae, 4),
                    "train_time_s": train_time,
                }
                print(f"  DCRNN → overall MAE: {metrics['overall']['MAE']:.4f} | "
                      f"robust @20%: {rob_mae:.4f}")
            except Exception as e:
                print(f"  DCRNN failed: {e}")
                import traceback; traceback.print_exc()

        all_results[label] = entry

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(config.RESULTS_DIR, "metrics",
                             f"{dataset_name}_sparsity_ablation.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved → {save_path}")

    print_summary(all_results)
    return all_results


def print_summary(results):
    print(f"\n{'='*65}")
    print("  SPARSITY ABLATION SUMMARY")
    print(f"{'='*65}")
    for model_type in ["STGCN", "DCRNN"]:
        print(f"\n  {model_type}:")
        print(f"  {'Epsilon':<10} {'Avg Conn':>9} {'MAE(15m)':>10} "
              f"{'MAE(60m)':>10} {'Overall':>10} {'Rob@20%':>10}")
        print(f"  {'-'*60}")
        for label, entry in results.items():
            if model_type not in entry["models"]:
                continue
            m    = entry["models"][model_type]
            conn = entry["graph_stats"]["avg_conn"]
            eps  = label.split("=")[1]
            print(f"  {eps:<10} {conn:>9.1f} {m['MAE_15min']:>10.4f} "
                  f"{m['MAE_60min']:>10.4f} {m['MAE_overall']:>10.4f} "
                  f"{m['robustness_20pct_MAE']:>10.4f}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="METR-LA",
                        choices=["METR-LA", "PEMS-BAY"])
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs per training run (30 is fine for ablation)")
    parser.add_argument("--models", nargs="+", default=["stgcn", "dcrnn"],
                        choices=["stgcn", "dcrnn"])
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.5])
    args = parser.parse_args()

    run_sparsity_ablation(
        dataset_name=args.dataset,
        epochs=args.epochs,
        epsilons=args.epsilons,
        models=args.models,
    )
