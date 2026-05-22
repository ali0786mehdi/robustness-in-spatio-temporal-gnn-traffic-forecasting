"""
Run robustness experiments on ALL models with multi-seed injection.

For each (scenario, ratio) combination, corruption is injected N times
with different random seeds. Results are reported as mean ± std,
making them statistically defensible for academic publication.

Models tested:
  - Persistence          (sanity baseline)
  - Historical Average   (sanity baseline, immune to input corruption)
  - ARIMA                (classical temporal, loaded from disk)
  - Random Forest        (classical, loaded from disk)
  - LSTM                 (deep temporal, loaded from disk)
  - STGCN               (graph neural network, loaded from disk)
  - DCRNN               (graph neural network, loaded from disk)

Usage:
  python run_robustness.py                      # 5 seeds (default)
  python run_robustness.py --n-seeds 3          # faster
  python run_robustness.py --dataset PEMS-BAY
"""

import sys
import os
import json
import torch
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.config import set_seed
from src.data_loader import prepare_dataset
from src.graph_builder import build_graph
from src.evaluate import evaluate_predictions
from src.robustness import inject_random_missing, inject_sensor_failure
from src.train import predict_model
from src.models.sanity_baselines import PersistenceModel, HistoricalAverageModel


# ─────────────────────────────────────────────────────────────────────────────
# Model Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_arima(dataset_name, pred_len):
    from src.models.arima_model import ARIMAForecaster
    save_path = config.get_model_path('arima', dataset_name)
    if not os.path.exists(save_path):
        print(f"  [skip] ARIMA checkpoint not found: {save_path}")
        return None
    model = ARIMAForecaster()
    model.load(save_path)
    if model.pred_len is None:
        model.pred_len = pred_len
    return model


def load_rf(dataset_name):
    from src.models.rf_model import RandomForestForecaster
    save_path = config.get_model_path('rf', dataset_name)
    if not os.path.exists(save_path):
        print(f"  [skip] Random Forest checkpoint not found: {save_path}")
        return None
    model = RandomForestForecaster()
    model.load(save_path)
    return model


def load_deep_model(model_name, dataset_name, num_sensors, graph_data=None):
    device    = config.DEVICE
    save_path = config.get_model_path(model_name, dataset_name)
    if not os.path.exists(save_path):
        print(f"  [skip] {model_name.upper()} checkpoint not found: {save_path}")
        return None

    if model_name == 'lstm':
        from src.models.lstm_model import LSTMModel
        model = LSTMModel(
            num_sensors=num_sensors,
            seq_len=config.SEQ_LEN,
            pred_len=config.PRED_LEN,
            hidden_dim=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            dropout=config.LSTM_DROPOUT,
        )
    elif model_name == 'stgcn':
        from src.models.stgcn import STGCN
        model = STGCN(
            num_sensors=num_sensors,
            seq_len=config.SEQ_LEN,
            pred_len=config.PRED_LEN,
            K=config.STGCN_K,
            channels=config.STGCN_CHANNELS,
        )
    elif model_name == 'dcrnn':
        from src.models.dcrnn import DCRNN
        num_supports = len(graph_data['diffusion_supports'])
        model = DCRNN(
            num_sensors=num_sensors,
            num_supports=num_supports,
            seq_len=config.SEQ_LEN,
            pred_len=config.PRED_LEN,
            hidden_dim=config.DCRNN_HIDDEN,
            num_layers=config.DCRNN_LAYERS,
        )
    else:
        return None

    state = torch.load(save_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  Loaded {model_name.upper()} from {save_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def eval_with_seeds(eval_fn, seeds, ratio_label):
    """
    Call eval_fn(seed) for each seed. Returns {"mean": float, "std": float, "n": int}.
    eval_fn must return a scalar MAE.
    """
    maes = []
    for seed in seeds:
        np.random.seed(seed)
        mae = eval_fn(seed)
        maes.append(mae)
    n = len(maes)
    return {
        "mean": round(float(np.mean(maes)), 4),
        "std":  round(float(np.std(maes, ddof=1)) if n > 1 else 0.0, 4),
        "n":    n,
        "all":  [round(m, 4) for m in maes],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_robustness(dataset_name="METR-LA", n_seeds=5):
    print(f"\n{'='*62}")
    print(f"  ROBUSTNESS EXPERIMENTS (ALL MODELS) — {dataset_name}")
    print(f"  Seeds per ratio: {n_seeds}  →  results reported as mean ± std")
    print(f"{'='*62}")

    seeds = list(range(n_seeds))       # [0, 1, 2, ... n_seeds-1]

    set_seed(42)
    config.get_device()

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

    graph_data = build_graph(
        train_raw,
        sigma=config.GRAPH_SIGMA,
        epsilon=config.GRAPH_EPSILON,
        K_cheb=config.STGCN_K,
        K_diff=config.DIFFUSION_STEPS,
    )

    # ── Timestamps for Historical Average ─────────────────────────────────────
    T          = len(data_prepared['raw_data'])
    train_end  = int(T * config.TRAIN_RATIO)
    val_end    = int(T * (config.TRAIN_RATIO + config.VAL_RATIO))
    test_start = val_end + config.SEQ_LEN

    ha_model = HistoricalAverageModel()
    ha_model.fit(train_raw, data_prepared['timestamps'][:train_end])

    test_X_orig     = data_prepared['splits']['test'][0]
    test_Y          = data_prepared['splits']['test'][1]
    n_test          = len(test_Y)
    test_timestamps = data_prepared['timestamps'][test_start : test_start + n_test]
    num_sensors     = test_X_orig.shape[2]

    # ── Load all models ───────────────────────────────────────────────────────
    from torch.utils.data import DataLoader, TensorDataset

    deep_models = {}
    for m in ['lstm', 'stgcn', 'dcrnn']:
        loaded = load_deep_model(m, dataset_name, num_sensors, graph_data)
        if loaded is not None:
            deep_models[m] = loaded

    rf_model    = load_rf(dataset_name)
    arima_model = load_arima(dataset_name, config.PRED_LEN)

    fill_value = 0.0
    ratios     = [0.0, 0.1, 0.2, 0.3, 0.4]
    scenarios  = {
        'random_missing': inject_random_missing,
        'sensor_failure': inject_sensor_failure,
    }

    results = {s: {} for s in scenarios}

    for scenario_name, inject_func in scenarios.items():
        print(f"\n{'─'*50}")
        print(f"  Scenario: {scenario_name.replace('_', ' ').title()}")
        print(f"{'─'*50}")

        for ratio in ratios:
            print(f"\n  Ratio: {ratio:.0%}")
            results[scenario_name][ratio] = {}

            # ── Helper: get a corrupted input for a given seed ────────────────
            def make_cx(seed):
                if ratio == 0.0:
                    return test_X_orig.copy()
                np.random.seed(seed)
                return inject_func(test_X_orig, ratio=ratio, fill_value=fill_value)

            # ── 1. Persistence ────────────────────────────────────────────────
            def eval_persistence(seed):
                cx    = make_cx(seed)
                cx_dn = cx * std + mean
                pm    = PersistenceModel(pred_len=config.PRED_LEN)
                p_dn  = pm.predict(cx_dn)
                p_norm = (p_dn - mean) / std
                return evaluate_predictions(p_norm, test_Y, mean, std)['overall']['MAE']

            res = eval_with_seeds(eval_persistence, seeds if ratio > 0 else [0], ratio)
            results[scenario_name][ratio]['Persistence'] = res
            print(f"    {'Persistence':<20} MAE: {res['mean']:.4f} ± {res['std']:.4f}")

            # ── 2. Historical Average (input-independent, 1 seed sufficient) ──
            p_ha_dn = ha_model.predict(test_timestamps, num_sensors, config.PRED_LEN)
            p_ha    = (p_ha_dn - mean) / std
            mae_ha  = evaluate_predictions(p_ha, test_Y, mean, std)['overall']['MAE']
            results[scenario_name][ratio]['HistoricalAverage'] = {
                "mean": round(mae_ha, 4), "std": 0.0, "n": 1, "all": [round(mae_ha, 4)]
            }
            print(f"    {'HistoricalAverage':<20} MAE: {mae_ha:.4f} ± 0.0000  (input-independent)")

            # ── 3. ARIMA ──────────────────────────────────────────────────────
            if arima_model is not None:
                def eval_arima(seed):
                    cx = make_cx(seed)
                    p  = arima_model.predict(cx)
                    return evaluate_predictions(p, test_Y, mean, std)['overall']['MAE']
                res = eval_with_seeds(eval_arima, seeds if ratio > 0 else [0], ratio)
                results[scenario_name][ratio]['ARIMA'] = res
                print(f"    {'ARIMA':<20} MAE: {res['mean']:.4f} ± {res['std']:.4f}")

            # ── 4. Random Forest ──────────────────────────────────────────────
            if rf_model is not None:
                def eval_rf(seed):
                    cx = make_cx(seed)
                    p  = rf_model.predict(cx)
                    return evaluate_predictions(p, test_Y, mean, std)['overall']['MAE']
                res = eval_with_seeds(eval_rf, seeds if ratio > 0 else [0], ratio)
                results[scenario_name][ratio]['RandomForest'] = res
                print(f"    {'RandomForest':<20} MAE: {res['mean']:.4f} ± {res['std']:.4f}")

            # ── 5–7. Deep models ──────────────────────────────────────────────
            for m_name, model in deep_models.items():
                def eval_deep(seed, _m=m_name, _model=model):
                    cx = make_cx(seed)
                    loader = DataLoader(
                        TensorDataset(torch.FloatTensor(cx), torch.FloatTensor(test_Y)),
                        batch_size=config.BATCH_SIZE, shuffle=False,
                    )
                    preds, gt, _ = predict_model(_model, loader, config, _m, graph_data)
                    return evaluate_predictions(preds, gt, mean, std)['overall']['MAE']

                res = eval_with_seeds(eval_deep, seeds if ratio > 0 else [0], ratio)
                results[scenario_name][ratio][m_name.upper()] = res
                print(f"    {m_name.upper():<20} MAE: {res['mean']:.4f} ± {res['std']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    # Convert ratio keys to strings for JSON serialisation
    json_results = {}
    for scen, ratios_dict in results.items():
        json_results[scen] = {}
        for ratio, model_dict in ratios_dict.items():
            json_results[scen][str(ratio)] = model_dict

    save_path = os.path.join(config.RESULTS_DIR, "metrics",
                             f"{dataset_name}_robustness.json")
    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n✓ Robustness results saved to {save_path}")
    print_summary(json_results)
    return json_results


def print_summary(results):
    """Print mean ± std degradation table."""
    print(f"\n{'='*68}")
    print("  ROBUSTNESS SUMMARY  (mean ± std over seeds)")
    print(f"{'='*68}")
    for scenario, data in results.items():
        ratio_keys = list(data.keys())
        worst_key  = ratio_keys[-1]
        worst_pct  = int(float(worst_key) * 100)
        print(f"\n  Scenario: {scenario.replace('_', ' ').title()}")
        available = list(data[ratio_keys[0]].keys())
        col = max(len(m) for m in available) + 2
        print(f"  {'Model':<{col}}  {'MAE 0%':>10}  {'MAE {:.0f}%'.format(worst_pct):>14}  {'Δ':>8}  {'Deg %':>7}")
        print(f"  {'-'*(col+48)}")
        for model in available:
            base_m  = data["0.0"][model]['mean']
            worst_m = data[worst_key][model]['mean']
            worst_s = data[worst_key][model]['std']
            delta   = worst_m - base_m
            pct     = delta / base_m * 100
            worst_str = f"{worst_m:.3f}±{worst_s:.3f}"
            print(f"  {model:<{col}}  {base_m:>10.4f}  {worst_str:>14}  {delta:>+8.3f}  {pct:>6.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='METR-LA',
                        choices=['METR-LA', 'PEMS-BAY'])
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Number of random seeds per corruption ratio (default: 5)')
    args = parser.parse_args()
    evaluate_robustness(args.dataset, n_seeds=args.n_seeds)
