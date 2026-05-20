"""
Run robustness experiments on ALL models.
Evaluates performance under varying levels of missing data and sensor failures.

Models tested:
  - Persistence          (sanity baseline, no training required)
  - Historical Average   (sanity baseline, no training required)
  - Random Forest        (classical baseline, loaded from disk)
  - LSTM                 (deep temporal, loaded from disk)
  - STGCN               (graph neural network, loaded from disk)
  - DCRNN               (graph neural network, loaded from disk)
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


def load_arima(dataset_name, pred_len):
    """Load a pre-fitted ARIMA model from disk."""
    from src.models.arima_model import ARIMAForecaster
    save_path = os.path.join(config.RESULTS_DIR, "models", f"arima_{dataset_name}_best.pkl")
    if not os.path.exists(save_path):
        print(f"  [skip] ARIMA checkpoint not found: {save_path}")
        print(f"         Run 'python run_baselines.py --dataset {dataset_name}' first.")
        return None
    model = ARIMAForecaster()
    model.load(save_path)
    if model.pred_len is None:
        model.pred_len = pred_len
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_deep_model(model_name, dataset_name, num_sensors, graph_data=None):
    """Load a trained deep learning model (LSTM / STGCN / DCRNN) from disk."""
    device = config.DEVICE
    save_path = os.path.join(config.RESULTS_DIR, "models", f"{model_name}_{dataset_name}_best.pt")

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

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def load_rf(dataset_name):
    """Load a trained Random Forest model from disk."""
    from src.models.rf_model import RandomForestForecaster
    save_path = os.path.join(config.RESULTS_DIR, "models", f"rf_{dataset_name}_best.pkl")
    if not os.path.exists(save_path):
        print(f"  [skip] Random Forest checkpoint not found: {save_path}")
        return None
    model = RandomForestForecaster(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        n_jobs=config.RF_N_JOBS,
    )
    model.load(save_path)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_robustness(dataset_name="METR-LA"):
    print(f"\n{'='*62}")
    print(f"  ROBUSTNESS EXPERIMENTS (ALL MODELS) — {dataset_name}")
    print(f"{'='*62}")

    set_seed()
    filepath = config.DATASETS[dataset_name]['path']

    data_prepared = prepare_dataset(
        filepath,
        seq_len=config.SEQ_LEN,
        pred_len=config.PRED_LEN,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        batch_size=config.BATCH_SIZE,
    )

    mean      = data_prepared['mean']
    std       = data_prepared['std']
    train_raw = data_prepared['train_raw']
    num_sensors = data_prepared['splits']['train'][0].shape[2]

    # Build graph (STGCN / DCRNN)
    graph_data = build_graph(
        train_raw,
        sigma=config.GRAPH_SIGMA,
        epsilon=config.GRAPH_EPSILON,
        K_cheb=config.STGCN_K,
        K_diff=config.DIFFUSION_STEPS,
    )

    # ── Prepare timestamps for Historical Average ──
    T = len(data_prepared['raw_data'])
    train_end = int(T * config.TRAIN_RATIO)
    val_end   = int(T * (config.TRAIN_RATIO + config.VAL_RATIO))
    test_start_idx = val_end + config.SEQ_LEN

    ha_model = HistoricalAverageModel()
    ha_model.fit(train_raw, data_prepared['timestamps'][:train_end])

    test_X_orig = data_prepared['splits']['test'][0]
    test_Y      = data_prepared['splits']['test'][1]
    n_test      = len(test_Y)
    test_timestamps = data_prepared['timestamps'][test_start_idx : test_start_idx + n_test]

    # ── Load deep models ──
    from torch.utils.data import DataLoader, TensorDataset
    deep_models = {}
    for m in ['lstm', 'stgcn', 'dcrnn']:
        loaded = load_deep_model(m, dataset_name, num_sensors, graph_data)
        if loaded is not None:
            deep_models[m] = loaded

    rf_model   = load_rf(dataset_name)
    arima_model = load_arima(dataset_name, config.PRED_LEN)

    fill_value = 0.0
    ratios     = [0.0, 0.1, 0.2, 0.3, 0.4]
    scenarios  = {
        'random_missing': inject_random_missing,
        'sensor_failure': inject_sensor_failure,
    }

    results = {s: {r: {} for r in ratios} for s in scenarios}

    for scenario_name, inject_func in scenarios.items():
        print(f"\n{'─'*50}")
        print(f"  Scenario: {scenario_name.replace('_', ' ').title()}")
        print(f"{'─'*50}")

        for ratio in ratios:
            print(f"\n  Ratio: {ratio:.0%}")

            # ── Apply corruption ──────────────────────────────
            if ratio == 0.0:
                cx = test_X_orig.copy()
            else:
                cx = inject_func(test_X_orig, ratio=ratio, fill_value=fill_value)

            # ── 1. Persistence ────────────────────────────────
            pm   = PersistenceModel(pred_len=config.PRED_LEN)
            cx_dn = cx * std + mean                       # denormalise corrupted input
            p_dn  = pm.predict(cx_dn)
            p_pm  = (p_dn - mean) / std                   # re-normalise for evaluate
            mae   = evaluate_predictions(p_pm, test_Y, mean, std)['overall']['MAE']
            results[scenario_name][ratio]['Persistence'] = mae
            print(f"    {'Persistence':<20} MAE: {mae:.4f}")

            # ── 2. Historical Average ─────────────────────────
            p_ha_dn = ha_model.predict(test_timestamps, num_sensors, config.PRED_LEN)
            p_ha    = (p_ha_dn - mean) / std
            mae     = evaluate_predictions(p_ha, test_Y, mean, std)['overall']['MAE']
            results[scenario_name][ratio]['HistoricalAverage'] = mae
            print(f"    {'HistoricalAverage':<20} MAE: {mae:.4f}")
            # Note: HA is independent of input sequences, so it is immune to input corruption.
            # Its score is constant across ratios (expected behaviour — acts as a floor).

            # ── 3. ARIMA ─────────────────────────────────────
            if arima_model is not None:
                p_arima = arima_model.predict(cx)
                mae     = evaluate_predictions(p_arima, test_Y, mean, std)['overall']['MAE']
                results[scenario_name][ratio]['ARIMA'] = mae
                print(f"    {'ARIMA':<20} MAE: {mae:.4f}")

            # ── 4. Random Forest ──────────────────────────────

            if rf_model is not None:
                p_rf = rf_model.predict(cx)
                mae  = evaluate_predictions(p_rf, test_Y, mean, std)['overall']['MAE']
                results[scenario_name][ratio]['RandomForest'] = mae
                print(f"    {'RandomForest':<20} MAE: {mae:.4f}")

            # ── 4–6. Deep models (LSTM / STGCN / DCRNN) ──────
            test_loader = DataLoader(
                TensorDataset(torch.FloatTensor(cx), torch.FloatTensor(test_Y)),
                batch_size=config.BATCH_SIZE, shuffle=False,
            )
            for m_name, model in deep_models.items():
                preds, gt, _ = predict_model(model, test_loader, config, m_name, graph_data)
                mae = evaluate_predictions(preds, gt, mean, std)['overall']['MAE']
                results[scenario_name][ratio][m_name.upper()] = mae
                print(f"    {m_name.upper():<20} MAE: {mae:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(config.RESULTS_DIR, "metrics", f"{dataset_name}_robustness.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRobustness results saved to {save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='METR-LA',
                        choices=['METR-LA', 'PEMS-BAY'])
    args = parser.parse_args()
    evaluate_robustness(args.dataset)
