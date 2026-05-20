"""
Run traditional baseline models: ARIMA, Random Forest, LSTM.
Usage: python run_baselines.py [--dataset METR-LA|PEMS-BAY|both]
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.config import set_seed
from src.data_loader import prepare_dataset
from src.graph_builder import build_graph
from src.evaluate import evaluate_predictions, print_results, save_results, save_efficiency
from src.train import train_model, predict_model
from src.models.sanity_baselines import HistoricalAverageModel, PersistenceModel

def run_sanity(data_prepared, dataset_name, mean, std):
    """Run Persistence and Historical Average baselines."""
    print(f"\n{'='*60}")
    print(f"  SANITY BASELINES — {dataset_name}")
    print(f"{'='*60}")
    
    test_X = data_prepared['splits']['test'][0]
    test_Y = data_prepared['splits']['test'][1]
    
    # 1. Persistence
    print("\n  Running Persistence (Copy-Last-Value)...")
    pm = PersistenceModel(pred_len=config.PRED_LEN)
    # Persistence takes denormalized input
    test_X_dn = test_X * std + mean
    preds_pm_dn = pm.predict(test_X_dn)
    # We must evaluate using normalized space predictions or pass denormalized to evaluate
    # evaluate_predictions automatically denormalizes, so we must feed it normalized preds
    preds_pm = (preds_pm_dn - mean) / std
    res_pm = evaluate_predictions(preds_pm, test_Y, mean, std)
    print_results(res_pm, 'Persistence')
    
    # 2. Historical Average
    print("\n  Running Historical Average...")
    ha = HistoricalAverageModel()
    ha.fit(data_prepared['train_raw'], data_prepared['timestamps'][:len(data_prepared['train_raw'])])
    
    # We need the timestamps corresponding to the start of each test sequence
    # test_X starts at index len(train) + len(val)
    T = len(data_prepared['raw_data'])
    train_end = int(T * config.TRAIN_RATIO)
    val_end = int(T * (config.TRAIN_RATIO + config.VAL_RATIO))
    
    # test_raw starts at val_end. 
    # test sequences X starts at val_end. Y starts at val_end + seq_len.
    test_start_idx = val_end + config.SEQ_LEN
    test_timestamps = data_prepared['timestamps'][test_start_idx : test_start_idx + len(test_Y)]
    
    num_sensors = test_X.shape[2]
    preds_ha_dn = ha.predict(test_timestamps, num_sensors, config.PRED_LEN)
    preds_ha = (preds_ha_dn - mean) / std
    res_ha = evaluate_predictions(preds_ha, test_Y, mean, std)
    print_results(res_ha, 'HistoricalAverage')
    
    return {'Persistence': res_pm, 'HistoricalAverage': res_ha}, {'Persistence': preds_pm, 'HistoricalAverage': preds_ha}



def run_arima(splits, dataset_name, mean, std):
    """Run ARIMA baseline."""
    from src.models.arima_model import ARIMAForecaster

    print(f"\n{'='*60}")
    print(f"  ARIMA — {dataset_name}")
    print(f"{'='*60}")

    train_X, train_Y = splits['train']
    test_X, test_Y = splits['test']

    # ARIMA needs the raw training time series (not sequences)
    # Reconstruct from first sequence + subsequent first steps
    T_train = len(train_X) + config.SEQ_LEN - 1
    num_sensors = train_X.shape[2]

    # Build full training series from sequences
    train_series = np.zeros((T_train, num_sensors))
    train_series[:config.SEQ_LEN] = train_X[0]
    for i in range(1, len(train_X)):
        train_series[config.SEQ_LEN + i - 1] = train_X[i, -1]

    model = ARIMAForecaster(
        order=config.ARIMA_ORDER,
        max_sensors=config.ARIMA_MAX_SENSORS,
    )

    predictions = model.fit_and_predict(
        train_series, test_X, pred_len=config.PRED_LEN
    )

    results = evaluate_predictions(predictions, test_Y, mean, std)
    print_results(results, model.get_name())
    return results, predictions


def run_random_forest(splits, dataset_name, mean, std):
    """Run Random Forest baseline."""
    from src.models.rf_model import RandomForestForecaster

    print(f"\n{'='*60}")
    print(f"  Random Forest — {dataset_name}")
    print(f"{'='*60}")

    train_X, train_Y = splits['train']
    test_X, test_Y = splits['test']

    model = RandomForestForecaster(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        n_jobs=config.RF_N_JOBS,
    )

    model.fit(train_X, train_Y)
    
    save_path = os.path.join(config.RESULTS_DIR, "models", f"rf_{dataset_name}_best.pkl")
    model.save(save_path)
    print(f"  Saved RF model to {save_path}")
    
    predictions = model.predict(test_X)

    results = evaluate_predictions(predictions, test_Y, mean, std)
    print_results(results, model.get_name())
    return results, predictions


def run_lstm(data_prepared, dataset_name, mean, std):
    """Run LSTM baseline."""
    from src.models.lstm_model import LSTMModel

    print(f"\n{'='*60}")
    print(f"  LSTM — {dataset_name}")
    print(f"{'='*60}")

    num_sensors = data_prepared['splits']['train'][0].shape[2]

    model = LSTMModel(
        num_sensors=num_sensors,
        seq_len=config.SEQ_LEN,
        pred_len=config.PRED_LEN,
        hidden_dim=config.LSTM_HIDDEN,
        num_layers=config.LSTM_LAYERS,
        dropout=config.LSTM_DROPOUT,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    history = train_model(
        model, data_prepared['loaders']['train'],
        data_prepared['loaders']['val'],
        config, 'lstm', dataset_name,
    )

    predictions, ground_truth, latency = predict_model(
        model, data_prepared['loaders']['test'], config, 'lstm',
    )
    history['efficiency']['inference_latency_ms'] = latency

    results = evaluate_predictions(predictions, ground_truth, mean, std)
    print_results(results, model.get_name())
    return results, predictions, history


def run_baselines_on_dataset(dataset_name):
    """Run all baselines on a single dataset."""
    print(f"\n{'#'*60}")
    print(f"  BASELINES — {dataset_name}")
    print(f"{'#'*60}\n")

    filepath = config.DATASETS[dataset_name]['path']

    data_prepared = prepare_dataset(
        filepath,
        seq_len=config.SEQ_LEN,
        pred_len=config.PRED_LEN,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        batch_size=config.BATCH_SIZE,
    )

    mean = data_prepared['mean']
    std = data_prepared['std']
    splits = data_prepared['splits']

    all_results = {}
    all_preds = {}
    histories = {}

    # 0. Sanity Baselines
    try:
        s_results, s_preds = run_sanity(data_prepared, dataset_name, mean, std)
        all_results.update(s_results)
        all_preds.update(s_preds)
    except Exception as e:
        print(f"  Sanity baselines failed: {e}")

    # 1. ARIMA
    try:
        results, preds = run_arima(splits, dataset_name, mean, std)
        all_results['ARIMA'] = results
        all_preds['ARIMA'] = preds
    except Exception as e:
        print(f"  ARIMA failed: {e}")

    # 2. Random Forest
    try:
        results, preds = run_random_forest(splits, dataset_name, mean, std)
        all_results['RandomForest'] = results
        all_preds['RandomForest'] = preds
    except Exception as e:
        print(f"  Random Forest failed: {e}")

    # 3. LSTM
    try:
        results, preds, history = run_lstm(data_prepared, dataset_name, mean, std)
        all_results['LSTM'] = results
        all_preds['LSTM'] = preds
        histories['LSTM'] = history
    except Exception as e:
        print(f"  LSTM failed: {e}")

    # Save results
    metrics_dir = os.path.join(config.RESULTS_DIR, 'metrics')
    save_results(all_results, metrics_dir, dataset_name + '_baselines')
    save_efficiency(histories, metrics_dir, dataset_name + '_baselines')

    return all_results, all_preds, histories, data_prepared


def main():
    parser = argparse.ArgumentParser(description='Run baseline models')
    parser.add_argument('--dataset', type=str, default='METR-LA',
                        choices=['METR-LA', 'PEMS-BAY', 'both'])
    args = parser.parse_args()

    set_seed()
    device = config.get_device()

    datasets = ['METR-LA', 'PEMS-BAY'] if args.dataset == 'both' else [args.dataset]

    for ds in datasets:
        run_baselines_on_dataset(ds)


if __name__ == '__main__':
    main()
