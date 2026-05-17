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
from src.evaluate import evaluate_predictions, print_results, save_results
from src.train import train_model, predict_model


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

    predictions, ground_truth = predict_model(
        model, data_prepared['loaders']['test'], config, 'lstm',
    )

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
    save_results(all_results, os.path.join(config.RESULTS_DIR, 'metrics'), dataset_name + '_baselines')

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
