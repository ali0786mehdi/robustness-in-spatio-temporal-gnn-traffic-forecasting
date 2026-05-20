"""
Master script: Run ALL models (baselines + GNNs), compare, and generate plots.
Usage: python run_all.py [--dataset METR-LA|PEMS-BAY|both]
"""

import sys
import os
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.config import set_seed
from src.data_loader import prepare_dataset
from src.graph_builder import build_graph
from src.evaluate import (
    evaluate_predictions, print_results, save_results, compare_models, save_efficiency
)
from src.train import train_model, predict_model
from src.visualize import generate_all_plots
from run_baselines import run_sanity


def run_full_pipeline(dataset_name):
    """
    Run ALL models on a single dataset, compare, and generate plots.
    """
    print(f"\n{'#'*70}")
    print(f"  FULL PIPELINE — {dataset_name}")
    print(f"{'#'*70}\n")

    # ====== Step 1: Prepare data ======
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
    num_sensors = splits['train'][0].shape[2]

    # ====== Step 2: Build graph ======
    graph_data = build_graph(
        data_prepared['train_raw'],
        sigma=config.GRAPH_SIGMA,
        epsilon=config.GRAPH_EPSILON,
        K_cheb=config.STGCN_K,
        K_diff=config.DIFFUSION_STEPS,
    )

    all_results = {}
    all_preds = {}
    histories = {}
    test_Y = splits['test'][1]  # Ground truth for all models

    # ====== Step 3: Traditional Baselines ======

    # --- Sanity Baselines ---
    try:
        s_results, s_preds = run_sanity(data_prepared, dataset_name, mean, std)
        all_results.update(s_results)
        all_preds.update(s_preds)
    except Exception as e:
        print(f"Sanity baselines failed: {e}")
        import traceback; traceback.print_exc()

    # --- ARIMA ---
    try:
        from src.models.arima_model import ARIMAForecaster
        print("\n>>> Running ARIMA...")
        train_X = splits['train'][0]
        T_train = len(train_X) + config.SEQ_LEN - 1
        train_series = np.zeros((T_train, num_sensors))
        train_series[:config.SEQ_LEN] = train_X[0]
        for i in range(1, len(train_X)):
            train_series[config.SEQ_LEN + i - 1] = train_X[i, -1]

        arima = ARIMAForecaster(order=config.ARIMA_ORDER,
                                max_sensors=config.ARIMA_MAX_SENSORS)
        preds = arima.fit_and_predict(train_series, splits['test'][0], config.PRED_LEN)
        results = evaluate_predictions(preds, test_Y, mean, std)
        print_results(results, arima.get_name())
        all_results['ARIMA'] = results
        all_preds['ARIMA'] = preds
    except Exception as e:
        print(f"  ARIMA failed: {e}")

    # --- Random Forest ---
    try:
        from src.models.rf_model import RandomForestForecaster
        print("\n>>> Running Random Forest...")
        rf = RandomForestForecaster(n_estimators=config.RF_N_ESTIMATORS,
                                    max_depth=config.RF_MAX_DEPTH,
                                    n_jobs=config.RF_N_JOBS)
        rf.fit(splits['train'][0], splits['train'][1])
        preds = rf.predict(splits['test'][0])
        results = evaluate_predictions(preds, test_Y, mean, std)
        print_results(results, rf.get_name())
        all_results['RandomForest'] = results
        all_preds['RandomForest'] = preds
    except Exception as e:
        print(f"  Random Forest failed: {e}")

    # --- LSTM ---
    try:
        from src.models.lstm_model import LSTMModel
        print("\n>>> Running LSTM...")
        lstm = LSTMModel(num_sensors=num_sensors, seq_len=config.SEQ_LEN,
                         pred_len=config.PRED_LEN, hidden_dim=config.LSTM_HIDDEN,
                         num_layers=config.LSTM_LAYERS, dropout=config.LSTM_DROPOUT)
        params = sum(p.numel() for p in lstm.parameters())
        print(f"  LSTM params: {params:,}")

        history = train_model(lstm, data_prepared['loaders']['train'],
                              data_prepared['loaders']['val'], config,
                              'lstm', dataset_name)
        preds, gt, latency = predict_model(lstm, data_prepared['loaders']['test'],
                                  config, 'lstm')
        history['efficiency']['inference_latency_ms'] = latency
        results = evaluate_predictions(preds, gt, mean, std)
        print_results(results, lstm.get_name())
        all_results['LSTM'] = results
        all_preds['LSTM'] = preds
        histories['LSTM'] = history
    except Exception as e:
        print(f"  LSTM failed: {e}")

    # ====== Step 4: GNN Models ======

    # --- STGCN ---
    try:
        from src.models.stgcn import STGCN
        print("\n>>> Running STGCN...")
        stgcn = STGCN(num_sensors=num_sensors, seq_len=config.SEQ_LEN,
                       pred_len=config.PRED_LEN, K=config.STGCN_K,
                       channels=config.STGCN_CHANNELS)
        params = sum(p.numel() for p in stgcn.parameters())
        print(f"  STGCN params: {params:,}")

        history = train_model(stgcn, data_prepared['loaders']['train'],
                              data_prepared['loaders']['val'], config,
                              'stgcn', dataset_name, graph_data=graph_data)
        preds, gt, latency = predict_model(stgcn, data_prepared['loaders']['test'],
                                  config, 'stgcn', graph_data=graph_data)
        history['efficiency']['inference_latency_ms'] = latency
        results = evaluate_predictions(preds, gt, mean, std)
        print_results(results, stgcn.get_name())
        all_results['STGCN'] = results
        all_preds['STGCN'] = preds
        histories['STGCN'] = history
    except Exception as e:
        print(f"  STGCN failed: {e}")
        import traceback; traceback.print_exc()

    # --- DCRNN ---
    try:
        from src.models.dcrnn import DCRNN
        print("\n>>> Running DCRNN...")
        num_supports = len(graph_data['diffusion_supports'])
        dcrnn = DCRNN(num_sensors=num_sensors, num_supports=num_supports,
                       seq_len=config.SEQ_LEN, pred_len=config.PRED_LEN,
                       hidden_dim=config.DCRNN_HIDDEN,
                       num_layers=config.DCRNN_LAYERS)
        params = sum(p.numel() for p in dcrnn.parameters())
        print(f"  DCRNN params: {params:,}")

        history = train_model(dcrnn, data_prepared['loaders']['train'],
                              data_prepared['loaders']['val'], config,
                              'dcrnn', dataset_name, graph_data=graph_data)
        preds, gt, latency = predict_model(dcrnn, data_prepared['loaders']['test'],
                                  config, 'dcrnn', graph_data=graph_data)
        history['efficiency']['inference_latency_ms'] = latency
        results = evaluate_predictions(preds, gt, mean, std)
        print_results(results, dcrnn.get_name())
        all_results['DCRNN'] = results
        all_preds['DCRNN'] = preds
        histories['DCRNN'] = history
    except Exception as e:
        print(f"  DCRNN failed: {e}")
        import traceback; traceback.print_exc()

    # ====== Step 5: Compare & Visualize ======
    compare_models(all_results, dataset_name)

    # Save all results
    metrics_dir = os.path.join(config.RESULTS_DIR, 'metrics')
    save_results(all_results, metrics_dir, dataset_name + '_all')
    save_efficiency(histories, metrics_dir, dataset_name + '_all')

    # Generate plots
    generate_all_plots(
        all_results=all_results,
        preds_dict=all_preds,
        ground_truth=test_Y,
        mean=mean,
        std=std,
        adj=graph_data['adj'],
        histories=histories,
        dataset_name=dataset_name,
        save_dir=os.path.join(config.RESULTS_DIR, 'plots'),
    )

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run all models')
    parser.add_argument('--dataset', type=str, default='METR-LA',
                        choices=['METR-LA', 'PEMS-BAY', 'both'])
    args = parser.parse_args()

    set_seed()
    config.get_device()

    datasets = ['METR-LA', 'PEMS-BAY'] if args.dataset == 'both' else [args.dataset]

    final_results = {}
    for ds in datasets:
        final_results[ds] = run_full_pipeline(ds)

    # Print grand summary
    print(f"\n{'='*70}")
    print(f"  GRAND SUMMARY")
    print(f"{'='*70}")
    for ds, results in final_results.items():
        print(f"\n  {ds}:")
        for model_name, r in results.items():
            if 'overall' in r:
                print(f"    {model_name:<20} MAE={r['overall']['MAE']:.4f}  "
                      f"RMSE={r['overall']['RMSE']:.4f}  "
                      f"MAPE={r['overall']['MAPE']:.2f}%")

    print(f"\nAll results saved to {config.RESULTS_DIR}/")
    print("Done!")


if __name__ == '__main__':
    main()
