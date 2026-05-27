"""
Run cross-dataset generalization experiments.
Evaluates domain-shift robustness by testing temporal models trained on one dataset
(e.g., PEMS-BAY) against another (e.g., METR-LA).
Note: Only temporal models (LSTM, Random Forest) support direct transfer.
      Spatial GNNs are bound to their specific training graph structures.
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
from src.evaluate import evaluate_predictions, print_results
from src.train import predict_model


def load_temporal_model(model_name, source_dataset, num_sensors):
    """Load a trained temporal model."""
    if model_name == 'lstm':
        save_path = config.get_model_path('lstm', source_dataset)
        if not os.path.exists(save_path):
            return None
            
        from src.models.lstm_model import LSTMModel
        model = LSTMModel(
            num_sensors=num_sensors,
            seq_len=config.SEQ_LEN,
            pred_len=config.PRED_LEN,
            hidden_dim=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            dropout=config.LSTM_DROPOUT,
        )
        model.load_state_dict(torch.load(save_path, map_location=config.DEVICE, weights_only=True))
        model.to(config.DEVICE)
        model.eval()
        return model
        
    elif model_name == 'rf':
        save_path = config.get_model_path('rf', source_dataset)
        if not os.path.exists(save_path):
            return None
            
        from src.models.rf_model import RandomForestForecaster
        model = RandomForestForecaster(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            n_jobs=config.RF_N_JOBS,
        )
        model.load(save_path)
        return model
        
    return None


def run_cross_dataset(source_dataset, target_dataset):
    print(f"\n{'='*60}")
    print(f"  CROSS-DATASET GENERALIZATION")
    print(f"  Source: {source_dataset}  ->  Target: {target_dataset}")
    print(f"{'='*60}")

    set_seed()
    
    # Prepare target dataset
    target_filepath = config.DATASETS[target_dataset]['path']
    data_prepared = prepare_dataset(
        target_filepath,
        seq_len=config.SEQ_LEN,
        pred_len=config.PRED_LEN,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        batch_size=config.BATCH_SIZE,
    )
    
    mean = data_prepared['mean']
    std = data_prepared['std']
    test_X, test_Y = data_prepared['splits']['test']
    num_sensors = test_X.shape[2]
    
    results = {}
    
    # 1. Evaluate LSTM
    print("\n>>> Testing Transferred LSTM...")
    lstm_model = load_temporal_model('lstm', source_dataset, num_sensors)
    if lstm_model:
        preds, gt, latency = predict_model(lstm_model, data_prepared['loaders']['test'], config, 'lstm')
        res = evaluate_predictions(preds, gt, mean, std)
        print_results(res, "Transferred LSTM")
        results['LSTM'] = res
    else:
        print(f"  Source model lstm_{source_dataset}_best.pt not found.")
        
    # 2. Evaluate Random Forest
    print("\n>>> Testing Transferred Random Forest...")
    rf_model = load_temporal_model('rf', source_dataset, num_sensors)
    if rf_model:
        preds = rf_model.predict(test_X)
        res = evaluate_predictions(preds, test_Y, mean, std)
        print_results(res, "Transferred Random Forest")
        results['RF'] = res
    else:
        print(f"  Source model rf_{source_dataset}_best.pkl not found.")

    # Save metrics
    if results:
        save_path = os.path.join(config.RESULTS_DIR, "metrics", f"cross_{source_dataset}_to_{target_dataset}.json")
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nCross-dataset results saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cross-dataset transfer')
    parser.add_argument('--source', type=str, default='PEMS-BAY')
    parser.add_argument('--target', type=str, default='METR-LA')
    args = parser.parse_args()
    
    run_cross_dataset(args.source, args.target)
