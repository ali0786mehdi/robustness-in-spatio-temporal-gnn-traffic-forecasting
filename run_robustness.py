"""
Run robustness experiments on trained models.
Evaluates model performance under varying levels of missing data and sensor failures.
"""

import sys
import os
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.config import set_seed
from src.data_loader import prepare_dataset
from src.graph_builder import build_graph, load_graph
from src.evaluate import evaluate_predictions
from src.robustness import inject_random_missing, inject_sensor_failure
from src.train import predict_model


def load_trained_model(model_name, dataset_name, num_sensors, graph_data=None):
    """Load a trained model from disk."""
    device = config.DEVICE
    save_path = os.path.join(config.RESULTS_DIR, "models", f"{model_name}_{dataset_name}_best.pt")
    
    if not os.path.exists(save_path):
        print(f"Warning: Model checkpoint not found at {save_path}")
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
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def evaluate_robustness(dataset_name="METR-LA"):
    print(f"\n{'='*60}")
    print(f"  ROBUSTNESS EXPERIMENTS — {dataset_name}")
    print(f"{'='*60}")

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
    
    mean = data_prepared['mean']
    std = data_prepared['std']
    num_sensors = data_prepared['splits']['train'][0].shape[2]
    
    # Build graph from training data
    train_end = int(len(data_prepared['raw_data']) * config.TRAIN_RATIO)
    train_raw = data_prepared['raw_data'][:train_end]
    graph_data = build_graph(
        train_raw,
        sigma=config.GRAPH_SIGMA,
        epsilon=config.GRAPH_EPSILON,
        K_cheb=config.STGCN_K,
        K_diff=config.DIFFUSION_STEPS,
    )
    
    # Load models
    models = {}
    for m in ['lstm', 'stgcn', 'dcrnn']:
        model = load_trained_model(m, dataset_name, num_sensors, graph_data)
        if model is not None:
            models[m] = model
            
    if not models:
        print("No trained models found. Please run train.py first.")
        return

    # Fill value in normalized space: 0 speed normalizes to -mean/std per sensor.
    # We use scalar 0.0 here — speed=0 (stopped/sensor dead) maps to a small negative
    # number in normalized space but 0.0 is a reasonable approximation for the purpose
    # of testing model robustness under missing inputs.
    fill_value = 0.0

    # Scenarios to test
    ratios = [0.0, 0.1, 0.2, 0.3, 0.4]
    scenarios = {
        'random_missing': inject_random_missing,
        'sensor_failure': inject_sensor_failure
    }
    
    results = {
        scenario_name: {ratio: {} for ratio in ratios}
        for scenario_name in scenarios
    }
    
    test_X_orig = data_prepared['splits']['test'][0]
    test_Y = data_prepared['splits']['test'][1]

    # Evaluate
    from torch.utils.data import DataLoader, TensorDataset

    for scenario_name, inject_func in scenarios.items():
        print(f"\n--- Scenario: {scenario_name} ---")
        for ratio in ratios:
            print(f"  Ratio: {ratio:.1%}")
            
            # Apply corruption
            if ratio == 0.0:
                corrupted_test_X = test_X_orig.copy()
            else:
                corrupted_test_X = inject_func(test_X_orig, ratio=ratio, fill_value=fill_value)
                
            # Create a temporary dataloader
            test_dataset = TensorDataset(torch.FloatTensor(corrupted_test_X), torch.FloatTensor(test_Y))
            test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
            
            for m_name, model in models.items():
                preds, gt, _ = predict_model(model, test_loader, config, m_name, graph_data)
                metrics = evaluate_predictions(preds, gt, mean, std)
                
                # Store overall MAE for easy access
                mae = metrics['overall']['MAE']
                results[scenario_name][ratio][m_name] = mae
                print(f"    {m_name.upper():<10} MAE: {mae:.4f}")

    # Save results
    save_path = os.path.join(config.RESULTS_DIR, "metrics", f"{dataset_name}_robustness.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRobustness results saved to {save_path}")
    

if __name__ == "__main__":
    evaluate_robustness("METR-LA")
