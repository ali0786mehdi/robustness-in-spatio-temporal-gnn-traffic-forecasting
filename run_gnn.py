"""
Run GNN models: STGCN and DCRNN.
Usage: python run_gnn.py [--dataset METR-LA|PEMS-BAY|both]
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


def run_stgcn(data_prepared, graph_data, dataset_name, mean, std, save_tag=None):
    """Run STGCN model."""
    from src.models.stgcn import STGCN

    print(f"\n{'='*60}")
    print(f"  STGCN — {dataset_name}" + (f" [{save_tag}]" if save_tag else ""))
    print(f"{'='*60}")

    num_sensors = data_prepared['splits']['train'][0].shape[2]

    model = STGCN(
        num_sensors=num_sensors,
        seq_len=config.SEQ_LEN,
        pred_len=config.PRED_LEN,
        K=config.STGCN_K,
        channels=config.STGCN_CHANNELS,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    history = train_model(
        model, data_prepared['loaders']['train'],
        data_prepared['loaders']['val'],
        config, 'stgcn', dataset_name,
        graph_data=graph_data, save_tag=save_tag,
    )

    predictions, ground_truth, latency = predict_model(
        model, data_prepared['loaders']['test'], config, 'stgcn',
        graph_data=graph_data,
    )
    history['efficiency']['inference_latency_ms'] = latency

    results = evaluate_predictions(predictions, ground_truth, mean, std)
    print_results(results, model.get_name())
    return results, predictions, history


def run_dcrnn(data_prepared, graph_data, dataset_name, mean, std, save_tag=None):
    """Run DCRNN model."""
    from src.models.dcrnn import DCRNN

    print(f"\n{'='*60}")
    print(f"  DCRNN — {dataset_name}" + (f" [{save_tag}]" if save_tag else ""))
    print(f"{'='*60}")

    num_sensors = data_prepared['splits']['train'][0].shape[2]
    num_supports = len(graph_data['diffusion_supports'])

    model = DCRNN(
        num_sensors=num_sensors,
        num_supports=num_supports,
        seq_len=config.SEQ_LEN,
        pred_len=config.PRED_LEN,
        hidden_dim=config.DCRNN_HIDDEN,
        num_layers=config.DCRNN_LAYERS,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    history = train_model(
        model, data_prepared['loaders']['train'],
        data_prepared['loaders']['val'],
        config, 'dcrnn', dataset_name,
        graph_data=graph_data, save_tag=save_tag,
    )

    predictions, ground_truth, latency = predict_model(
        model, data_prepared['loaders']['test'], config, 'dcrnn',
        graph_data=graph_data,
    )
    history['efficiency']['inference_latency_ms'] = latency

    results = evaluate_predictions(predictions, ground_truth, mean, std)
    print_results(results, model.get_name())
    return results, predictions, history


def run_gnn_on_dataset(dataset_name, ablation=None, max_epochs=None,
                       models=('stgcn', 'dcrnn')):
    """Run all GNN models on a single dataset."""
    print(f"\n{'#'*60}")
    print(f"  GNN MODELS — {dataset_name}" + (f" [ABLATION: {ablation.upper()}]" if ablation else ""))
    if max_epochs is not None:
        config.EPOCHS = max_epochs
        print(f"  [Epochs overridden: {max_epochs}]")
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
    
    # Pass train_raw to build_graph
    graph_data = build_graph(
        data_prepared['train_raw'],
        sigma=config.GRAPH_SIGMA,
        epsilon=config.GRAPH_EPSILON,
        K_cheb=config.STGCN_K,
        K_diff=config.DIFFUSION_STEPS,
        ablation=ablation
    )

    all_results = {}
    all_preds = {}
    histories = {}

    # 1. STGCN
    if 'stgcn' in models:
        try:
            results, preds, history = run_stgcn(
                data_prepared, graph_data, dataset_name, mean, std,
                save_tag=f"ablation_{ablation}" if ablation else None,
            )
            all_results['STGCN'] = results
            all_preds['STGCN'] = preds
            histories['STGCN'] = history
        except Exception as e:
            print(f"  STGCN failed: {e}")
            import traceback; traceback.print_exc()

    # 2. DCRNN
    if 'dcrnn' in models:
        try:
            results, preds, history = run_dcrnn(
                data_prepared, graph_data, dataset_name, mean, std,
                save_tag=f"ablation_{ablation}" if ablation else None,
            )
            all_results['DCRNN'] = results
            all_preds['DCRNN'] = preds
            histories['DCRNN'] = history
        except Exception as e:
            print(f"  DCRNN failed: {e}")
            import traceback; traceback.print_exc()

    # Save results
    metrics_dir = os.path.join(config.RESULTS_DIR, 'metrics')
    save_name = dataset_name + f"_gnn{'_' + ablation if ablation else ''}"
    save_results(all_results, metrics_dir, save_name)
    save_efficiency(histories, metrics_dir, save_name)

    return all_results, all_preds, histories, data_prepared, graph_data


def main():
    parser = argparse.ArgumentParser(description='Run GNN models')
    parser.add_argument('--dataset', type=str, default='METR-LA',
                        choices=['METR-LA', 'PEMS-BAY', 'both'])
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['random', 'identity'],
                        help='Override graph with random or identity matrix for ablation study')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs (e.g. 30 for quick ablations)')
    parser.add_argument('--model', type=str, default='both',
                        choices=['stgcn', 'dcrnn', 'both'],
                        help='Which GNN to train (default: both)')
    args = parser.parse_args()

    set_seed()
    device = config.get_device()

    models = ('stgcn', 'dcrnn') if args.model == 'both' else (args.model,)
    datasets = ['METR-LA', 'PEMS-BAY'] if args.dataset == 'both' else [args.dataset]

    for ds in datasets:
        run_gnn_on_dataset(ds, ablation=args.ablation,
                           max_epochs=args.epochs, models=models)


if __name__ == '__main__':
    main()
