"""
Model Validation & Results Analysis
=====================================
Loads pre-trained models from results/models/ and evaluates them.
NO training happens here — everything uses saved checkpoints and JSON results.

Usage:
    python validate.py                             # Validate all saved models on METR-LA
    python validate.py --dataset PEMS-BAY          # Validate on PEMS-BAY
    python validate.py --mode compare              # Compare METR-LA vs PEMS-BAY
    python validate.py --mode sanity               # Quick data + graph sanity checks
"""

import sys
import os
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.config import set_seed
from src.data_loader import prepare_dataset, denormalize_data
from src.graph_builder import build_graph
from src.evaluate import evaluate_predictions, print_results, compare_models


# ============================================================================
#  HELPER: Load a saved model checkpoint and predict on test data
# ============================================================================

def load_and_predict(model, checkpoint_path, test_X, device, batch_size,
                     graph_tensors=None, model_type='lstm'):
    """
    Load saved weights into model, run inference on test_X, return predictions.
    No training — purely inference.
    """
    if not os.path.exists(checkpoint_path):
        return None

    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=True, map_location=device)
    )
    model = model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(test_X), batch_size):
            batch = test_X[i:i + batch_size]

            if model_type == 'stgcn':
                pred = model(batch, graph_tensors)
            elif model_type == 'dcrnn':
                pred = model(batch, graph_tensors)
            else:
                pred = model(batch)

            preds.append(pred.cpu().numpy())

    return np.concatenate(preds, axis=0)


# ============================================================================
#  MODE 1: VALIDATE — Load saved models, evaluate, run sanity checks
# ============================================================================

def validate_models(dataset_name="METR-LA"):
    """
    Load all saved model checkpoints and evaluate on the test set.
    Also loads baseline results from JSON. Runs sanity checks on all results.
    """
    print(f"\n{'='*70}")
    print(f"  VALIDATING SAVED MODELS — {dataset_name}")
    print(f"  (Loading checkpoints from results/models/, NO training)")
    print(f"{'='*70}\n")

    set_seed()
    filepath = config.DATASETS[dataset_name]['path']

    # --- Data ---
    data = prepare_dataset(filepath, batch_size=config.BATCH_SIZE)
    mean, std = data['mean'], data['std']
    num_sensors = data['splits']['train'][0].shape[2]
    device = config.DEVICE

    test_X = torch.FloatTensor(data['splits']['test'][0]).to(device)
    test_Y = data['splits']['test'][1]

    # --- Graph ---
    graph = build_graph(data['train_raw'],
                        sigma=config.GRAPH_SIGMA, epsilon=config.GRAPH_EPSILON)

    models_dir = os.path.join(config.RESULTS_DIR, 'models')
    all_results = {}

    # --- 1. Load baseline results from JSON ---
    baselines_path = os.path.join(config.RESULTS_DIR, 'metrics',
                                  f'{dataset_name}_baselines_results.json')
    if os.path.exists(baselines_path):
        with open(baselines_path) as f:
            baseline_data = json.load(f)
        for name, r in baseline_data.items():
            all_results[name] = r
            mae = r.get('overall', {}).get('MAE', 0)
            print(f"  ✓ {name:<15} loaded from JSON   (MAE={mae:.4f})")
    else:
        print(f"  ⚠ No baseline results: {baselines_path}")

    # --- 2. Load & evaluate LSTM ---
    ckpt = os.path.join(models_dir, f'lstm_{dataset_name}_best.pt')
    if os.path.exists(ckpt):
        from src.models.lstm_model import LSTMModel
        model = LSTMModel(num_sensors, config.SEQ_LEN, config.PRED_LEN,
                          config.LSTM_HIDDEN, config.LSTM_LAYERS, config.LSTM_DROPOUT)
        preds = load_and_predict(model, ckpt, test_X, device, config.BATCH_SIZE)
        if preds is not None:
            r = evaluate_predictions(preds, test_Y, mean, std)
            all_results['LSTM'] = r
            print(f"  ✓ {'LSTM':<15} loaded checkpoint  (MAE={r['overall']['MAE']:.4f})")
    else:
        print(f"  ⚠ LSTM checkpoint not found")

    # --- 3. Load & evaluate STGCN ---
    ckpt = os.path.join(models_dir, f'stgcn_{dataset_name}_best.pt')
    if os.path.exists(ckpt):
        from src.models.stgcn import STGCN
        model = STGCN(num_sensors, config.SEQ_LEN, config.PRED_LEN,
                       K=config.STGCN_K, channels=config.STGCN_CHANNELS)
        cheb = [torch.FloatTensor(p).to(device) for p in graph['cheb_polys']]
        preds = load_and_predict(model, ckpt, test_X, device, config.BATCH_SIZE,
                                 graph_tensors=cheb, model_type='stgcn')
        if preds is not None:
            r = evaluate_predictions(preds, test_Y, mean, std)
            all_results['STGCN'] = r
            print(f"  ✓ {'STGCN':<15} loaded checkpoint  (MAE={r['overall']['MAE']:.4f})")
    else:
        print(f"  ⚠ STGCN checkpoint not found")

    # --- 4. Load & evaluate DCRNN ---
    ckpt = os.path.join(models_dir, f'dcrnn_{dataset_name}_best.pt')
    if os.path.exists(ckpt):
        from src.models.dcrnn import DCRNN
        n_sup = len(graph['diffusion_supports'])
        model = DCRNN(num_sensors, n_sup, config.SEQ_LEN, config.PRED_LEN,
                       config.DCRNN_HIDDEN, config.DCRNN_LAYERS)
        supports = [torch.FloatTensor(s).to(device)
                     for s in graph['diffusion_supports']]
        preds = load_and_predict(model, ckpt, test_X, device, config.BATCH_SIZE,
                                 graph_tensors=supports, model_type='dcrnn')
        if preds is not None:
            r = evaluate_predictions(preds, test_Y, mean, std)
            all_results['DCRNN'] = r
            print(f"  ✓ {'DCRNN':<15} loaded checkpoint  (MAE={r['overall']['MAE']:.4f})")
    else:
        print(f"  ⚠ DCRNN checkpoint not found")

    # --- 5. Also load GNN JSON results if available ---
    gnn_path = os.path.join(config.RESULTS_DIR, 'metrics',
                            f'{dataset_name}_gnn_results.json')
    if os.path.exists(gnn_path):
        with open(gnn_path) as f:
            gnn_data = json.load(f)
        print(f"\n  Also loaded GNN results from JSON for reference")

    # --- SANITY CHECKS ---
    if all_results:
        print(f"\n{'='*70}")
        print(f"  SANITY CHECKS")
        print(f"{'='*70}")

        for name, r in all_results.items():
            if 'overall' not in r:
                continue
            mae = r['overall']['MAE']
            rmse = r['overall']['RMSE']
            mape = r['overall']['MAPE']

            issues = []
            if mae < 1.0:
                issues.append("⚠ MAE < 1.0 — suspiciously low, possible data leakage")
            if mae > 15.0:
                issues.append("⚠ MAE > 15 — very high, model may not have converged")
            if rmse < mae:
                issues.append("✗ RMSE < MAE — mathematically impossible, bug detected")
            if mape < 1.0:
                issues.append("⚠ MAPE < 1% — suspiciously low")
            if mape > 50.0:
                issues.append("⚠ MAPE > 50% — very high")

            # Check error growth (15min should be < 60min)
            if '15min' in r and '60min' in r:
                if r['60min']['MAE'] < r['15min']['MAE']:
                    issues.append("⚠ 60min MAE < 15min MAE — unexpected, should grow")

            if not issues:
                print(f"\n  ✓ {name}: All checks passed (MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%)")
            else:
                print(f"\n  {name}:")
                for issue in issues:
                    print(f"    {issue}")

        # GNN vs LSTM comparison
        print(f"\n{'='*70}")
        print(f"  GNN EFFECTIVENESS")
        print(f"{'='*70}")

        if 'LSTM' in all_results and 'overall' in all_results['LSTM']:
            lstm_mae = all_results['LSTM']['overall']['MAE']
            for gnn in ['STGCN', 'DCRNN']:
                if gnn in all_results and 'overall' in all_results[gnn]:
                    gnn_mae = all_results[gnn]['overall']['MAE']
                    diff = (lstm_mae - gnn_mae) / lstm_mae * 100
                    if diff > 0:
                        print(f"  ✓ {gnn} beats LSTM by {diff:.1f}% — graph structure helps!")
                    else:
                        print(f"  ⚠ {gnn} worse than LSTM by {abs(diff):.1f}%")

        # Full comparison table
        print()
        compare_models(all_results, dataset_name)

    return all_results


# ============================================================================
#  MODE 2: COMPARE — Side-by-side METR-LA vs PEMS-BAY
# ============================================================================

def compare_datasets():
    """
    Load results from both datasets and show side-by-side comparison.
    Uses saved JSON files only — no model loading needed.
    """
    print(f"\n{'='*70}")
    print(f"  CROSS-DATASET COMPARISON (METR-LA vs PEMS-BAY)")
    print(f"{'='*70}\n")

    metrics_dir = os.path.join(config.RESULTS_DIR, 'metrics')
    all_data = {}

    for dataset in ['METR-LA', 'PEMS-BAY']:
        for suffix in ['_all', '_baselines', '_gnn']:
            path = os.path.join(metrics_dir, f'{dataset}{suffix}_results.json')
            if os.path.exists(path):
                with open(path) as f:
                    results = json.load(f)
                if dataset not in all_data:
                    all_data[dataset] = {}
                all_data[dataset].update(results)
                print(f"  Loaded: {os.path.basename(path)}")

    if len(all_data) < 2:
        available = list(all_data.keys()) if all_data else []
        print(f"\n  Need results from both datasets. Available: {available}")
        print(f"  Run: python run_all.py --dataset METR-LA")
        print(f"  Run: python run_all.py --dataset PEMS-BAY")
        return

    # Collect all model names
    models = sorted(set().union(*(d.keys() for d in all_data.values())))
    horizons = ['15min', '30min', '60min']

    for metric in ['MAE', 'RMSE', 'MAPE']:
        print(f"\n  ┌─── {metric} {'─'*50}")
        header = f"  │ {'Model':<15}"
        for ds in ['METR-LA', 'PEMS-BAY']:
            for h in horizons:
                header += f"  {ds[:4]}_{h}"
        print(header)
        print(f"  │ {'-'*65}")

        for model in models:
            row = f"  │ {model:<15}"
            for ds in ['METR-LA', 'PEMS-BAY']:
                for h in horizons:
                    val = all_data.get(ds, {}).get(model, {}).get(h, {}).get(metric, None)
                    row += f"  {val:>9.2f}" if val else f"  {'—':>9}"
            print(row)
        print(f"  └{'─'*68}")

    # Consistency check
    print(f"\n  CONSISTENCY:")
    for model in models:
        r_la = all_data.get('METR-LA', {}).get(model, {}).get('overall', {})
        r_bay = all_data.get('PEMS-BAY', {}).get(model, {}).get('overall', {})
        if r_la.get('MAE') and r_bay.get('MAE'):
            ratio = r_bay['MAE'] / r_la['MAE']
            status = "✓ consistent" if 0.3 < ratio < 3.0 else "⚠ inconsistent"
            print(f"  {model:<15} LA={r_la['MAE']:.2f}  BAY={r_bay['MAE']:.2f}  "
                  f"ratio={ratio:.2f}  {status}")


# ============================================================================
#  MODE 3: SANITY — Quick data + graph checks (no models)
# ============================================================================

def sanity_check(dataset_name="METR-LA"):
    """
    Quick checks on data quality and graph construction.
    Very fast — no model loading.
    """
    print(f"\n{'='*70}")
    print(f"  DATA & GRAPH SANITY CHECK — {dataset_name}")
    print(f"{'='*70}\n")

    filepath = config.DATASETS[dataset_name]['path']
    data = prepare_dataset(filepath, batch_size=16)
    raw = data['raw_data']
    num_sensors = data['splits']['train'][0].shape[2]

    passed = 0
    failed = 0

    # Data checks
    tests = [
        ("No NaN in data", not np.isnan(raw).any()),
        ("No negative speeds", raw.min() >= 0),
        ("Max speed < 200 mph", raw.max() < 200),
        (f"Sensor count = {num_sensors}", num_sensors > 0),
        ("Train/val/test splits exist", all(k in data['splits'] for k in ['train','val','test'])),
        ("Train > Val > Test sizes",
         len(data['splits']['train'][0]) > len(data['splits']['val'][0]) >
         len(data['splits']['test'][0]) * 0.3),
    ]

    for desc, result in tests:
        if result:
            print(f"  ✓ {desc}")
            passed += 1
        else:
            print(f"  ✗ {desc}")
            failed += 1

    # Graph checks
    graph = build_graph(data['train_raw'], sigma=config.GRAPH_SIGMA,
                        epsilon=config.GRAPH_EPSILON)
    adj = graph['adj']

    graph_tests = [
        ("Adjacency is square", adj.shape[0] == adj.shape[1] == num_sensors),
        ("Adjacency is symmetric", np.allclose(adj, adj.T, atol=1e-5)),
        ("Self-loops present", np.all(np.diag(adj) >= 0.99)),
        ("No negative weights", adj.min() >= 0),
        ("Not fully connected", np.count_nonzero(adj) < adj.size),
        (f"Chebyshev polys: {len(graph['cheb_polys'])} matrices", len(graph['cheb_polys']) > 0),
        (f"Diffusion supports: {len(graph['diffusion_supports'])} matrices",
         len(graph['diffusion_supports']) > 0),
    ]

    print()
    for desc, result in graph_tests:
        if result:
            print(f"  ✓ {desc}")
            passed += 1
        else:
            print(f"  ✗ {desc}")
            failed += 1

    # Checkpoint existence
    print()
    models_dir = os.path.join(config.RESULTS_DIR, 'models')
    for mname in ['lstm', 'stgcn', 'dcrnn']:
        ckpt = os.path.join(models_dir, f'{mname}_{dataset_name}_best.pt')
        exists = os.path.exists(ckpt)
        size = os.path.getsize(ckpt) / 1024 if exists else 0
        if exists:
            print(f"  ✓ {mname.upper()} checkpoint exists ({size:.0f} KB)")
            passed += 1
        else:
            print(f"  ⚠ {mname.upper()} checkpoint missing: {ckpt}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"  RESULT: {passed} passed, {failed} failed")
    print(f"{'='*70}\n")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate saved models and results (NO training)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate.py                        # Validate METR-LA models
  python validate.py --dataset PEMS-BAY     # Validate PEMS-BAY models
  python validate.py --mode compare         # Compare both datasets
  python validate.py --mode sanity          # Quick data/graph checks
        """
    )
    parser.add_argument('--mode', type=str, default='validate',
                        choices=['validate', 'compare', 'sanity'],
                        help='validate: load & eval saved models | '
                             'compare: cross-dataset | sanity: data checks')
    parser.add_argument('--dataset', type=str, default='METR-LA',
                        choices=['METR-LA', 'PEMS-BAY'])
    args = parser.parse_args()

    set_seed()
    config.get_device()

    if args.mode == 'validate':
        validate_models(args.dataset)
    elif args.mode == 'compare':
        compare_datasets()
    elif args.mode == 'sanity':
        sanity_check(args.dataset)


if __name__ == '__main__':
    main()
