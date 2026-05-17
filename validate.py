"""
Validation & Testing Script
============================
Use this to:
1. Quick smoke test — verify all models produce valid outputs
2. Validate a trained model — check predictions make sense
3. Cross-dataset check — compare METR-LA vs PEMS-BAY results
4. Load saved results and compare

Usage:
    python validate.py --mode smoke       # Quick 1-epoch test (no real training)
    python validate.py --mode check       # Validate saved model checkpoints
    python validate.py --mode compare     # Compare results across datasets
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


def smoke_test(dataset_name="METR-LA"):
    """
    Quick 1-epoch test on a small subset to verify everything works.
    This does NOT produce real results — only checks for crashes/errors.
    """
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST — {dataset_name}")
    print(f"  (1 epoch, small batch, just checking for errors)")
    print(f"{'='*60}\n")

    set_seed()
    filepath = config.DATASETS[dataset_name]['path']

    # Load data
    data = prepare_dataset(filepath, batch_size=16)
    mean, std = data['mean'], data['std']
    splits = data['splits']
    num_sensors = splits['train'][0].shape[2]

    # Build graph
    train_end = int(len(data['raw_data']) * config.TRAIN_RATIO)
    graph_data = build_graph(data['raw_data'][:train_end],
                             sigma=config.GRAPH_SIGMA,
                             epsilon=config.GRAPH_EPSILON)

    device = config.DEVICE
    test_X = torch.FloatTensor(splits['test'][0][:16]).to(device)  # Just 16 samples
    test_Y = splits['test'][1][:16]

    results = {}
    passed = 0
    failed = 0

    # ---- Test 1: LSTM ----
    try:
        from src.models.lstm_model import LSTMModel
        model = LSTMModel(num_sensors, 12, 12, 64, 2, 0.3).to(device)
        with torch.no_grad():
            pred = model(test_X).cpu().numpy()

        assert pred.shape == (16, 12, num_sensors), f"Bad shape: {pred.shape}"
        assert not np.isnan(pred).any(), "NaN in predictions"
        assert not np.isinf(pred).any(), "Inf in predictions"

        r = evaluate_predictions(pred, test_Y, mean, std)
        results['LSTM'] = r
        print(f"  ✓ LSTM — shape OK, no NaN, MAE={r['overall']['MAE']:.4f}")
        passed += 1
    except Exception as e:
        print(f"  ✗ LSTM FAILED: {e}")
        failed += 1

    # ---- Test 2: STGCN ----
    try:
        from src.models.stgcn import STGCN
        model = STGCN(num_sensors, 12, 12, K=3).to(device)
        cheb = [torch.FloatTensor(p).to(device) for p in graph_data['cheb_polys']]
        with torch.no_grad():
            pred = model(test_X, cheb).cpu().numpy()

        assert pred.shape == (16, 12, num_sensors), f"Bad shape: {pred.shape}"
        assert not np.isnan(pred).any(), "NaN in predictions"
        assert not np.isinf(pred).any(), "Inf in predictions"

        r = evaluate_predictions(pred, test_Y, mean, std)
        results['STGCN'] = r
        print(f"  ✓ STGCN — shape OK, no NaN, MAE={r['overall']['MAE']:.4f}")
        passed += 1
    except Exception as e:
        print(f"  ✗ STGCN FAILED: {e}")
        failed += 1

    # ---- Test 3: DCRNN ----
    try:
        from src.models.dcrnn import DCRNN
        num_supports = len(graph_data['diffusion_supports'])
        model = DCRNN(num_sensors, num_supports, 12, 12, 64, 2).to(device)
        supports = [torch.FloatTensor(s).to(device)
                     for s in graph_data['diffusion_supports']]
        with torch.no_grad():
            pred = model(test_X, supports).cpu().numpy()

        assert pred.shape == (16, 12, num_sensors), f"Bad shape: {pred.shape}"
        assert not np.isnan(pred).any(), "NaN in predictions"
        assert not np.isinf(pred).any(), "Inf in predictions"

        r = evaluate_predictions(pred, test_Y, mean, std)
        results['DCRNN'] = r
        print(f"  ✓ DCRNN — shape OK, no NaN, MAE={r['overall']['MAE']:.4f}")
        passed += 1
    except Exception as e:
        print(f"  ✗ DCRNN FAILED: {e}")
        failed += 1

    # ---- Test 4: Data sanity ----
    try:
        raw = data['raw_data']
        assert not np.isnan(raw).any(), "NaN in cleaned data"
        assert raw.min() >= 0, f"Negative speed: {raw.min()}"
        assert raw.max() < 200, f"Unrealistic speed: {raw.max()}"
        assert len(data['sensor_ids']) == num_sensors

        print(f"\n  ✓ Data sanity — no NaN, speed range [{raw.min():.1f}, {raw.max():.1f}] mph")
        print(f"    Sensors: {num_sensors}, Timesteps: {len(raw)}")
        passed += 1
    except Exception as e:
        print(f"  ✗ Data sanity FAILED: {e}")
        failed += 1

    # ---- Test 5: Graph sanity ----
    try:
        adj = graph_data['adj']
        assert adj.shape == (num_sensors, num_sensors)
        assert np.allclose(adj, adj.T, atol=1e-5), "Adjacency not symmetric"
        assert np.all(np.diag(adj) == 1.0), "Missing self-loops"
        assert adj.min() >= 0, "Negative edge weights"

        edges = np.count_nonzero(adj) - num_sensors  # exclude self-loops
        print(f"  ✓ Graph sanity — symmetric, {edges} edges, "
              f"sparsity {1 - np.count_nonzero(adj)/(num_sensors**2):.1%}")
        passed += 1
    except Exception as e:
        print(f"  ✗ Graph sanity FAILED: {e}")
        failed += 1

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST RESULT: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")

    return failed == 0


def check_saved_models(dataset_name="METR-LA"):
    """
    Load saved model checkpoints and validate their predictions on test data.
    Run this AFTER training to verify models were saved correctly.
    """
    print(f"\n{'='*60}")
    print(f"  CHECKING SAVED MODELS — {dataset_name}")
    print(f"{'='*60}\n")

    set_seed()
    filepath = config.DATASETS[dataset_name]['path']
    data = prepare_dataset(filepath, batch_size=config.BATCH_SIZE)
    mean, std = data['mean'], data['std']
    splits = data['splits']
    num_sensors = splits['train'][0].shape[2]

    # Build graph
    train_end = int(len(data['raw_data']) * config.TRAIN_RATIO)
    graph_data = build_graph(data['raw_data'][:train_end],
                             sigma=config.GRAPH_SIGMA,
                             epsilon=config.GRAPH_EPSILON)

    device = config.DEVICE
    test_X = torch.FloatTensor(splits['test'][0]).to(device)
    test_Y = splits['test'][1]

    models_dir = os.path.join(config.RESULTS_DIR, 'models')
    all_results = {}

    # ---- Check LSTM ----
    checkpoint = os.path.join(models_dir, f'lstm_{dataset_name}_best.pt')
    if os.path.exists(checkpoint):
        try:
            from src.models.lstm_model import LSTMModel
            model = LSTMModel(num_sensors, 12, 12, config.LSTM_HIDDEN,
                              config.LSTM_LAYERS, config.LSTM_DROPOUT).to(device)
            model.load_state_dict(torch.load(checkpoint, weights_only=True,
                                             map_location=device))
            model.eval()

            # Predict in batches
            preds = []
            with torch.no_grad():
                for i in range(0, len(test_X), config.BATCH_SIZE):
                    batch = test_X[i:i+config.BATCH_SIZE]
                    preds.append(model(batch).cpu().numpy())
            pred = np.concatenate(preds, axis=0)

            r = evaluate_predictions(pred, test_Y, mean, std)
            print_results(r, 'LSTM (saved)')
            all_results['LSTM'] = r
        except Exception as e:
            print(f"  ✗ LSTM load failed: {e}")
    else:
        print(f"  ⚠ LSTM checkpoint not found: {checkpoint}")

    # ---- Check STGCN ----
    checkpoint = os.path.join(models_dir, f'stgcn_{dataset_name}_best.pt')
    if os.path.exists(checkpoint):
        try:
            from src.models.stgcn import STGCN
            model = STGCN(num_sensors, 12, 12, K=config.STGCN_K,
                          channels=config.STGCN_CHANNELS).to(device)
            model.load_state_dict(torch.load(checkpoint, weights_only=True,
                                             map_location=device))
            model.eval()
            cheb = [torch.FloatTensor(p).to(device)
                    for p in graph_data['cheb_polys']]

            preds = []
            with torch.no_grad():
                for i in range(0, len(test_X), config.BATCH_SIZE):
                    batch = test_X[i:i+config.BATCH_SIZE]
                    preds.append(model(batch, cheb).cpu().numpy())
            pred = np.concatenate(preds, axis=0)

            r = evaluate_predictions(pred, test_Y, mean, std)
            print_results(r, 'STGCN (saved)')
            all_results['STGCN'] = r
        except Exception as e:
            print(f"  ✗ STGCN load failed: {e}")
    else:
        print(f"  ⚠ STGCN checkpoint not found: {checkpoint}")

    # ---- Check DCRNN ----
    checkpoint = os.path.join(models_dir, f'dcrnn_{dataset_name}_best.pt')
    if os.path.exists(checkpoint):
        try:
            from src.models.dcrnn import DCRNN
            num_supports = len(graph_data['diffusion_supports'])
            model = DCRNN(num_sensors, num_supports, 12, 12,
                          config.DCRNN_HIDDEN, config.DCRNN_LAYERS).to(device)
            model.load_state_dict(torch.load(checkpoint, weights_only=True,
                                             map_location=device))
            model.eval()
            supports = [torch.FloatTensor(s).to(device)
                        for s in graph_data['diffusion_supports']]

            preds = []
            with torch.no_grad():
                for i in range(0, len(test_X), config.BATCH_SIZE):
                    batch = test_X[i:i+config.BATCH_SIZE]
                    preds.append(model(batch, supports).cpu().numpy())
            pred = np.concatenate(preds, axis=0)

            r = evaluate_predictions(pred, test_Y, mean, std)
            print_results(r, 'DCRNN (saved)')
            all_results['DCRNN'] = r
        except Exception as e:
            print(f"  ✗ DCRNN load failed: {e}")
    else:
        print(f"  ⚠ DCRNN checkpoint not found: {checkpoint}")

    # ---- Sanity checks on results ----
    if all_results:
        print(f"\n{'='*60}")
        print(f"  SANITY CHECKS")
        print(f"{'='*60}")

        for name, r in all_results.items():
            mae = r['overall']['MAE']
            rmse = r['overall']['RMSE']
            mape = r['overall']['MAPE']

            checks = []
            if mae < 1.0:
                checks.append("⚠ MAE suspiciously low — possible data leakage")
            elif mae > 15.0:
                checks.append("⚠ MAE very high — model may not have trained")
            else:
                checks.append("✓ MAE in reasonable range")

            if rmse < mae:
                checks.append("✗ RMSE < MAE — impossible, something is wrong")
            else:
                checks.append("✓ RMSE > MAE (correct)")

            if mape < 1.0:
                checks.append("⚠ MAPE suspiciously low")
            elif mape > 50.0:
                checks.append("⚠ MAPE very high")
            else:
                checks.append("✓ MAPE in reasonable range")

            print(f"\n  {name}:")
            for c in checks:
                print(f"    {c}")

        # Compare: GNN should generally beat LSTM
        if 'LSTM' in all_results and 'STGCN' in all_results:
            lstm_mae = all_results['LSTM']['overall']['MAE']
            stgcn_mae = all_results['STGCN']['overall']['MAE']
            if stgcn_mae < lstm_mae:
                pct = (lstm_mae - stgcn_mae) / lstm_mae * 100
                print(f"\n  ✓ STGCN beats LSTM by {pct:.1f}% MAE — graph structure helps!")
            else:
                print(f"\n  ⚠ STGCN did NOT beat LSTM — check training or hyperparameters")

        if 'LSTM' in all_results and 'DCRNN' in all_results:
            lstm_mae = all_results['LSTM']['overall']['MAE']
            dcrnn_mae = all_results['DCRNN']['overall']['MAE']
            if dcrnn_mae < lstm_mae:
                pct = (lstm_mae - dcrnn_mae) / lstm_mae * 100
                print(f"  ✓ DCRNN beats LSTM by {pct:.1f}% MAE — diffusion convolution works!")
            else:
                print(f"  ⚠ DCRNN did NOT beat LSTM — check training or hyperparameters")

    return all_results


def compare_datasets():
    """
    Load and compare results from both METR-LA and PEMS-BAY.
    Shows side-by-side comparison to check consistency.
    """
    print(f"\n{'='*60}")
    print(f"  CROSS-DATASET COMPARISON")
    print(f"{'='*60}\n")

    metrics_dir = os.path.join(config.RESULTS_DIR, 'metrics')

    all_data = {}
    for dataset_name in ['METR-LA', 'PEMS-BAY']:
        # Try loading saved results
        for suffix in ['_all', '_baselines', '_gnn']:
            path = os.path.join(metrics_dir, f'{dataset_name}{suffix}_results.json')
            if os.path.exists(path):
                with open(path) as f:
                    results = json.load(f)
                if dataset_name not in all_data:
                    all_data[dataset_name] = {}
                all_data[dataset_name].update(results)
                print(f"  Loaded: {path}")

    if not all_data:
        print("  No saved results found. Run training first:")
        print("    python run_all.py --dataset METR-LA")
        print("    python run_all.py --dataset PEMS-BAY")
        return

    # Print comparison
    models = set()
    for ds_results in all_data.values():
        models.update(ds_results.keys())

    horizons = ['15min', '30min', '60min']

    for metric in ['MAE', 'RMSE', 'MAPE']:
        print(f"\n  --- {metric} ---")
        header = f"  {'Model':<20}"
        for ds in all_data:
            for h in horizons:
                header += f" {ds[:4]}_{h:>5}"
        print(header)
        print(f"  {'-'*(len(header)-2)}")

        for model_name in sorted(models):
            row = f"  {model_name:<20}"
            for ds in all_data:
                for h in horizons:
                    if model_name in all_data[ds] and h in all_data[ds][model_name]:
                        val = all_data[ds][model_name][h][metric]
                        row += f" {val:>10.2f}"
                    else:
                        row += f" {'N/A':>10}"
            print(row)

    # Consistency check
    print(f"\n  --- CONSISTENCY CHECK ---")
    for model_name in sorted(models):
        if all(model_name in all_data.get(ds, {}) for ds in ['METR-LA', 'PEMS-BAY']):
            mae_la = all_data['METR-LA'][model_name].get('overall', {}).get('MAE', 0)
            mae_bay = all_data['PEMS-BAY'][model_name].get('overall', {}).get('MAE', 0)
            if mae_la > 0 and mae_bay > 0:
                ratio = mae_bay / mae_la
                status = "✓" if 0.3 < ratio < 3.0 else "⚠"
                print(f"  {status} {model_name}: METR-LA MAE={mae_la:.2f}, "
                      f"PEMS-BAY MAE={mae_bay:.2f} (ratio={ratio:.2f})")


def main():
    parser = argparse.ArgumentParser(description='Validate models and results')
    parser.add_argument('--mode', type=str, default='smoke',
                        choices=['smoke', 'check', 'compare'],
                        help='smoke: quick test | check: validate saved models | '
                             'compare: cross-dataset comparison')
    parser.add_argument('--dataset', type=str, default='METR-LA',
                        choices=['METR-LA', 'PEMS-BAY'])
    args = parser.parse_args()

    set_seed()
    config.get_device()

    if args.mode == 'smoke':
        success = smoke_test(args.dataset)
        if success:
            print("All smoke tests passed! Ready to train.")
        else:
            print("Some tests failed. Fix errors before training.")

    elif args.mode == 'check':
        all_results = check_saved_models(args.dataset)
        if all_results:
            compare_models(all_results, args.dataset)

    elif args.mode == 'compare':
        compare_datasets()


if __name__ == '__main__':
    main()
