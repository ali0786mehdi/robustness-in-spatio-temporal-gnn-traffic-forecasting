"""
Evaluation module for traffic forecasting.
Computes MAE, RMSE, MAPE at multiple prediction horizons.
"""

import numpy as np
import json
import os


def masked_mae(preds, targets, null_val=0.0):
    """Mean Absolute Error, ignoring entries where target ≈ null_val."""
    mask = np.abs(targets) > null_val + 1e-5
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(preds[mask] - targets[mask]))


def masked_rmse(preds, targets, null_val=0.0):
    """Root Mean Squared Error, ignoring near-zero targets."""
    mask = np.abs(targets) > null_val + 1e-5
    if mask.sum() == 0:
        return 0.0
    return np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2))


def masked_mape(preds, targets, null_val=0.0):
    """Mean Absolute Percentage Error, ignoring near-zero targets."""
    mask = np.abs(targets) > null_val + 1e-5
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100


def evaluate_predictions(preds, targets, mean, std, horizons=None):
    """
    Compute evaluation metrics at specified horizons.
    Predictions and targets are de-normalized before evaluation.

    Args:
        preds (np.ndarray): Normalized predictions, shape (N, pred_len, sensors).
        targets (np.ndarray): Normalized targets, shape (N, pred_len, sensors).
        mean (np.ndarray): Per-sensor mean for denormalization.
        std (np.ndarray): Per-sensor std for denormalization.
        horizons (dict): {name: step_index}, e.g. {'15min': 3, '30min': 6, '60min': 12}.

    Returns:
        dict: Nested dict of metrics per horizon.
    """
    if horizons is None:
        horizons = {'15min': 3, '30min': 6, '60min': 12}

    # De-normalize
    preds_dn = preds * std + mean
    targets_dn = targets * std + mean

    results = {}

    # Overall metrics (all horizons combined)
    results['overall'] = {
        'MAE': float(masked_mae(preds_dn, targets_dn)),
        'RMSE': float(masked_rmse(preds_dn, targets_dn)),
        'MAPE': float(masked_mape(preds_dn, targets_dn)),
    }

    # Per-horizon metrics
    for horizon_name, step_idx in horizons.items():
        # step_idx is 1-indexed horizon (3 = 15min), take steps 0..step_idx-1
        h_preds = preds_dn[:, step_idx - 1, :]  # At exact horizon step
        h_targets = targets_dn[:, step_idx - 1, :]

        results[horizon_name] = {
            'MAE': float(masked_mae(h_preds, h_targets)),
            'RMSE': float(masked_rmse(h_preds, h_targets)),
            'MAPE': float(masked_mape(h_preds, h_targets)),
        }

    return results


def print_results(results, model_name):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  Results: {model_name}")
    print(f"{'='*60}")
    print(f"  {'Horizon':<12} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>8}")
    print(f"  {'-'*40}")

    for horizon in ['15min', '30min', '60min', 'overall']:
        if horizon in results:
            r = results[horizon]
            print(f"  {horizon:<12} {r['MAE']:8.4f} {r['RMSE']:8.4f} {r['MAPE']:8.2f}")

    print(f"{'='*60}\n")


def save_results(all_results, save_dir, dataset_name):
    """
    Save all model results to JSON.

    Args:
        all_results (dict): {model_name: results_dict}.
        save_dir (str): Directory to save results.
        dataset_name (str): Dataset name for filename.
    """
    save_path = os.path.join(save_dir, f"{dataset_name}_results.json")
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {save_path}")


def save_efficiency(histories, save_dir, dataset_name):
    """
    Extract efficiency metrics from histories and save to JSON.
    """
    efficiency_data = {}
    for model_name, hist in histories.items():
        if 'efficiency' in hist:
            efficiency_data[model_name] = hist['efficiency']
            
    if efficiency_data:
        save_path = os.path.join(save_dir, f"{dataset_name}_efficiency.json")
        with open(save_path, 'w') as f:
            json.dump(efficiency_data, f, indent=2)
        print(f"Efficiency metrics saved to {save_path}")


def load_results(save_dir, dataset_name):
    """Load results from JSON."""
    save_path = os.path.join(save_dir, f"{dataset_name}_results.json")
    with open(save_path, 'r') as f:
        return json.load(f)


def compare_models(all_results, dataset_name):
    """
    Print a comparison table of all models.

    Args:
        all_results (dict): {model_name: results_dict}.
        dataset_name (str): Dataset name.
    """
    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON — {dataset_name}")
    print(f"{'='*80}")

    horizons = ['15min', '30min', '60min', 'overall']

    for horizon in horizons:
        print(f"\n  --- {horizon.upper()} ---")
        print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>8}")
        print(f"  {'-'*52}")

        # Sort by MAE
        model_scores = []
        for model_name, results in all_results.items():
            if horizon in results:
                model_scores.append((model_name, results[horizon]))

        model_scores.sort(key=lambda x: x[1]['MAE'])

        for model_name, r in model_scores:
            print(f"  {model_name:<25} {r['MAE']:8.4f} {r['RMSE']:8.4f} {r['MAPE']:8.2f}")

    print(f"\n{'='*80}\n")
