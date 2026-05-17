"""
Visualization module for traffic forecasting results.
Generates comparison plots, prediction curves, and analysis charts.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('Agg')  # Non-interactive backend
plt.style.use('seaborn-v0_8-darkgrid')

# Color palette for models
MODEL_COLORS = {
    'ARIMA': '#e74c3c',
    'RandomForest': '#e67e22',
    'LSTM': '#2ecc71',
    'STGCN': '#3498db',
    'DCRNN': '#9b59b6',
    'Ground Truth': '#2c3e50',
}


def plot_predictions(preds_dict, ground_truth, mean, std, sensor_idx=0,
                     time_range=(0, 200), save_path=None):
    """
    Plot predicted vs ground truth time series for a single sensor.

    Args:
        preds_dict: {model_name: predictions array (N, pred_len, sensors)}
        ground_truth: Ground truth array (N, pred_len, sensors)
        mean, std: For de-normalization
        sensor_idx: Which sensor to plot
        time_range: (start, end) sample indices
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    start, end = time_range

    # De-normalize ground truth (single step ahead for clarity)
    gt = ground_truth[start:end, 0, sensor_idx] * std[sensor_idx] + mean[sensor_idx]
    ax.plot(gt, color=MODEL_COLORS['Ground Truth'], linewidth=2,
            label='Ground Truth', alpha=0.9)

    for model_name, preds in preds_dict.items():
        pred = preds[start:end, 0, sensor_idx] * std[sensor_idx] + mean[sensor_idx]
        color = MODEL_COLORS.get(model_name, '#7f8c8d')
        ax.plot(pred, color=color, linewidth=1.2, label=model_name, alpha=0.8)

    ax.set_xlabel('Time Step (5-min intervals)', fontsize=12)
    ax.set_ylabel('Traffic Speed (mph)', fontsize=12)
    ax.set_title(f'Traffic Speed Prediction — Sensor {sensor_idx}', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_model_comparison(all_results, dataset_name, save_path=None):
    """
    Bar chart comparing MAE/RMSE/MAPE across all models at 60-min horizon.

    Args:
        all_results: {model_name: results_dict}
        dataset_name: Dataset name for title
        save_path: Path to save figure
    """
    models = list(all_results.keys())
    metrics = ['MAE', 'RMSE', 'MAPE']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, metric in enumerate(metrics):
        values = []
        colors = []
        for m in models:
            if '60min' in all_results[m]:
                values.append(all_results[m]['60min'][metric])
            else:
                values.append(all_results[m]['overall'][metric])
            colors.append(MODEL_COLORS.get(m, '#95a5a6'))

        bars = axes[idx].bar(models, values, color=colors, edgecolor='white',
                             linewidth=0.5)
        axes[idx].set_title(f'{metric} (60-min horizon)', fontsize=13)
        axes[idx].set_ylabel(metric, fontsize=11)
        axes[idx].tick_params(axis='x', rotation=30)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle(f'Model Comparison — {dataset_name}', fontsize=15, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_horizon_performance(all_results, dataset_name, save_path=None):
    """
    Line chart showing how error increases with prediction horizon.

    Args:
        all_results: {model_name: results_dict}
        dataset_name: Dataset name for title
        save_path: Path to save figure
    """
    horizons = ['15min', '30min', '60min']
    horizon_labels = ['15 min', '30 min', '60 min']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['MAE', 'RMSE', 'MAPE']

    for idx, metric in enumerate(metrics):
        for model_name, results in all_results.items():
            values = [results[h][metric] for h in horizons if h in results]
            color = MODEL_COLORS.get(model_name, '#95a5a6')
            axes[idx].plot(horizon_labels[:len(values)], values,
                          marker='o', linewidth=2, label=model_name, color=color)

        axes[idx].set_title(f'{metric} vs Horizon', fontsize=13)
        axes[idx].set_xlabel('Prediction Horizon', fontsize=11)
        axes[idx].set_ylabel(metric, fontsize=11)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle(f'Horizon Analysis — {dataset_name}', fontsize=15, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_adjacency_matrix(adj, dataset_name, save_path=None):
    """
    Heatmap of the adjacency matrix.

    Args:
        adj: Adjacency matrix (N, N)
        dataset_name: Dataset name for title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(adj, cmap='YlOrRd', ax=ax, xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Edge Weight'})
    ax.set_title(f'Adjacency Matrix — {dataset_name}', fontsize=14)
    ax.set_xlabel('Sensor', fontsize=12)
    ax.set_ylabel('Sensor', fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_training_curves(histories, dataset_name, save_path=None):
    """
    Plot training and validation loss curves for deep models.

    Args:
        histories: {model_name: {'train_loss': [...], 'val_loss': [...]}}
        dataset_name: Dataset name for title
        save_path: Path to save figure
    """
    n_models = len(histories)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, history) in enumerate(histories.items()):
        color = MODEL_COLORS.get(model_name, '#3498db')
        axes[idx].plot(history['train_loss'], label='Train', color=color, alpha=0.8)
        axes[idx].plot(history['val_loss'], label='Val', color=color,
                      linestyle='--', alpha=0.8)
        axes[idx].set_title(f'{model_name} Loss', fontsize=13)
        axes[idx].set_xlabel('Epoch', fontsize=11)
        axes[idx].set_ylabel('MSE Loss', fontsize=11)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle(f'Training Curves — {dataset_name}', fontsize=15, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_spatial_error(preds, targets, mean, std, dataset_name, model_name,
                       save_path=None):
    """
    Bar chart of per-sensor MAE showing spatial error distribution.

    Args:
        preds, targets: (N, pred_len, sensors) — normalized
        mean, std: For de-normalization
        dataset_name, model_name: For title
        save_path: Path to save figure
    """
    # De-normalize
    preds_dn = preds * std + mean
    targets_dn = targets * std + mean

    # Per-sensor MAE (across all samples and horizons)
    sensor_mae = np.mean(np.abs(preds_dn - targets_dn), axis=(0, 1))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(sensor_mae)), sensor_mae, color='#3498db', alpha=0.7, width=1.0)
    ax.set_xlabel('Sensor Index', fontsize=12)
    ax.set_ylabel('MAE (mph)', fontsize=12)
    ax.set_title(f'Per-Sensor MAE — {model_name} on {dataset_name}', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def generate_all_plots(all_results, preds_dict, ground_truth, mean, std,
                       adj, histories, dataset_name, save_dir):
    """
    Generate all visualization plots.

    Args:
        all_results: {model_name: metrics_dict}
        preds_dict: {model_name: predictions array}
        ground_truth: Ground truth array
        mean, std: Normalization params
        adj: Adjacency matrix
        histories: {model_name: training history} (deep models only)
        dataset_name: Dataset name
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nGenerating plots for {dataset_name}...")

    # 1. Prediction vs ground truth (first 3 sensors)
    for s in [0, 50, 100]:
        num_sensors = ground_truth.shape[2]
        if s < num_sensors:
            plot_predictions(
                preds_dict, ground_truth, mean, std, sensor_idx=s,
                save_path=os.path.join(save_dir, f'{dataset_name}_pred_sensor{s}.png')
            )

    # 2. Model comparison bar chart
    plot_model_comparison(
        all_results, dataset_name,
        save_path=os.path.join(save_dir, f'{dataset_name}_comparison.png')
    )

    # 3. Horizon performance
    plot_horizon_performance(
        all_results, dataset_name,
        save_path=os.path.join(save_dir, f'{dataset_name}_horizons.png')
    )

    # 4. Adjacency matrix heatmap
    plot_adjacency_matrix(
        adj, dataset_name,
        save_path=os.path.join(save_dir, f'{dataset_name}_adjacency.png')
    )

    # 5. Training curves (deep models only)
    if histories:
        plot_training_curves(
            histories, dataset_name,
            save_path=os.path.join(save_dir, f'{dataset_name}_training.png')
        )

    # 6. Spatial error for best GNN model
    for model_name in ['STGCN', 'DCRNN']:
        if model_name in preds_dict:
            plot_spatial_error(
                preds_dict[model_name], ground_truth, mean, std,
                dataset_name, model_name,
                save_path=os.path.join(save_dir, f'{dataset_name}_{model_name}_spatial.png')
            )

    print(f"All plots saved to {save_dir}/")
