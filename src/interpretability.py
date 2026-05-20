"""
Interpretability and visualization module for research paper outputs.
Generates heatmaps and case study plots to explain model behavior.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_adjacency_heatmap(adj_matrix, save_dir, dataset_name, max_sensors=50):
    """
    Plot a heatmap of the adjacency matrix.
    To avoid clutter, plots only a subset of sensors if the graph is too large.
    """
    plt.figure(figsize=(10, 8))
    subset_adj = adj_matrix[:max_sensors, :max_sensors]
    
    # Mask zero values for better visibility
    mask = subset_adj == 0
    sns.heatmap(subset_adj, mask=mask, cmap="YlOrRd", square=True, 
                cbar_kws={'label': 'Correlation Similarity'},
                xticklabels=False, yticklabels=False)
                
    plt.title(f"Sensor Adjacency Heatmap ({dataset_name}, Subset N={max_sensors})", fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{dataset_name}_adjacency_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved adjacency heatmap to {save_path}")


def plot_spatial_error_heatmap(predictions, ground_truth, save_dir, dataset_name, model_name):
    """
    Plot the average Mean Absolute Error (MAE) for each sensor.
    Helps identify which physical locations are hardest to predict.
    """
    # predictions, ground_truth shape: (num_samples, pred_len, num_sensors)
    # Calculate MAE per sensor (averaged over time and horizons)
    error = np.abs(predictions - ground_truth)
    sensor_mae = np.mean(error, axis=(0, 1))  # Shape: (num_sensors,)
    
    # Sort sensors by error
    sorted_indices = np.argsort(sensor_mae)[::-1]  # Highest error first
    sorted_mae = sensor_mae[sorted_indices]
    
    plt.figure(figsize=(12, 5))
    # Plot top 50 hardest sensors
    top_n = min(50, len(sorted_mae))
    plt.bar(range(top_n), sorted_mae[:top_n], color='crimson', alpha=0.8)
    
    plt.title(f"Top {top_n} Hardest-to-Predict Sensors ({model_name} on {dataset_name})", fontsize=14)
    plt.xlabel("Sensor Rank (by Error)", fontsize=12)
    plt.ylabel("Mean Absolute Error (Speed)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_spatial_error.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved spatial error heatmap to {save_path}")


def plot_congestion_propagation(predictions_dict, ground_truth, timestamps, save_dir, dataset_name, start_idx=100):
    """
    Plot a case study showing how a sudden drop in speed (congestion)
    propagates across a few selected sensors, and how models react.
    """
    # Find a window with significant speed variation
    # ground_truth shape: (samples, pred_len, sensors)
    # Let's just take a slice of 24 samples (2 hours) for a few sensors
    window_size = 24
    
    if start_idx + window_size > len(ground_truth):
        start_idx = 0
        
    # Pick a few sensors that have high variance in this window
    window_gt = ground_truth[start_idx:start_idx+window_size, 0, :] # Look at t+1 predictions
    variances = np.var(window_gt, axis=0)
    top_sensors = np.argsort(variances)[::-1][:3]  # Top 3 most variable sensors
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    time_axis = np.arange(window_size)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, sensor_idx in enumerate(top_sensors):
        ax = axes[i]
        
        # Plot ground truth
        gt_seq = ground_truth[start_idx:start_idx+window_size, 0, sensor_idx]
        ax.plot(time_axis, gt_seq, 'k-', linewidth=2, label='Ground Truth')
        
        # Plot models
        for j, (model_name, preds) in enumerate(predictions_dict.items()):
            if model_name in ['STGCN', 'DCRNN', 'LSTM']: # Focus on deep models
                pred_seq = preds[start_idx:start_idx+window_size, 0, sensor_idx]
                ax.plot(time_axis, pred_seq, linestyle='--', color=colors[j % len(colors)], 
                        linewidth=1.5, label=model_name)
                
        ax.set_ylabel("Speed", fontsize=12)
        ax.set_title(f"Sensor ID Index: {sensor_idx}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.legend(loc='upper right', ncol=4)
            
    axes[-1].set_xlabel("Time Steps (5-min intervals)", fontsize=12)
    plt.suptitle(f"Congestion Shockwave Case Study ({dataset_name})", fontsize=16, y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{dataset_name}_congestion_case_study.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved congestion case study to {save_path}")
