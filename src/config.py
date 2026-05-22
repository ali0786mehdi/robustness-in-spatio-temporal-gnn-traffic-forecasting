"""
Central configuration for the Traffic Forecasting project.
All hyperparameters, paths, and settings are defined here.
"""

import os
import torch

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

METR_LA_PATH = os.path.join(DATASET_DIR, "METR-LA.csv")
PEMS_BAY_PATH = os.path.join(DATASET_DIR, "PEMS-BAY.csv")

# Create results directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"),   exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

# ── Model save directories (structured) ──────────────────────────────────────
MODELS_DIR          = os.path.join(RESULTS_DIR, "models")
BASELINE_MODELS_DIR = os.path.join(MODELS_DIR,  "baseline_models")
GNN_MODELS_DIR      = os.path.join(MODELS_DIR,  "gnn_models")

# GNN sub-dirs (one per architecture)
_GNN_MODEL_NAMES = ["stgcn", "dcrnn"]
for _name in _GNN_MODEL_NAMES:
    os.makedirs(os.path.join(GNN_MODELS_DIR, _name), exist_ok=True)
os.makedirs(BASELINE_MODELS_DIR, exist_ok=True)


def get_model_path(model_name: str, dataset_name: str, tag: str = None) -> str:
    """
    Return the canonical save/load path for a trained model checkpoint.

    GNN models  → results/models/gnn_models/<model>/
    Baselines   → results/models/baseline_models/

    Args:
        model_name:   'stgcn', 'dcrnn', 'lstm', 'arima', or 'rf'
        dataset_name: 'METR-LA' or 'PEMS-BAY'
        tag:          optional suffix, e.g. 'ablation_random'

    Returns:
        Absolute path string (file not created).
    """
    tag_str = f"_{tag}" if tag else ""
    ext     = ".pkl" if model_name in ("arima", "rf") else ".pt"
    fname   = f"{model_name}_{dataset_name}{tag_str}_best{ext}"

    if model_name in _GNN_MODEL_NAMES:
        return os.path.join(GNN_MODELS_DIR, model_name, fname)
    else:
        return os.path.join(BASELINE_MODELS_DIR, fname)


# ============================================================
# Dataset configuration
# ============================================================
DATASETS = {
    "METR-LA": {
        "path": METR_LA_PATH,
        "num_sensors": 207,
    },
    "PEMS-BAY": {
        "path": PEMS_BAY_PATH,
        "num_sensors": 325,
    },
}

# ============================================================
# Data preprocessing
# ============================================================
# Train / Validation / Test split ratios (chronological)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# Sequence parameters
SEQ_LEN = 12       # Input sequence length (12 steps = 1 hour at 5-min intervals)
PRED_LEN = 12      # Prediction horizon (12 steps = 1 hour ahead)

# ============================================================
# Graph construction
# ============================================================
# Correlation-based adjacency matrix parameters
GRAPH_SIGMA = 0.1           # Gaussian kernel bandwidth
GRAPH_EPSILON = 0.3         # Sparsity threshold (set edges < epsilon to 0)
DIFFUSION_STEPS = 2         # For DCRNN diffusion convolution

# ============================================================
# Training hyperparameters
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 128          # Increased from 64 — safe with AMP on RTX 4060 8GB
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 100
PATIENCE      = 15               # Early stopping patience
GRAD_CLIP     = 5.0             # Gradient clipping max norm
SCHEDULER_PATIENCE = 5      # LR scheduler patience
SCHEDULER_FACTOR   = 0.5      # LR scheduler reduction factor

# ── Performance flags ────────────────────────────────────────────────────────────
USE_AMP         = True   # Automatic Mixed Precision (fp16 Tensor Cores on RTX)
CUDNN_BENCHMARK = True   # Auto-select fastest cuDNN kernel for fixed input shape
                          # Note: disables exact bit-for-bit reproducibility
                          # Set False if you need exact gradient reproducibility

# ============================================================
# Model hyperparameters
# ============================================================

# LSTM
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3

# STGCN
STGCN_CHANNELS = [1, 16, 32, 64]     # Channel progression
STGCN_KERNEL_SIZE = 3                 # Temporal convolution kernel
STGCN_K = 3                          # Chebyshev polynomial order

# DCRNN
DCRNN_HIDDEN = 64
DCRNN_LAYERS = 2
DCRNN_DROPOUT = 0.3
DCRNN_FILTER_TYPE = "dual_random_walk"  # or "laplacian", "random_walk"

# ============================================================
# Random Forest
# ============================================================
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
RF_N_JOBS = -1

# ============================================================
# ARIMA
# ============================================================
ARIMA_ORDER = (3, 0, 1)       # (p, d, q) — default, can be auto-tuned
ARIMA_MAX_SENSORS = 30        # Fit ARIMA on subset of sensors for speed

# ============================================================
# Evaluation
# ============================================================
# Report metrics at these horizons (in number of steps)
EVAL_HORIZONS = {
    "15min": 3,
    "30min": 6,
    "60min": 12,
}

# ============================================================
# Reproducibility
# ============================================================
SEED = 42


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic=True forces slower algos for exact reproducibility.
        # We only enable it when CUDNN_BENCHMARK is off.
        torch.backends.cudnn.deterministic = not CUDNN_BENCHMARK
        torch.backends.cudnn.benchmark     = CUDNN_BENCHMARK


def get_device():
    """Get the computation device and print info."""
    device = DEVICE
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Using CPU — training will be slower for GNN models")
    return device
