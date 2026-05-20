"""
Data loading and preprocessing module.
Handles CSV loading, normalization, sliding window creation, and DataLoader setup.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    """PyTorch Dataset for traffic time-series sequences."""

    def __init__(self, X, Y):
        """
        Args:
            X (np.ndarray): Input sequences, shape (num_samples, seq_len, num_sensors).
            Y (np.ndarray): Target sequences, shape (num_samples, pred_len, num_sensors).
        """
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_csv(filepath):
    """
    Load traffic CSV file.

    Args:
        filepath (str): Path to CSV file.

    Returns:
        tuple: (data_array, sensor_ids, timestamps)
            - data_array: np.ndarray of shape (T, N)
            - sensor_ids: list of sensor ID strings
            - timestamps: pandas DatetimeIndex
    """
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    sensor_ids = list(df.columns)
    timestamps = df.index
    data = df.values.astype(np.float32)

    print(f"  Shape: {data.shape} (timesteps × sensors)")
    print(f"  Time range: {timestamps[0]} to {timestamps[-1]}")
    print(f"  Missing values: {np.isnan(data).sum()} ({np.isnan(data).mean():.2%})")
    print(f"  Zero values: {(data == 0).sum()} ({(data == 0).mean():.2%})")

    return data, sensor_ids, timestamps


def handle_missing_values(data):
    """
    Handle missing and zero values in traffic data.
    Strategy: forward-fill → backward-fill → column mean.

    Args:
        data (np.ndarray): Raw data, shape (T, N).

    Returns:
        np.ndarray: Cleaned data.
    """
    # Convert to DataFrame for easy filling
    df = pd.DataFrame(data)

    # Forward fill then backward fill
    df = df.ffill().bfill()

    # Fill any remaining NaNs with column mean
    df = df.fillna(df.mean())

    # If still NaN (entire column was NaN), fill with 0
    df = df.fillna(0)

    cleaned = df.values.astype(np.float32)

    # Replace zeros with column mean (0 usually means sensor failure in traffic data)
    for col in range(cleaned.shape[1]):
        col_data = cleaned[:, col]
        mask = col_data == 0
        if mask.sum() > 0 and mask.sum() < len(col_data):
            col_mean = col_data[~mask].mean()
            cleaned[mask, col] = col_mean

    print(f"  After cleaning — NaN: {np.isnan(cleaned).sum()}, Zeros: {(cleaned == 0).sum()}")

    return cleaned


def normalize_data(data, mean=None, std=None):
    """
    Z-score normalization per sensor.

    Args:
        data (np.ndarray): Data, shape (T, N).
        mean (np.ndarray, optional): Pre-computed mean per sensor.
        std (np.ndarray, optional): Pre-computed std per sensor.

    Returns:
        tuple: (normalized_data, mean, std)
    """
    if mean is None:
        mean = data.mean(axis=0)  # (N,)
    if std is None:
        std = data.std(axis=0)    # (N,)
        std[std < 1e-5] = 1.0     # Avoid division by zero

    normalized = (data - mean) / std
    return normalized, mean, std


def denormalize_data(data, mean, std):
    """
    Reverse Z-score normalization.

    Args:
        data (np.ndarray or torch.Tensor): Normalized data.
        mean (np.ndarray): Mean per sensor.
        std (np.ndarray): Std per sensor.

    Returns:
        De-normalized data in the same format as input.
    """
    if isinstance(data, torch.Tensor):
        mean_t = torch.FloatTensor(mean).to(data.device)
        std_t = torch.FloatTensor(std).to(data.device)
        return data * std_t + mean_t
    else:
        return data * std + mean


def create_sequences(data, seq_len=12, pred_len=12):
    """
    Create sliding window input-output sequences.

    Args:
        data (np.ndarray): Normalized data, shape (T, N).
        seq_len (int): Input sequence length.
        pred_len (int): Prediction horizon length.

    Returns:
        tuple: (X, Y)
            - X: shape (num_samples, seq_len, N)
            - Y: shape (num_samples, pred_len, N)
    """
    T, N = data.shape
    num_samples = T - seq_len - pred_len + 1

    X = np.zeros((num_samples, seq_len, N), dtype=np.float32)
    Y = np.zeros((num_samples, pred_len, N), dtype=np.float32)

    for i in range(num_samples):
        X[i] = data[i : i + seq_len]
        Y[i] = data[i + seq_len : i + seq_len + pred_len]

    print(f"  Created {num_samples} sequences: X={X.shape}, Y={Y.shape}")
    return X, Y


def split_data(data, train_ratio=0.7, val_ratio=0.1):
    """
    Chronological train/val/test split on raw flat data.

    Args:
        data (np.ndarray): Data array, shape (T, N).
        train_ratio (float): Fraction for training.
        val_ratio (float): Fraction for validation.

    Returns:
        tuple: (train_data, val_data, test_data)
    """
    T = len(data)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(f"  Raw splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data


def create_dataloaders(splits, batch_size=64):
    """
    Create PyTorch DataLoaders from data splits.

    Args:
        splits (dict): Output from split_data().
        batch_size (int): Batch size.

    Returns:
        dict: Dictionary with 'train', 'val', 'test' DataLoaders.
    """
    loaders = {}
    for name, (X, Y) in splits.items():
        dataset = TrafficDataset(X, Y)
        shuffle = (name == "train")
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

    return loaders


def prepare_dataset(filepath, seq_len=12, pred_len=12, train_ratio=0.7,
                    val_ratio=0.1, batch_size=64):
    """
    Full data preparation pipeline with STRICT chronological splitting.

    Args:
        filepath (str): Path to CSV file.
        seq_len (int): Input sequence length.
        pred_len (int): Prediction horizon.
        train_ratio (float): Training split ratio.
        val_ratio (float): Validation split ratio.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        dict: Dictionary containing:
            - 'loaders': DataLoaders for train/val/test
            - 'splits': Sequence dict {'train': (X,Y), 'val': (X,Y), 'test': (X,Y)}
            - 'raw_data': Cleaned raw data (before normalization)
            - 'train_raw': Raw training chunk (used for graphs/normalization)
            - 'norm_data': Normalized data (full sequence)
            - 'mean': Per-sensor mean
            - 'std': Per-sensor std
            - 'sensor_ids': List of sensor IDs
            - 'timestamps': DatetimeIndex
    """
    # Load
    raw_data, sensor_ids, timestamps = load_csv(filepath)

    # Clean
    print("Handling missing values...")
    raw_data = handle_missing_values(raw_data)

    # Split first to prevent sequence overlap leakage
    print("Splitting data chunks...")
    train_raw, val_raw, test_raw = split_data(raw_data, train_ratio, val_ratio)

    # Normalize (compute stats ONLY on training portion)
    print("Normalizing based on training stats...")
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)
    std[std < 1e-5] = 1.0

    train_norm = (train_raw - mean) / std
    val_norm = (val_raw - mean) / std
    test_norm = (test_raw - mean) / std

    # Create sequences separately for each chunk to prevent boundary leakage
    print("Creating strictly partitioned sequences...")
    X_train, Y_train = create_sequences(train_norm, seq_len, pred_len)
    X_val, Y_val = create_sequences(val_norm, seq_len, pred_len)
    X_test, Y_test = create_sequences(test_norm, seq_len, pred_len)

    splits = {
        "train": (X_train, Y_train),
        "val": (X_val, Y_val),
        "test": (X_test, Y_test)
    }

    # DataLoaders
    loaders = create_dataloaders(splits, batch_size)

    # Reconstruct fully normalized array just for returning 'norm_data' backward compatibility
    norm_data = (raw_data - mean) / std

    return {
        "loaders": loaders,
        "splits": splits,
        "raw_data": raw_data,
        "train_raw": train_raw,
        "norm_data": norm_data,
        "mean": mean,
        "std": std,
        "sensor_ids": sensor_ids,
        "timestamps": timestamps,
    }
