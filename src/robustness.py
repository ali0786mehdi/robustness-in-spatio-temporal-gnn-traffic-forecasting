"""
Robustness evaluation module.
Provides functions to inject missing data and sensor failures into test sets.
"""

import numpy as np

def inject_random_missing(data, ratio=0.2, fill_value=0.0):
    """
    Randomly drop a percentage of all observations.

    Args:
        data (np.ndarray): Data array, shape (num_samples, seq_len, num_sensors) or (T, N).
        ratio (float): Fraction of data to drop (0.0 to 1.0).
        fill_value (float): Value to insert in place of dropped data.

    Returns:
        np.ndarray: Corrupted data array.
    """
    corrupted_data = data.copy()
    mask = np.random.rand(*corrupted_data.shape) < ratio
    corrupted_data[mask] = fill_value
    return corrupted_data

def inject_sensor_failure(data, ratio=0.1, fill_value=0.0):
    """
    Simulate complete failure of a percentage of sensors.
    The selected sensors will output `fill_value` for all timesteps.

    Args:
        data (np.ndarray): Data array, shape (num_samples, seq_len, num_sensors) or (T, N).
        ratio (float): Fraction of sensors to fail (0.0 to 1.0).
        fill_value (float): Value to output for failed sensors.

    Returns:
        np.ndarray: Corrupted data array.
    """
    corrupted_data = data.copy()
    
    # Determine the sensor dimension based on input shape
    if data.ndim == 3:
        num_sensors = data.shape[2]
    elif data.ndim == 2:
        num_sensors = data.shape[1]
    else:
        raise ValueError("Data must be 2D (T, N) or 3D (samples, seq_len, sensors)")

    num_failed = int(num_sensors * ratio)
    if num_failed == 0 and ratio > 0:
        num_failed = 1
        
    failed_sensors = np.random.choice(num_sensors, size=num_failed, replace=False)
    
    if data.ndim == 3:
        corrupted_data[:, :, failed_sensors] = fill_value
    else:
        corrupted_data[:, failed_sensors] = fill_value
        
    return corrupted_data
