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
        fill_value: Scalar value to insert in place of dropped data.
                    If an array is passed, we take fill_value.flat[0] as the scalar.

    Returns:
        np.ndarray: Corrupted data array.
    """
    corrupted_data = data.copy()
    # Ensure fill_value is scalar — boolean fancy-indexing requires a 0-d / scalar
    scalar_fill = float(np.asarray(fill_value).flat[0])
    mask = np.random.rand(*corrupted_data.shape) < ratio
    corrupted_data[mask] = scalar_fill
    return corrupted_data


def inject_sensor_failure(data, ratio=0.1, fill_value=0.0):
    """
    Simulate complete failure of a percentage of sensors.
    The selected sensors will output `fill_value` for all timesteps.

    Args:
        data (np.ndarray): Data array, shape (num_samples, seq_len, num_sensors) or (T, N).
        ratio (float): Fraction of sensors to fail (0.0 to 1.0).
        fill_value: Scalar or per-sensor array to use for failed sensors.
                    If array, must broadcast to (num_sensors,) or be scalar.

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

    num_failed = max(1, int(num_sensors * ratio)) if ratio > 0 else 0
    if num_failed == 0:
        return corrupted_data

    failed_sensors = np.random.choice(num_sensors, size=num_failed, replace=False)

    # Build a per-sensor fill array of shape (num_sensors,)
    fill_arr = np.asarray(fill_value).flatten()
    if fill_arr.size == num_sensors:
        per_sensor_fill = fill_arr
    else:
        per_sensor_fill = np.full(num_sensors, float(fill_arr.flat[0]))

    if data.ndim == 3:
        for s in failed_sensors:
            corrupted_data[:, :, s] = per_sensor_fill[s]
    else:
        for s in failed_sensors:
            corrupted_data[:, s] = per_sensor_fill[s]

    return corrupted_data

