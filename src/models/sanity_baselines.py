"""
Sanity Baseline Models
======================
Simple baselines to establish a lower bound for performance.
If a complex deep learning model cannot beat these, it is not learning effectively.
"""

import numpy as np

class PersistenceModel:
    """
    Persistence (Copy Last Value) Baseline.
    Assumes the future traffic speed will be exactly the same as the last observed speed.
    ŷ_{t+h} = x_t
    """
    def __init__(self, pred_len):
        self.pred_len = pred_len

    def predict(self, X):
        """
        Args:
            X (np.ndarray): Input sequences, shape (num_samples, seq_len, num_sensors).
                            Values should be de-normalized (actual speeds).
        Returns:
            np.ndarray: Predictions, shape (num_samples, pred_len, num_sensors).
        """
        # Take the last timestep from the input sequence
        last_observed = X[:, -1, :]  # Shape: (num_samples, num_sensors)
        
        # Tile it across the prediction horizon
        # Shape: (num_samples, pred_len, num_sensors)
        preds = np.tile(last_observed[:, np.newaxis, :], (1, self.pred_len, 1))
        return preds


class HistoricalAverageModel:
    """
    Historical Average Baseline.
    Computes the average traffic speed for each time-of-day interval
    (e.g., 288 intervals for 5-minute data) using the training set ONLY.
    """
    def __init__(self, intervals_per_day=288):
        self.intervals_per_day = intervals_per_day
        self.ha_table = None

    def fit(self, train_raw, timestamps):
        """
        Build the historical average table from training data.
        
        Args:
            train_raw (np.ndarray): Raw training data, shape (T_train, N).
            timestamps (pd.DatetimeIndex): Timestamps corresponding to the training data.
        """
        num_sensors = train_raw.shape[1]
        
        # Initialize table: (intervals_per_day, num_sensors)
        # Using a list to collect values for each interval, then average
        table_sums = np.zeros((self.intervals_per_day, num_sensors))
        table_counts = np.zeros((self.intervals_per_day, 1))
        
        # Calculate time-of-day index for each timestamp
        # e.g., 00:00 = 0, 00:05 = 1, ..., 23:55 = 287
        minutes_of_day = timestamps.hour * 60 + timestamps.minute
        interval_indices = (minutes_of_day // (24 * 60 // self.intervals_per_day)).astype(int)
        
        for t in range(len(train_raw)):
            idx = interval_indices[t]
            table_sums[idx] += train_raw[t]
            table_counts[idx] += 1
            
        # Avoid division by zero
        table_counts[table_counts == 0] = 1
        
        self.ha_table = table_sums / table_counts

    def predict(self, timestamps, num_sensors, pred_len):
        """
        Predict future values based on the historical average table.
        
        Args:
            timestamps (pd.DatetimeIndex): Timestamps of the TARGET sequences.
                                           Shape: (num_samples,) representing the time of the first predicted step.
            num_sensors (int): Number of sensors.
            pred_len (int): Prediction horizon.
            
        Returns:
            np.ndarray: Predictions, shape (num_samples, pred_len, num_sensors).
        """
        if self.ha_table is None:
            raise ValueError("HistoricalAverageModel must be fit before calling predict.")
            
        num_samples = len(timestamps)
        preds = np.zeros((num_samples, pred_len, num_sensors))
        
        # Calculate time-of-day index for the first target step
        minutes_of_day = timestamps.hour * 60 + timestamps.minute
        start_indices = (minutes_of_day // (24 * 60 // self.intervals_per_day)).astype(int)
        
        for i in range(num_samples):
            for h in range(pred_len):
                # Calculate the interval index for step h
                idx = (start_indices[i] + h) % self.intervals_per_day
                preds[i, h, :] = self.ha_table[idx]
                
        return preds
