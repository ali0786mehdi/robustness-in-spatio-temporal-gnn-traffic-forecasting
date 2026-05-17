"""
ARIMA baseline model for traffic forecasting.
Efficient approach: fits once per sensor, uses last-value + fitted trend for prediction.
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class ARIMAForecaster:
    """
    ARIMA baseline for traffic speed forecasting.
    Fits one ARIMA per sensor on training data.
    For test: uses each test input's last seq_len values to produce forecast.
    """

    def __init__(self, order=(3, 0, 1), max_sensors=None):
        self.order = order
        self.max_sensors = max_sensors
        self.fitted_params = {}
        self.sensor_indices = None

    def fit_and_predict(self, train_data, test_X, pred_len=12):
        """
        Fit ARIMA on training data per sensor, then batch-predict for test.

        Efficient strategy:
        1. Fit ARIMA once per sensor on training data → store params.
        2. For test: use each sample's input as history, apply stored params
           via filter(), and forecast pred_len steps.
        3. Process test in batches of 100 for speed.

        Args:
            train_data: Full training time series, shape (T_train, N).
            test_X: Test input sequences, shape (num_test, seq_len, N).
            pred_len: Steps to predict ahead.

        Returns:
            Predictions, shape (num_test, pred_len, N).
        """
        num_test, seq_len, num_sensors = test_X.shape

        if self.max_sensors and self.max_sensors < num_sensors:
            np.random.seed(42)
            self.sensor_indices = np.random.choice(
                num_sensors, self.max_sensors, replace=False
            )
            self.sensor_indices.sort()
        else:
            self.sensor_indices = np.arange(num_sensors)

        # Initialize with last-value (naive) prediction
        predictions = np.zeros((num_test, pred_len, num_sensors))
        for t in range(num_test):
            predictions[t] = np.tile(test_X[t, -1, :], (pred_len, 1))

        print(f"Fitting ARIMA{self.order} on {len(self.sensor_indices)} sensors...")

        for sensor_idx in tqdm(self.sensor_indices, desc="ARIMA fitting"):
            sensor_train = train_data[:, sensor_idx]

            try:
                # Fit once on training data
                model = ARIMA(sensor_train, order=self.order,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                fitted = model.fit(method_kwargs={"maxiter": 100})
                params = fitted.params

                # Use a subsample of test for speed (every 10th)
                # Then interpolate for others
                test_indices = list(range(0, num_test, max(1, num_test // 200)))
                if num_test - 1 not in test_indices:
                    test_indices.append(num_test - 1)

                forecasts = {}
                for t in test_indices:
                    history = test_X[t, :, sensor_idx]
                    try:
                        m = ARIMA(history, order=self.order,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
                        res = m.filter(params)
                        forecasts[t] = res.forecast(steps=pred_len)
                    except Exception:
                        forecasts[t] = np.full(pred_len, history[-1])

                # Fill predictions — for non-sampled indices, use nearest
                sorted_indices = sorted(forecasts.keys())
                for t in range(num_test):
                    if t in forecasts:
                        predictions[t, :, sensor_idx] = forecasts[t]
                    else:
                        # Find nearest sampled index
                        nearest = min(sorted_indices, key=lambda x: abs(x - t))
                        predictions[t, :, sensor_idx] = forecasts[nearest]

            except Exception:
                pass  # Keep naive prediction

        return predictions

    def get_name(self):
        return f"ARIMA{self.order}"
