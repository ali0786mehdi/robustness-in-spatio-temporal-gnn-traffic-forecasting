"""
ARIMA baseline model for traffic forecasting.
Efficient approach: fits once per sensor on training data, stores params,
then predicts independently for any (possibly corrupted) test input.
"""

import pickle
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class ARIMAForecaster:
    """
    ARIMA baseline for traffic speed forecasting.
    Fits one ARIMA per sensor on training data. Stores fitted params.
    Predict can then be called repeatedly on any (possibly corrupted) test_X.
    """

    def __init__(self, order=(3, 0, 1), max_sensors=None):
        self.order = order
        self.max_sensors = max_sensors
        self.fitted_params = {}   # sensor_idx -> np.ndarray of ARIMA params
        self.sensor_indices = None
        self.pred_len = None

    def fit(self, train_data, pred_len=12):
        """
        Fit ARIMA models for each sensor on training data.
        Stores fitted params for fast repeated prediction.

        Args:
            train_data: shape (T_train, N)
            pred_len:   number of steps to forecast
        """
        num_sensors = train_data.shape[1]
        self.pred_len = pred_len

        if self.max_sensors and self.max_sensors < num_sensors:
            np.random.seed(42)
            self.sensor_indices = np.sort(
                np.random.choice(num_sensors, self.max_sensors, replace=False)
            )
        else:
            self.sensor_indices = np.arange(num_sensors)

        print(f"Fitting ARIMA{self.order} on {len(self.sensor_indices)} sensors...")

        for sensor_idx in tqdm(self.sensor_indices, desc="ARIMA fit"):
            sensor_train = train_data[:, sensor_idx]
            try:
                model = ARIMA(sensor_train, order=self.order,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                fitted = model.fit(method_kwargs={"maxiter": 100})
                self.fitted_params[sensor_idx] = fitted.params
            except Exception:
                self.fitted_params[sensor_idx] = None  # Will fall back to persistence

        print(f"  ARIMA fitted on {len(self.fitted_params)} sensors.")

    def predict(self, test_X):
        """
        Generate predictions for a (possibly corrupted) test set.
        Uses stored fitted params — no re-fitting required.

        Args:
            test_X: shape (num_test, seq_len, N)

        Returns:
            predictions: shape (num_test, pred_len, N)
        """
        if not self.fitted_params:
            raise RuntimeError("ARIMAForecaster must be fit() before predict().")

        num_test, seq_len, num_sensors = test_X.shape
        pred_len = self.pred_len

        # Initialize with last-value (naive) as fallback
        predictions = np.zeros((num_test, pred_len, num_sensors))
        for t in range(num_test):
            predictions[t] = np.tile(test_X[t, -1, :], (pred_len, 1))

        for sensor_idx in self.sensor_indices:
            params = self.fitted_params.get(sensor_idx)
            if params is None:
                continue  # Keep naive prediction

            # Subsample test for speed (sample ~200 points, interpolate rest)
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

            # Fill non-sampled indices with nearest sampled forecast
            sorted_indices = sorted(forecasts.keys())
            for t in range(num_test):
                if t in forecasts:
                    predictions[t, :, sensor_idx] = forecasts[t]
                else:
                    nearest = min(sorted_indices, key=lambda x: abs(x - t))
                    predictions[t, :, sensor_idx] = forecasts[nearest]

        return predictions

    # ── Convenience wrapper (backward compatible) ─────────────────────────────
    def fit_and_predict(self, train_data, test_X, pred_len=12):
        """Fit then predict in one call. Backward compatible with run_baselines.py."""
        self.fit(train_data, pred_len=pred_len)
        return self.predict(test_X)

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, filepath):
        """Pickle the fitted params to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'max_sensors': self.max_sensors,
                'fitted_params': self.fitted_params,
                'sensor_indices': self.sensor_indices,
                'pred_len': self.pred_len,
            }, f)
        print(f"  ARIMA params saved to {filepath}")

    def load(self, filepath):
        """Load fitted params from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.order = state['order']
        self.max_sensors = state['max_sensors']
        self.fitted_params = state['fitted_params']
        self.sensor_indices = state['sensor_indices']
        self.pred_len = state['pred_len']

    def get_name(self):
        return f"ARIMA{self.order}"
