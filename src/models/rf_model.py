"""
Random Forest baseline model for traffic forecasting.
Uses lagged speed values as features for per-sensor regression.
Optimized: subsamples training data for speed.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


class RandomForestForecaster:
    """
    Random Forest baseline for traffic speed forecasting.
    Uses lagged values (past seq_len steps) as features.
    Trains one RF per prediction horizon step.
    """

    def __init__(self, n_estimators=100, max_depth=15, n_jobs=-1,
                 max_train_samples=50000):
        """
        Args:
            n_estimators: Number of trees.
            max_depth: Maximum tree depth.
            n_jobs: Number of parallel jobs (-1 = all cores).
            max_train_samples: Max training samples (subsampled for speed).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.max_train_samples = max_train_samples
        self.models = {}

    def fit(self, train_X, train_Y):
        """
        Train Random Forest models.

        For each prediction horizon h:
          - Flatten input: (num_samples, seq_len, N) → (num_samples * N, seq_len)
          - Target: (num_samples, N) → (num_samples * N,)
          - Subsample if too large

        Args:
            train_X: Training inputs, shape (num_samples, seq_len, N).
            train_Y: Training targets, shape (num_samples, pred_len, N).
        """
        num_samples, seq_len, num_sensors = train_X.shape
        pred_len = train_Y.shape[1]

        # Reshape: treat each sensor independently
        # X: (num_samples * N, seq_len)
        X_flat = train_X.transpose(0, 2, 1).reshape(-1, seq_len)
        total_samples = X_flat.shape[0]

        # Subsample if too large
        if total_samples > self.max_train_samples:
            np.random.seed(42)
            idx = np.random.choice(total_samples, self.max_train_samples, replace=False)
            X_train = X_flat[idx]
            print(f"Training Random Forest (trees={self.n_estimators}, "
                  f"depth={self.max_depth}, "
                  f"subsampled {self.max_train_samples}/{total_samples})...")
        else:
            X_train = X_flat
            idx = None
            print(f"Training Random Forest (trees={self.n_estimators}, "
                  f"depth={self.max_depth}, samples={total_samples})...")

        for h in tqdm(range(pred_len), desc="RF horizons"):
            Y_flat = train_Y[:, h, :].reshape(-1)
            Y_train = Y_flat[idx] if idx is not None else Y_flat

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=self.n_jobs,
                random_state=42,
            )
            rf.fit(X_train, Y_train)
            self.models[h] = rf

        print(f"  Trained {pred_len} RF models")

    def predict(self, test_X):
        """
        Predict using trained Random Forest models.

        Args:
            test_X: Test inputs, shape (num_test, seq_len, N).

        Returns:
            Predictions, shape (num_test, pred_len, N).
        """
        num_test, seq_len, num_sensors = test_X.shape
        pred_len = len(self.models)

        X_flat = test_X.transpose(0, 2, 1).reshape(-1, seq_len)

        predictions = np.zeros((num_test, pred_len, num_sensors))

        for h in range(pred_len):
            Y_pred = self.models[h].predict(X_flat)
            predictions[:, h, :] = Y_pred.reshape(num_test, num_sensors)

        return predictions

    def get_name(self):
        return f"RandomForest(n={self.n_estimators})"
