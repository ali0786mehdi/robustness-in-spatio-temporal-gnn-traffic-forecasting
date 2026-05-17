"""
LSTM baseline model for traffic forecasting.
Shared LSTM model across all sensors.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM model for traffic speed forecasting.

    Input: (batch, seq_len, num_sensors)
    Output: (batch, pred_len, num_sensors)
    """

    def __init__(self, num_sensors, seq_len=12, pred_len=12,
                 hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_sensors = num_sensors
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * num_sensors),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_sensors)
        Returns:
            (batch, pred_len, num_sensors)
        """
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        output = output.view(batch_size, self.pred_len, self.num_sensors)
        return output

    def get_name(self):
        return f"LSTM(h={self.hidden_dim}, L={self.num_layers})"
