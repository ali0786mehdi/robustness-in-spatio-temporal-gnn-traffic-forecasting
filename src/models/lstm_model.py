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

        # Treat each sensor as an independent time series (univariate LSTM)
        # This isolates temporal modeling from spatial modeling (which GNNs handle)
        self.lstm = nn.LSTM(
            input_size=1,  # 1 feature (speed) per sensor
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),  # Output pred_len steps for 1 sensor
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_sensors)
        Returns:
            (batch, pred_len, num_sensors)
        """
        batch_size = x.size(0)
        
        # Reshape to (batch * num_sensors, seq_len, 1)
        # This allows the LSTM to process each sensor's history independently
        x_reshaped = x.transpose(1, 2).reshape(batch_size * self.num_sensors, -1, 1)
        
        lstm_out, _ = self.lstm(x_reshaped)
        last_out = lstm_out[:, -1, :]  # (batch * num_sensors, hidden_dim)
        
        output = self.fc(last_out)     # (batch * num_sensors, pred_len)
        
        # Reshape back to (batch, num_sensors, pred_len) then transpose to (batch, pred_len, num_sensors)
        output = output.view(batch_size, self.num_sensors, self.pred_len).transpose(1, 2)
        return output

    def get_name(self):
        return f"LSTM(h={self.hidden_dim}, L={self.num_layers})"
