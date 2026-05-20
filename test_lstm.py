import torch
from src.models.lstm_model import LSTMModel

model = LSTMModel(num_sensors=207)
x = torch.randn(32, 12, 207) # batch, seq_len, num_sensors
y = model(x)
print("Input:", x.shape)
print("Output:", y.shape)
