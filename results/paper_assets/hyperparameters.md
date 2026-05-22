# Hyperparameter Configuration

## Training
- Batch size: 128
- Learning rate: 0.001
- Weight decay: 0.0001
- Max epochs: 100
- Early stopping patience: 15
- Gradient clip norm: 5.0
- LR scheduler: ReduceLROnPlateau(factor=0.5, patience=5)
- AMP enabled: True
- cuDNN benchmark: True
- Random seed: 42

## Data
- Input sequence length: 12 steps (= 60 min)
- Prediction horizon: 12 steps (= 60 min)
- Train/Val/Test split: 0.7/0.1/0.20000000000000004
- Normalization: Z-score per sensor (train stats only)

## Graph Construction
- Method: Pearson correlation → Gaussian kernel → threshold
- Gaussian σ: 0.1
- Sparsity threshold ε: 0.3
- Diffusion steps K: 2
- Data source: Training set only (no leakage)

## LSTM
- Hidden dim: 64
- Layers: 2
- Dropout: 0.3

## STGCN
- Channel progression: [1, 16, 32, 64]
- Temporal kernel size: 3
- Chebyshev order K: 3

## DCRNN
- Hidden dim: 64
- Layers: 2
- Dropout: 0.3
- Filter type: dual_random_walk
- Teacher forcing: linear decay over first 50% of epochs

## Random Forest
- Estimators: 100
- Max depth: 15

## ARIMA
- Order (p,d,q): (3, 0, 1)
- Max sensors: 30