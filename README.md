# Traffic Forecasting: GNN vs Traditional Models

**A Comparative Study of Traditional Machine Learning Models and Graph Neural Networks for Traffic Speed Prediction**

> Research project comparing ARIMA, Random Forest, LSTM (temporal-only models) against STGCN and DCRNN (spatio-temporal graph models) on real-world traffic sensor datasets.

---

## 📁 Project Structure

```
GraphNN/
├── dataset/
│   ├── METR-LA.csv              # Los Angeles traffic (207 sensors, 34K timesteps)
│   └── PEMS-BAY.csv             # Bay Area traffic (325 sensors, 52K timesteps)
├── src/
│   ├── config.py                # All hyperparameters & paths
│   ├── data_loader.py           # CSV loading, normalization, sequences
│   ├── graph_builder.py         # Adjacency matrix from correlations
│   ├── evaluate.py              # MAE, RMSE, MAPE metrics
│   ├── train.py                 # Training loop (early stopping, LR schedule)
│   ├── visualize.py             # All plots and charts
│   └── models/
│       ├── arima_model.py       # ARIMA baseline (statistical)
│       ├── rf_model.py          # Random Forest baseline (ML)
│       ├── lstm_model.py        # LSTM baseline (deep learning)
│       ├── stgcn.py             # STGCN — Graph Conv + Temporal Conv
│       └── dcrnn.py             # DCRNN — Diffusion Conv + GRU
├── results/
│   ├── metrics/                 # Saved JSON results
│   ├── models/                  # Saved model checkpoints (.pt)
│   └── plots/                   # Generated visualization plots
├── run_baselines.py             # Run ARIMA + RF + LSTM
├── run_gnn.py                   # Run STGCN + DCRNN
├── run_all.py                   # Run everything + comparison + plots
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🔧 Installation

### 1. Create virtual environment

```bash
cd GraphNN
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch ≥ 2.0, NumPy, Pandas, Scikit-learn, Statsmodels, Matplotlib, Seaborn, SciPy, tqdm

### 3. Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

> **Note:** GPU (CUDA) is recommended for STGCN and DCRNN training. CPU works but will be significantly slower.

---

## 📊 Datasets

| Dataset | Sensors | Timesteps | Interval | Duration | Feature |
|---------|---------|-----------|----------|----------|---------|
| **METR-LA** | 207 | 34,272 | 5 min | Mar–Jun 2012 | Speed (mph) |
| **PEMS-BAY** | 325 | 52,116 | 5 min | Jan–May 2017 | Speed (mph) |

Both CSVs should be placed in the `dataset/` folder (already included).

---

## 🚀 How to Train & Get Results

Following our research strategy:
* **Classical Baselines** (ARIMA, Random Forest, LSTM) are trained and evaluated on **both** datasets (METR-LA and PEMS-BAY) to provide complete cross-dataset coverage and consistency with prior work.
* **Graph Neural Network Models** (STGCN, DCRNN) are trained and evaluated **only on PEMS-BAY**, which is the larger, harder, and more representative benchmark. This limits high computational overhead while keeping the GNN analysis robust.

### Running Baselines (METR-LA & PEMS-BAY)

To run the classical baselines on both datasets:
```bash
# Run ARIMA, RF, LSTM on METR-LA
python run_baselines.py --dataset METR-LA

# Run ARIMA, RF, LSTM on PEMS-BAY
python run_baselines.py --dataset PEMS-BAY
```

### Running Graph Neural Networks (PEMS-BAY Only)

To train and evaluate GNN models on the primary benchmark:
```bash
# Run STGCN and DCRNN on PEMS-BAY
python run_gnn.py --dataset PEMS-BAY
```

---

## ⏱️ Expected Training Times (RTX 4060, 8GB)

| Model | METR-LA | PEMS-BAY |
|-------|---------|----------|
| ARIMA | ~10 min | ~15 min |
| Random Forest | ~1 min | ~2 min |
| LSTM | ~5 min | ~8 min |
| STGCN | *N/A (Skipped)* | ~10 min |
| DCRNN | *N/A (Skipped)* | ~20 min |
| **Total** | **~16 min** | **~55 min** |

> On CPU only: multiply deep model times by ~5–10x.


---

## ⚙️ Configuration

All hyperparameters are in **`src/config.py`**. Key settings you may want to adjust:

```python
# Data split
TRAIN_RATIO = 0.7       # 70% training
VAL_RATIO = 0.1         # 10% validation
TEST_RATIO = 0.2        # 20% testing

# Sequence parameters
SEQ_LEN = 12            # Input: 12 steps = 1 hour
PRED_LEN = 12           # Predict: 12 steps = 1 hour ahead

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 15            # Early stopping

# Graph construction
GRAPH_SIGMA = 0.1        # Gaussian kernel bandwidth
GRAPH_EPSILON = 0.3      # Sparsity threshold

# ARIMA
ARIMA_MAX_SENSORS = 30   # Fit ARIMA on subset for speed

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
```

---

## 📈 Evaluation Metrics

All metrics are computed on **de-normalized** predictions (actual speed in mph).

| Metric | Formula | Description |
|--------|---------|-------------|
| **MAE** | mean(\|y - ŷ\|) | Mean Absolute Error |
| **RMSE** | √mean((y - ŷ)²) | Root Mean Squared Error |
| **MAPE** | mean(\|y - ŷ\| / \|y\|) × 100 | Mean Absolute Percentage Error |

Results are reported at three horizons (standard in traffic forecasting):
- **15 min** (3 steps ahead)
- **30 min** (6 steps ahead)
- **60 min** (12 steps ahead)

---

## 🏗️ Model Architectures

### Traditional Baselines (No Graph Structure)

| Model | Type | How It Works |
|-------|------|-------------|
| **ARIMA(3,0,1)** | Statistical | Per-sensor univariate time series model |
| **Random Forest** | ML | Lagged values as features, one RF per horizon |
| **LSTM** | Deep Learning | Shared multi-layer LSTM across all sensors |

### Graph Neural Networks (Uses Road Network Structure)

| Model | Spatial | Temporal | Paper |
|-------|---------|----------|-------|
| **STGCN** | Chebyshev Graph Conv | Gated 1D Temporal Conv | Yu et al., 2018 |
| **DCRNN** | Diffusion Graph Conv | GRU (Seq2Seq) | Li et al., 2018 |

### Key Difference

- **Traditional models** learn from time patterns only (each sensor independently or jointly)
- **GNN models** also learn from the **spatial graph structure** — how sensors are connected on the road network — capturing traffic propagation patterns

---

## 📊 Output Files

After training, you will find:

### `results/metrics/`
- `METR-LA_all_results.json` — All model metrics in JSON format
- Can be loaded for custom analysis

### `results/models/`
- `lstm_METR-LA_best.pt` — Best LSTM checkpoint
- `stgcn_METR-LA_best.pt` — Best STGCN checkpoint
- `dcrnn_METR-LA_best.pt` — Best DCRNN checkpoint

### `results/plots/`
- `METR-LA_comparison.png` — Bar chart comparing all models
- `METR-LA_horizons.png` — Error vs prediction horizon
- `METR-LA_pred_sensor0.png` — Prediction vs ground truth
- `METR-LA_adjacency.png` — Graph adjacency heatmap
- `METR-LA_training.png` — Training/validation loss curves
- `METR-LA_STGCN_spatial.png` — Per-sensor error distribution

---

## 🔬 Research Workflow

### For your paper, follow this training order:

1. **Verify with Baselines on METR-LA** — Run ARIMA, Random Forest, and LSTM on the smaller METR-LA dataset to ensure everything works smoothly.
2. **Train Baselines on PEMS-BAY** — Run ARIMA, Random Forest, and LSTM on the larger PEMS-BAY dataset to establish complete benchmark coverage across both road networks.
3. **Train GNNs on PEMS-BAY only** — Run STGCN and DCRNN on PEMS-BAY. This focuses your computational resources on the most challenging, representative benchmark (325 sensors vs 207).
4. **Collect and Analyze** the comparative results. Use the generated plots and metrics to support your findings.

### Writing Justification for Your Paper
When writing the methodology or experimental setup in your paper, you can present this strategy as follows:
> *"To ensure a broad and robust evaluation, we benchmark classical temporal baselines (ARIMA, Random Forest, LSTM) on both METR-LA and PEMS-BAY datasets. In contrast, due to their substantially higher computational overhead and spatial complexity, the spatio-temporal Graph Neural Network models (STGCN, DCRNN) are evaluated on PEMS-BAY, which serves as our primary, most representative, and computationally demanding benchmark."*

### Suggested Paper Structure

| Chapter | Content |
|---------|---------|
| 1. Introduction | Traffic forecasting problem, why spatial-temporal GNNs are preferred over pure temporal models. |
| 2. Literature Review | Traditional forecasting (ARIMA), machine learning (Random Forest), deep learning (LSTM), and GNNs (STGCN, DCRNN). |
| 3. Methodology | Data cleaning, Z-score normalization, sequence generation, correlation-based graph construction, and model architectures. |
| 4. Experimental Setup | Dataset descriptions, train/val/test splits, configuration details, and computational resource allocation strategy. |
| 5. Results & Discussion | Model comparison tables across horizons, error growth analysis, GNN performance improvement margins. |
| 6. Conclusion | Summary of findings, model strengths/weaknesses, and directions for future research. |

---

## 🔁 Reproducibility

- **Random seed**: Fixed at 42 (configurable in `config.py`)
- **Data splits**: Chronological (no random shuffle)
- **Early stopping**: Prevents overfitting
- **All hyperparameters**: Documented in `config.py`

---

## 📚 References

- **STGCN**: Yu, B., Yin, H., & Zhu, Z. (2018). *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.* IJCAI 2018.
- **DCRNN**: Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* ICLR 2018.
- **METR-LA / PEMS-BAY**: Li, Y., et al. (2018). Same as DCRNN paper.

---

## 💡 Tips

- If you run out of GPU memory, reduce `BATCH_SIZE` in `config.py` (try 32 or 16)
- If ARIMA is too slow, reduce `ARIMA_MAX_SENSORS` (default: 30)
- To skip ARIMA entirely, comment out the ARIMA block in `run_all.py`
- For faster experiments, reduce `EPOCHS` to 50
