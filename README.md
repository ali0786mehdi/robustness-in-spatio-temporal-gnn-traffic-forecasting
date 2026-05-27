# Robustness in Spatio-Temporal GNN Traffic Forecasting

A research-grade comparative benchmark evaluating the **accuracy**, **robustness**, and **efficiency** of traditional ML and Graph Neural Network models for traffic speed forecasting under realistic sensor degradation conditions.

> **Paper Focus:** *Robust comparative analysis of traditional and spatio-temporal graph neural network models for traffic forecasting under realistic sensor degradation conditions.*

---

## Key Findings

### 🏆 Accuracy Rankings (Overall MAE, lower is better)

<table>
<tr><th>Rank</th><th>Model</th><th>METR-LA</th><th>PEMS-BAY</th><th>Type</th></tr>
<tr><td>1</td><td><b>DCRNN</b></td><td><b>3.548</b></td><td><b>1.905</b></td><td>Graph Neural Network</td></tr>
<tr><td>2</td><td>STGCN</td><td>3.634</td><td>2.299</td><td>Graph Neural Network</td></tr>
<tr><td>3</td><td>LSTM</td><td>3.707</td><td>2.071</td><td>Deep Learning</td></tr>
<tr><td>4</td><td>Random Forest</td><td>3.758</td><td>2.113</td><td>Traditional ML</td></tr>
<tr><td>5</td><td>Persistence</td><td>3.856</td><td>2.170</td><td>Baseline</td></tr>
<tr><td>6</td><td>ARIMA</td><td>3.990</td><td>2.294</td><td>Classical</td></tr>
<tr><td>7</td><td>Historical Avg</td><td>5.120</td><td>3.559</td><td>Baseline</td></tr>
</table>

### 🛡️ Robustness Under 40% Random Missing Data (METR-LA)

| Model | Clean MAE | Corrupted MAE | Degradation |
|---|---|---|---|
| **STGCN** | 3.634 | 4.614 ± 0.004 | **27.0%** ← Most robust GNN |
| Persistence | 3.856 | 4.970 ± 0.003 | 28.9% |
| Random Forest | 3.758 | 4.897 ± 0.002 | 30.3% |
| LSTM | 3.707 | 4.885 ± 0.001 | 31.8% |
| **DCRNN** | 3.548 | 4.771 ± 0.004 | **34.5%** ← Least robust |

### 💡 Core Research Insight

> **DCRNN achieves the best accuracy but is the least robust under random data corruption.** STGCN trades ~2% accuracy for 27% better robustness — its spectral graph convolution (Chebyshev polynomials) smooths noise across the spatial domain, while DCRNN's autoregressive decoder amplifies corruption through sequential recurrence.
>
> Under **sensor failure**, the pattern reverses: non-graph LSTM (31.1%) is more stable than both GNNs because graph models propagate localized sensor failures through message passing.

---

## Models

| Model | Spatial Learning | Temporal Learning | Parameters |
|---|---|---|---|
| Persistence | — | Copy last value | 0 |
| Historical Average | — | Time-of-day lookup | 0 |
| ARIMA(2,1,2) | — | Per-sensor ARIMA | N/A |
| Random Forest | — | Flatten → predict | ~450K |
| LSTM | — | 2-layer LSTM | ~55K |
| **STGCN** | Chebyshev spectral conv (K=3) | Gated 1D CNN | ~80K |
| **DCRNN** | Bidirectional diffusion conv (K=2) | DCGRU seq2seq | ~223K |

---

## Datasets

| Property | METR-LA | PEMS-BAY |
|---|---|---|
| Sensors | 207 | 325 |
| Timesteps | 34,272 | 52,116 |
| Duration | Mar–Jun 2012 (4 months) | Jan–May 2017 (6 months) |
| Interval | 5 minutes | 5 minutes |
| Feature | Speed (mph) | Speed (mph) |
| Source | LA highway loop detectors | Bay Area Caltrans PeMS |

---

## Project Structure

```
GraphNN/
├── src/                              # Core library
│   ├── config.py                     # Hyperparameters, paths, reproducibility
│   ├── data_loader.py                # CSV loading, normalization, sequence creation
│   ├── graph_builder.py              # Correlation-based adjacency, Chebyshev, diffusion
│   ├── evaluate.py                   # MAE, RMSE, MAPE at 15/30/60 min horizons
│   ├── train.py                      # Training loop (AMP, early stopping, LR schedule)
│   ├── robustness.py                 # Corruption injection (random missing, sensor failure)
│   ├── visualize.py                  # Plotting utilities
│   └── models/
│       ├── sanity_baselines.py       # Persistence, Historical Average
│       ├── arima_model.py            # ARIMA(2,1,2) per sensor
│       ├── rf_model.py               # Random Forest
│       ├── lstm_model.py             # 2-layer LSTM
│       ├── stgcn.py                  # Chebyshev graph conv + temporal conv
│       └── dcrnn.py                  # Diffusion conv + GRU seq2seq
├── dataset/
│   ├── METR-LA.csv                   # 207 sensors × 34K timesteps
│   └── PEMS-BAY.csv                  # 325 sensors × 52K timesteps
├── results/
│   ├── metrics/                      # JSON results (committed)
│   ├── plots/                        # Generated figures (committed)
│   ├── models/                       # Checkpoints (gitignored)
│   └── paper_assets/                 # LaTeX tables, CSVs (committed)
├── run_baselines.py                  # Train all baselines
├── run_gnn.py                        # Train STGCN / DCRNN (+ ablation support)
├── run_robustness.py                 # Multi-seed robustness evaluation
├── run_sparsity_analysis.py          # Graph spectral analysis
├── run_cross_dataset.py              # Cross-dataset transfer experiment
├── plot_robustness.py                # Robustness curves with ±1σ bands
├── generate_paper_assets.py          # LaTeX tables + CSV summaries
├── validate.py                       # Full pipeline sanity check
├── TRAINING_GUIDE.md                 # Step-by-step training instructions
├── SETUP.md                          # Environment setup guide
└── requirements.txt                  # Python dependencies
```

---

## Setup

```bash
# Clone
git clone https://github.com/kasim672/robustness-in-spatio-temporal-gnn-traffic-forecasting.git
cd robustness-in-spatio-temporal-gnn-traffic-forecasting

# Create environment
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, NVIDIA GPU with CUDA (RTX 3060+ recommended), 16GB RAM

> 📖 For detailed environment setup (Windows/Linux, CUDA installation, troubleshooting), see [SETUP](SETUP.md).

---

## Installation & Training

> 📖 For complete step-by-step training instructions, see [Installation](installation.md).

### Train everything (~14–16 hrs on RTX 4060)

```bash
# Phase 1 — METR-LA
python -u run_gnn.py --dataset METR-LA --model dcrnn     # ~2.5 hrs
python -u run_gnn.py --dataset METR-LA --model stgcn     # ~2.5 hrs
python -u run_baselines.py --dataset METR-LA             # ~45 min

# Phase 2 — PEMS-BAY
python -u run_gnn.py --dataset PEMS-BAY --model dcrnn    # ~4-5 hrs
python -u run_gnn.py --dataset PEMS-BAY --model stgcn    # ~3-4 hrs
python -u run_baselines.py --dataset PEMS-BAY            # ~45 min
```

### Evaluate and generate outputs (~30 min, no retraining)

```bash
# Robustness
python -u run_robustness.py --dataset METR-LA --n-seeds 5
python -u run_robustness.py --dataset PEMS-BAY --n-seeds 5

# Graph analysis
python -u run_sparsity_analysis.py

# Plots & paper assets
python plot_robustness.py
python generate_paper_assets.py

# Validate
python validate.py --dataset METR-LA
python validate.py --dataset PEMS-BAY
```

---

## Ablation Studies

### Graph Structure Ablation

Tests whether the learned graph structure actually helps, or if GNNs are just strong temporal learners.

```bash
# Identity graph — no spatial edges (each sensor isolated)
python -u run_gnn.py --dataset METR-LA --ablation identity --model stgcn
python -u run_gnn.py --dataset METR-LA --ablation identity --model dcrnn

# Random graph — same density, random edges
python -u run_gnn.py --dataset METR-LA --ablation random --model stgcn
python -u run_gnn.py --dataset METR-LA --ablation random --model dcrnn
```

Ablation checkpoints save to separate files (e.g., `stgcn_METR-LA_ablation_identity_best.pt`) and **never overwrite** production models.

### Cross-Dataset Transfer

Tests whether temporal patterns generalize across cities.

```bash
python run_cross_dataset.py --source PEMS-BAY --target METR-LA
```

> Only temporal models (LSTM, RF) support direct transfer. GNNs are bound to their training graph.

---

## Per-Horizon Results

### METR-LA

| Model | 15 min | 30 min | 60 min | Overall |
|---|---|---|---|---|
| DCRNN | **2.868** | **3.540** | 4.541 | **3.548** |
| STGCN | 3.214 | 3.605 | **4.242** | 3.634 |
| LSTM | 2.988 | 3.685 | 4.759 | 3.707 |
| Random Forest | 3.049 | 3.753 | 4.798 | 3.758 |
| Persistence | 3.161 | 3.831 | 4.905 | 3.856 |
| ARIMA | 3.324 | 3.947 | 5.009 | 3.990 |
| Hist. Average | 5.120 | 5.120 | 5.120 | 5.120 |

### PEMS-BAY

| Model | 15 min | 30 min | 60 min | Overall |
|---|---|---|---|---|
| DCRNN | **1.472** | **1.942** | **2.519** | **1.905** |
| LSTM | 1.523 | 2.083 | 2.864 | 2.071 |
| Random Forest | 1.553 | 2.135 | 2.920 | 2.113 |
| Persistence | 1.591 | 2.171 | 3.039 | 2.170 |
| ARIMA | 1.747 | 2.292 | 3.119 | 2.294 |
| STGCN | 2.077 | 2.302 | 2.604 | 2.299 |
| Hist. Average | 3.559 | 3.559 | 3.559 | 3.559 |

---

## Pipeline Integrity

All results are validated against research-grade standards:

- ✅ **Chronological splits** — no random shuffling of time-series data
- ✅ **Train-only normalization** — mean/std computed from training set only
- ✅ **Train-only graph** — adjacency built from training data, no test leakage
- ✅ **Per-chunk sequences** — no boundary leakage between train/val/test
- ✅ **De-normalized metrics** — MAE/RMSE/MAPE computed in original mph units
- ✅ **Model fairness** — all models share identical splits, horizons, and metrics
- ✅ **Multi-seed robustness** — 5 corruption seeds, reports mean ± std
- ✅ **Reproducibility** — fixed seed (42), deterministic config, saved metadata

---

## Performance Optimizations

| Optimization | Impact |
|---|---|
| Automatic Mixed Precision (fp16) | ~1.5–1.8× speedup |
| DataLoader (4 workers, persistent, prefetch) | ~1.1–1.3× speedup |
| Batch size 128 (from 64) | ~1.3× speedup |
| cuDNN benchmark auto-tuning | ~1.1× speedup |
| **Combined** | **~2.5–3.5× faster training** |

---

## Reproducibility

```python
# All scripts use:
set_seed(42)  # Fixed in src/config.py

# Robustness variance is over corruption realizations, not model weights
# Seeds [0..4] for corruption injection → reports mean ± std
```

All metrics (`.json`) and plots (`.png`) are committed to git.  
Model weights are gitignored (too large) — reproduce by following the training guide.

---

## Citation

If you use this benchmark in your research:

```bibtex
@misc{graphnn2026,
  title={Robustness in Spatio-Temporal GNN Traffic Forecasting},
  author={Kasim},
  year={2026},
  url={https://github.com/kasim672/robustness-in-spatio-temporal-gnn-traffic-forecasting}
}
```

---

## License

This project is for academic research purposes.
