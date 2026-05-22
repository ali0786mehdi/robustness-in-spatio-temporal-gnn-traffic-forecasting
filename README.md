# Robustness in Spatio-Temporal GNN Traffic Forecasting

**A comparative benchmark of traditional ML and Graph Neural Networks for traffic speed prediction, with robustness analysis and graph topology ablation.**

> Models: Persistence · Historical Average · ARIMA · Random Forest · LSTM · STGCN · DCRNN  
> Datasets: METR-LA (207 sensors) · PEMS-BAY (325 sensors)

---

## 📁 Project Structure

```
GraphNN/
├── dataset/
│   ├── METR-LA.csv              # Los Angeles traffic (207 sensors, 34K timesteps)
│   └── PEMS-BAY.csv             # Bay Area traffic (325 sensors, 52K timesteps)
├── src/
│   ├── config.py                # All hyperparameters, paths, get_model_path()
│   ├── data_loader.py           # CSV loading, normalization, sequence creation
│   ├── graph_builder.py         # Correlation-based adjacency matrix
│   ├── evaluate.py              # MAE, RMSE, MAPE per horizon
│   ├── train.py                 # Training loop (early stopping, LR schedule)
│   ├── robustness.py            # Corruption injection (random missing, sensor failure)
│   └── models/
│       ├── sanity_baselines.py  # Persistence, Historical Average
│       ├── arima_model.py       # ARIMA (fit/predict/save/load)
│       ├── rf_model.py          # Random Forest (save/load)
│       ├── lstm_model.py        # Univariate LSTM per sensor
│       ├── stgcn.py             # STGCN — Chebyshev graph conv + temporal conv
│       └── dcrnn.py             # DCRNN — Bidirectional diffusion conv + GRU
├── results/
│   ├── metrics/                 # Saved JSON results (committed to git)
│   ├── plots/                   # Generated figures (committed to git)
│   └── models/
│       ├── gnn_models/
│       │   ├── stgcn/           # stgcn_<dataset>_best.pt
│       │   └── dcrnn/           # dcrnn_<dataset>_best.pt
│       └── baseline_models/     # lstm/arima/rf checkpoints
├── run_baselines.py             # Train ARIMA + RF + LSTM
├── run_gnn.py                   # Train STGCN + DCRNN (with ablation support)
├── run_robustness.py            # Multi-seed robustness experiment (no retraining)
├── run_sparsity_analysis.py     # Graph spectral analysis (no retraining)
├── run_sparsity_ablation.py     # Graph sparsity training ablation
├── run_cross_dataset.py         # Cross-dataset transfer experiment
├── plot_robustness.py           # Robustness curves with ±1σ error bands
├── plot_sparsity_ablation.py    # 4-panel sparsity figure
├── validate.py                  # Full pipeline sanity check
└── requirements.txt             # Python dependencies
```

---

## ⚙️ Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Training — Run in This Order

### Step 1: Baseline Models (fast, ~30–60 min total)

```bash
# METR-LA
python -u run_baselines.py --dataset METR-LA

# PEMS-BAY
python -u run_baselines.py --dataset PEMS-BAY
```

Trains: ARIMA (slow, ~20 min), Random Forest, LSTM.  
Saves to: `results/models/baseline_models/`

---

### Step 2: GNN Models (slow, GPU required)

> ⚠️ Run each separately. Each takes ~2.5 hrs on RTX 4060 (early stop ~epoch 35).

```bash
# METR-LA — STGCN first (faster, early stops)
python -u run_gnn.py --dataset METR-LA --model stgcn

# METR-LA — DCRNN (slower, runs full epochs)
python -u run_gnn.py --dataset METR-LA --model dcrnn

# PEMS-BAY (already trained — skip unless checkpoints are missing)
# python -u run_gnn.py --dataset PEMS-BAY
```

Saves to: `results/models/gnn_models/stgcn/` and `gnn_models/dcrnn/`

> **Important**: Never use `--ablation` without `--model` — ablation results save to separate files  
> (e.g. `stgcn_METR-LA_ablation_random_best.pt`) and never overwrite production checkpoints.

---

### Step 3: Robustness Experiment (no retraining, ~15–20 min)

```bash
python -u run_robustness.py --dataset METR-LA --n-seeds 5
```

Loads all trained models, evaluates under 0–40% corruption across 5 random seeds.  
Outputs `mean ± std MAE` per model per corruption ratio.  
Saves to: `results/metrics/METR-LA_robustness.json`

---

### Step 4: Graph Sparsity Analysis (no retraining, ~25 min)

```bash
python -u run_sparsity_analysis.py
```

Computes spectral properties (λ₂, noise ratio, degree distribution) at ε ∈ {0.1, 0.2, 0.3, 0.5}.  
No training required. Saves JSON + 6-panel figure.

---

### Step 5: Generate All Plots

```bash
python plot_robustness.py        # Robustness curves with ±1σ error bands
python plot_sparsity_ablation.py # Sparsity analysis figure (if results exist)
```

Saves to: `results/plots/`

---

### Step 6: Validate Everything

```bash
python validate.py --dataset METR-LA
python validate.py --dataset PEMS-BAY
```

Runs full sanity checks: data splits, normalization, graph integrity, checkpoint existence.

---

## 🔬 Ablation Studies

### Graph Structure Ablation (random / identity graph)

```bash
# Identity graph — no spatial edges (each sensor sees only itself)
python -u run_gnn.py --dataset METR-LA --ablation identity --model stgcn

# Random graph — same sparsity, random edges
python -u run_gnn.py --dataset METR-LA --ablation random --model stgcn
```

Results save to `results/metrics/METR-LA_gnn_ablation_<type>.json`.  
Checkpoints save to `gnn_models/stgcn/stgcn_METR-LA_ablation_<type>_best.pt` (separate from main).

### Cross-Dataset Transfer

```bash
python run_cross_dataset.py --source PEMS-BAY --target METR-LA
```

---

## 📊 Key Results (METR-LA, Multi-Seed Robustness)

| Model | MAE (clean) | MAE @40% missing | Degradation |
|---|---|---|---|
| Persistence | 3.856 | 4.970 ± 0.003 | +28.9% |
| Hist. Average | 5.120 | 5.120 ± 0.000 | 0% (immune) |
| ARIMA | 3.914 | 4.988 ± 0.008 | +27.4% |
| Random Forest | 3.758 | 4.897 ± 0.002 | +30.3% |
| LSTM | 3.721 | 4.878 ± 0.001 | +31.1% |
| **STGCN** | 3.744 | **4.584 ± 0.003** | **+22.4%** |
| DCRNN | 4.948 | 5.421 ± 0.001 | +9.6%* |

*DCRNN checkpoint is currently from ablation run — needs retraining (Step 2).

---

## 🗂 Graph Topology Findings (No Retraining)

| ε | Edges | Avg Conn | Isolated % | λ₂ (Fiedler) | Noise Ratio |
|---|---|---|---|---|---|
| 0.1 | 424 | 0.83 | 33.8% | 0.292 | 4.95 |
| 0.2 | 308 | 0.75 | 39.6% | 0.288 | 5.08 |
| **0.3** (default) | 240 | 0.67 | 48.3% | **0.294** | 5.12 |
| 0.5 | 148 | 0.49 | 59.9% | 0.329 | 4.77 |

> Even at ε=0.1 (~3× more edges), 1/3 of sensors remain isolated. Algebraic connectivity λ₂ barely changes, suggesting the correlation-based graph provides limited spatial signal regardless of threshold.

---

## 🔁 Reproducibility

```bash
# Fixed seed set in all scripts
set_seed(42)  # src/config.py

# Robustness variance is over corruption realizations, not model weights
# Seeds [0..4] used for corruption injection → reports mean ± std
```

All metrics JSON files are committed to git. Model weights are gitignored (too large).  
To reproduce from scratch: follow Steps 1–6 above.
