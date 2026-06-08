# Robustness in Spatio-Temporal GNN Traffic Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/CUDA-Required-green?logo=nvidia" />
  <img src="https://img.shields.io/badge/License-Academic-lightgrey" />
  <img src="https://img.shields.io/badge/Status-Research%20Complete-brightgreen" />
</p>

> **Paper Focus:** *Robust comparative analysis of traditional and spatio-temporal graph neural network models for traffic forecasting under realistic sensor degradation conditions.*

A research-grade comparative benchmark evaluating the **accuracy**, **robustness**, and **efficiency** of traditional ML and Graph Neural Network models for traffic speed forecasting under realistic sensor degradation conditions across two large-scale real-world datasets.

---

## Table of Contents

- [Motivation](#motivation)
- [Key Contributions](#key-contributions)
- [Key Findings](#key-findings)
  - [Accuracy Rankings](#-accuracy-rankings-overall-mae-lower-is-better)
  - [Robustness Under Corruption](#-robustness-under-40-random-missing-data-metr-la)
  - [Core Research Insight](#-core-research-insight)
- [Models](#models)
- [Datasets](#datasets)
- [Per-Horizon Results](#per-horizon-results)
- [Ablation Studies](#ablation-studies)
- [Pipeline Integrity](#pipeline-integrity)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
  - [Environment Requirements](#environment-requirements)
  - [Quick Start](#quick-start)
  - [Full Training](#full-training)
  - [Evaluate & Generate Outputs](#evaluate--generate-outputs)
- [Performance Optimizations](#performance-optimizations)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Motivation

Urban traffic forecasting is a critical component of intelligent transportation systems — but real-world deployments are constantly subject to sensor failures, communication drops, and missing data. Most benchmarks evaluate models only under clean conditions, masking a critical gap: **a model that achieves state-of-the-art accuracy under clean data may be the worst-performing model when sensors degrade**.

This project directly addresses that gap by stress-testing seven models — from simple statistical baselines to state-of-the-art GNNs — under two realistic degradation scenarios: random missing data and localized sensor failure, across two large-scale highway datasets.

---

## Key Contributions

- **First systematic robustness comparison** of DCRNN and STGCN under controlled sensor degradation conditions on METR-LA and PEMS-BAY
- **Identified accuracy–robustness tradeoff** in spatio-temporal GNNs: DCRNN leads on MAE but is the least robust under random corruption
- **Mechanistic explanation** for observed tradeoffs: spectral smoothing (STGCN) vs. autoregressive recurrence (DCRNN) under noise
- **Revealed sensor-failure inversion**: non-graph LSTM outperforms both GNNs under localized failure due to graph message-passing propagating errors
- **Fully reproducible pipeline** with chronological splits, train-only normalization, multi-seed robustness evaluation, and committed metrics/plots
- **Graph structure ablation** isolating spatial learning contribution from temporal learning in GNN performance

---

## Key Findings

### 🏆 Accuracy Rankings (Overall MAE, lower is better)

| Rank | Model | METR-LA | PEMS-BAY | Type |
|------|-------|---------|---------|------|
| 1 | **DCRNN** | **3.548** | **1.905** | Graph Neural Network |
| 2 | STGCN | 3.634 | 2.299 | Graph Neural Network |
| 3 | LSTM | 3.707 | 2.071 | Deep Learning |
| 4 | Random Forest | 3.758 | 2.113 | Traditional ML |
| 5 | Persistence | 3.856 | 2.170 | Baseline |
| 6 | ARIMA | 3.990 | 2.294 | Classical |
| 7 | Historical Avg | 5.120 | 3.559 | Baseline |

### 🛡️ Robustness Under 40% Random Missing Data (METR-LA)

| Model | Clean MAE | Corrupted MAE | Degradation |
|-------|-----------|---------------|-------------|
| **STGCN** | 3.634 | 4.614 ± 0.004 | **27.0%** ← Most robust GNN |
| Persistence | 3.856 | 4.970 ± 0.003 | 28.9% |
| Random Forest | 3.758 | 4.897 ± 0.002 | 30.3% |
| LSTM | 3.707 | 4.885 ± 0.001 | 31.8% |
| **DCRNN** | 3.548 | 4.771 ± 0.004 | **34.5%** ← Least robust |

> Variance reported over 5 corruption seeds. Model weights are fixed; only corruption realizations vary.

### 💡 Core Research Insight

**DCRNN achieves the best accuracy but is the least robust under random data corruption.** STGCN trades ~2% accuracy for 27% better robustness — its spectral graph convolution (Chebyshev polynomials) smooths noise across the spatial domain, while DCRNN's autoregressive decoder amplifies corruption through sequential recurrence.

Under **sensor failure**, the pattern reverses: non-graph LSTM (31.1% degradation) is more stable than both GNNs because graph models propagate localized sensor failures through message passing.

**Implication for deployment:** In environments where sensor reliability is high, use DCRNN. In environments with frequent data gaps or sensor dropout, STGCN or LSTM may offer better real-world performance despite lower benchmark accuracy.

---

## Models

| Model | Spatial Learning | Temporal Learning | Parameters |
|-------|-----------------|-------------------|------------|
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
|----------|---------|---------|
| Sensors | 207 | 325 |
| Timesteps | 34,272 | 52,116 |
| Duration | Mar–Jun 2012 (4 months) | Jan–May 2017 (6 months) |
| Interval | 5 minutes | 5 minutes |
| Feature | Speed (mph) | Speed (mph) |
| Source | LA highway loop detectors | Bay Area Caltrans PeMS |

---

## Per-Horizon Results

### METR-LA

| Model | 15 min | 30 min | 60 min | Overall |
|-------|--------|--------|--------|---------|
| DCRNN | **2.868** | **3.540** | 4.541 | **3.548** |
| STGCN | 3.214 | 3.605 | **4.242** | 3.634 |
| LSTM | 2.988 | 3.685 | 4.759 | 3.707 |
| Random Forest | 3.049 | 3.753 | 4.798 | 3.758 |
| Persistence | 3.161 | 3.831 | 4.905 | 3.856 |
| ARIMA | 3.324 | 3.947 | 5.009 | 3.990 |
| Hist. Average | 5.120 | 5.120 | 5.120 | 5.120 |

> **Notable:** STGCN outperforms DCRNN at the 60-min horizon on METR-LA — spectral smoothing provides more stable long-range predictions.

### PEMS-BAY

| Model | 15 min | 30 min | 60 min | Overall |
|-------|--------|--------|--------|---------|
| DCRNN | **1.472** | **1.942** | **2.519** | **1.905** |
| LSTM | 1.523 | 2.083 | 2.864 | 2.071 |
| Random Forest | 1.553 | 2.135 | 2.920 | 2.113 |
| Persistence | 1.591 | 2.171 | 3.039 | 2.170 |
| ARIMA | 1.747 | 2.292 | 3.119 | 2.294 |
| STGCN | 2.077 | 2.302 | 2.604 | 2.299 |
| Hist. Average | 3.559 | 3.559 | 3.559 | 3.559 |

> **Notable:** STGCN ranks last overall on PEMS-BAY but has the best 60-min degradation curve, again reflecting spectral smoothing's long-horizon stability.

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

> Ablation checkpoints save to separate files (e.g., `stgcn_METR-LA_ablation_identity_best.pt`) and **never overwrite** production models.

### Cross-Dataset Transfer

Tests whether temporal patterns generalize across cities.

```bash
python run_cross_dataset.py --source PEMS-BAY --target METR-LA
```

> Only temporal models (LSTM, RF) support direct transfer. GNNs are bound to their training graph topology.

---

## Pipeline Integrity

All results are validated against research-grade standards:

| Check | Status | Description |
|-------|--------|-------------|
| Chronological splits | ✅ | No random shuffling of time-series data |
| Train-only normalization | ✅ | Mean/std computed from training set only |
| Train-only graph | ✅ | Adjacency built from training data, no test leakage |
| Per-chunk sequences | ✅ | No boundary leakage between train/val/test |
| De-normalized metrics | ✅ | MAE/RMSE/MAPE computed in original mph units |
| Model fairness | ✅ | All models share identical splits, horizons, and metrics |
| Multi-seed robustness | ✅ | 5 corruption seeds, reports mean ± std |
| Reproducibility | ✅ | Fixed seed (42), deterministic config, saved metadata |

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
├── installation.md                   # Step-by-step training instructions
├── SETUP.md                          # Environment setup guide
└── requirements.txt                  # Python dependencies
```

---

## Setup & Installation

### Environment Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11 |
| GPU VRAM | 8GB | 12GB+ (RTX 3060+) |
| RAM | 16GB | 32GB |
| CUDA | 11.8 | 12.x |
| Disk Space | ~5GB | ~10GB (with checkpoints) |

> 📖 For full environment setup including Windows/Linux CUDA installation and troubleshooting, see [SETUP.md](SETUP.md).

### Quick Start

Clone and install in one shot:

```bash
git clone https://github.com/kasim672/robustness-in-spatio-temporal-gnn-traffic-forecasting.git
cd robustness-in-spatio-temporal-gnn-traffic-forecasting

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Full Training

> ⏱️ **Total training time: ~14–16 hours on RTX 4060.** Run overnight. Results and plots auto-save to `results/`.

```bash
# Phase 1 — METR-LA
python -u run_gnn.py --dataset METR-LA --model dcrnn     # ~2.5 hrs
python -u run_gnn.py --dataset METR-LA --model stgcn     # ~2.5 hrs
python -u run_baselines.py --dataset METR-LA             # ~45 min

# Phase 2 — PEMS-BAY
python -u run_gnn.py --dataset PEMS-BAY --model dcrnn    # ~4–5 hrs
python -u run_gnn.py --dataset PEMS-BAY --model stgcn    # ~3–4 hrs
python -u run_baselines.py --dataset PEMS-BAY            # ~45 min
```

> 📖 For step-by-step training instructions with checkpointing and resume support, see [installation.md](installation.md).

### Evaluate & Generate Outputs

Run this after training (~30 min, no retraining needed):

```bash
# Robustness evaluation (5 seeds each)
python -u run_robustness.py --dataset METR-LA --n-seeds 5
python -u run_robustness.py --dataset PEMS-BAY --n-seeds 5

# Graph spectral analysis
python -u run_sparsity_analysis.py

# Generate plots and paper assets
python plot_robustness.py
python generate_paper_assets.py

# Validate full pipeline integrity
python validate.py --dataset METR-LA
python validate.py --dataset PEMS-BAY
```

---

## Performance Optimizations

| Optimization | Speedup |
|---|---|
| Automatic Mixed Precision (fp16) | ~1.5–1.8× |
| DataLoader (4 workers, persistent, prefetch) | ~1.1–1.3× |
| Batch size 128 (from 64) | ~1.3× |
| cuDNN benchmark auto-tuning | ~1.1× |
| **Combined** | **~2.5–3.5× faster training** |

---

## Reproducibility

```python
# All scripts use a fixed seed:
set_seed(42)  # Configured in src/config.py
```

- Robustness variance is over **corruption realizations, not model weights** — seeds `[0..4]` for corruption injection → reports mean ± std
- All metrics (`.json`) and plots (`.png`) are committed to the repository
- Model weights are gitignored (too large) — reproduce by following [installation.md](installation.md)

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{graphnn2026,
  title   = {Robustness in Spatio-Temporal GNN Traffic Forecasting},
  author  = {Kasim},
  year    = {2026},
  url     = {https://github.com/kasim672/robustness-in-spatio-temporal-gnn-traffic-forecasting}
}
```

---

## Acknowledgements

- **METR-LA** dataset: [DCRNN authors](https://github.com/liyaguang/DCRNN)
- **PEMS-BAY** dataset: California Department of Transportation [Caltrans PeMS](https://pems.dot.ca.gov/)
- Architecture references: [DCRNN (Li et al., 2018)](https://arxiv.org/abs/1707.01926) · [STGCN (Yu et al., 2018)](https://arxiv.org/abs/1709.04875)

---

## License

This project is intended for **academic research purposes only**.
