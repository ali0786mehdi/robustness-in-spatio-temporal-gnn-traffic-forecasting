# Training Instructions for Friend

## Before You Start

```bash
cd ~/GraphNN
source venv/bin/activate
```

Check GPU is available:
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

---

## ⚠️ IMPORTANT — Read This First

The METR-LA DCRNN checkpoint was accidentally overwritten by an ablation run.  
It **must be retrained first** before anything else.

> All commands below are safe to run in order. Nothing will overwrite production checkpoints — the `save_tag` system in the code handles this automatically.

---

## Step 1 — Fix DCRNN METR-LA (PRIORITY, ~2.5 hrs)

```bash
python -u run_gnn.py --dataset METR-LA --model dcrnn
```

Wait for this to finish completely before moving on.  
Saves to: `results/models/gnn_models/dcrnn/dcrnn_METR-LA_best.pt`

---

## Step 2 — Retrain STGCN METR-LA (~2.5 hrs)

```bash
python -u run_gnn.py --dataset METR-LA --model stgcn
```

Saves to: `results/models/gnn_models/stgcn/stgcn_METR-LA_best.pt`

---

## Step 3 — Baselines METR-LA (~30–45 min)

```bash
python -u run_baselines.py --dataset METR-LA
```

Trains ARIMA (slow), Random Forest, LSTM.  
Saves to: `results/models/baseline_models/`

---

## Step 4 — Baselines PEMS-BAY (~30–45 min)

```bash
python -u run_baselines.py --dataset PEMS-BAY
```

---

## Step 5 — PEMS-BAY GNNs (skip if checkpoints already exist)

Check if already done:
```bash
ls results/models/gnn_models/stgcn/stgcn_PEMS-BAY_best.pt
ls results/models/gnn_models/dcrnn/dcrnn_PEMS-BAY_best.pt
```

If files exist → skip this step.  
If missing → run:
```bash
python -u run_gnn.py --dataset PEMS-BAY --model stgcn
python -u run_gnn.py --dataset PEMS-BAY --model dcrnn
```

---

## Step 6 — Robustness Experiment (no retraining, ~20 min)

Run only AFTER Steps 1 and 2 are done.

```bash
python -u run_robustness.py --dataset METR-LA --n-seeds 5
```

---

## Step 7 — Sparsity Analysis (no retraining, ~25 min)

```bash
python -u run_sparsity_analysis.py
```

---

## Step 8 — Ablation Studies (SAFE — saves to separate files)

These will NOT overwrite any production checkpoints.  
Results save to separate files with `ablation_` prefix automatically.

```bash
# Identity graph ablation — STGCN only (~2.5 hrs)
python -u run_gnn.py --dataset METR-LA --ablation identity --model stgcn
```

Saves to:  
`results/models/gnn_models/stgcn/stgcn_METR-LA_ablation_identity_best.pt`  ← separate file  
`results/metrics/METR-LA_gnn_ablation_identity.json`

```bash
# Random graph ablation — STGCN only (~2.5 hrs)
python -u run_gnn.py --dataset METR-LA --ablation random --model stgcn
```

Saves to:  
`results/models/gnn_models/stgcn/stgcn_METR-LA_ablation_random_best.pt`  ← separate file  
`results/metrics/METR-LA_gnn_ablation_random.json`

---

## Step 9 — Generate All Plots

```bash
python plot_robustness.py
python run_sparsity_analysis.py  # already run in Step 7, skip if done
```

---

## Step 10 — Validate Everything

```bash
python validate.py --dataset METR-LA
python validate.py --dataset PEMS-BAY
```

Should show all checkpoints found and metrics loaded.

---

## Step 11 — Push Results to GitHub

```bash
git add results/metrics/ results/plots/
git commit -m "results: retrained METR-LA GNNs, robustness + ablation results"
git push
```

> Do NOT `git add results/models/` — model weights are gitignored (too large).  
> Only `results/metrics/*.json` and `results/plots/*.png` go to GitHub.

---

## Time Estimate Summary

| Step | Description | Time |
|---|---|---|
| 1 | DCRNN METR-LA retrain | ~2.5 hrs |
| 2 | STGCN METR-LA retrain | ~2.5 hrs |
| 3 | Baselines METR-LA | ~40 min |
| 4 | Baselines PEMS-BAY | ~40 min |
| 5 | PEMS-BAY GNNs (if needed) | ~5 hrs |
| 6 | Robustness (no training) | ~20 min |
| 7 | Sparsity analysis (no training) | ~25 min |
| 8 | Ablation × 2 (optional) | ~5 hrs total |
| **Total without ablations** | | **~6.5 hrs** |
| **Total with ablations** | | **~11.5 hrs** |

---

## If Something Fails

If a script crashes mid-run, just re-run the same command.  
Training is checkpointed — it will restart from epoch 1 but will not produce partial results.

If you see `[skip] checkpoint not found`, it means a previous step hasn't finished yet.  
Run the steps in order.

---

## Quick Reference — What Each Script Does

| Script | What it does | Retrains? |
|---|---|---|
| `run_baselines.py` | Trains ARIMA, RF, LSTM | Yes |
| `run_gnn.py` | Trains STGCN, DCRNN | Yes |
| `run_robustness.py` | Loads models, tests under corruption | No |
| `run_sparsity_analysis.py` | Spectral graph analysis | No |
| `plot_robustness.py` | Generates robustness figure | No |
| `validate.py` | Sanity checks everything | No |
