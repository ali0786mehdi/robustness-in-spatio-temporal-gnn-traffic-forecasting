# Training Instructions — Full Clean Run (Both Datasets)

## Before You Start

```bash
cd ~/GraphNN
source venv/bin/activate
```

Verify GPU:
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

---

## ⚠️ Read This First

- Run every command in order, one at a time
- Wait for each to finish before running the next
- Do NOT use `--ablation` for the main training steps — that is only for Step 9
- If a run crashes, just re-run the same command — it restarts from epoch 1

---

## PHASE 1 — Train All Models on METR-LA

### Step 1 — DCRNN METR-LA (~2.5 hrs) ← DO THIS FIRST

```bash
python -u run_gnn.py --dataset METR-LA --model dcrnn
```

Saves to: `results/models/gnn_models/dcrnn/dcrnn_METR-LA_best.pt`

---

### Step 2 — STGCN METR-LA (~2.5 hrs)

```bash
python -u run_gnn.py --dataset METR-LA --model stgcn
```

Saves to: `results/models/gnn_models/stgcn/stgcn_METR-LA_best.pt`

---

### Step 3 — Baselines METR-LA (~45 min)

```bash
python -u run_baselines.py --dataset METR-LA
```

Saves to: `results/models/baseline_models/`
- `arima_METR-LA_best.pkl`
- `rf_METR-LA_best.pkl`
- `lstm_METR-LA_best.pt`

---

## PHASE 2 — Train All Models on PEMS-BAY

### Step 4 — DCRNN PEMS-BAY (~4–5 hrs)

```bash
python -u run_gnn.py --dataset PEMS-BAY --model dcrnn
```

Saves to: `results/models/gnn_models/dcrnn/dcrnn_PEMS-BAY_best.pt`

---

### Step 5 — STGCN PEMS-BAY (~3–4 hrs)

```bash
python -u run_gnn.py --dataset PEMS-BAY --model stgcn
```

Saves to: `results/models/gnn_models/stgcn/stgcn_PEMS-BAY_best.pt`

---

### Step 6 — Baselines PEMS-BAY (~45 min)

```bash
python -u run_baselines.py --dataset PEMS-BAY
```

Saves to: `results/models/baseline_models/`
- `arima_PEMS-BAY_best.pkl`
- `rf_PEMS-BAY_best.pkl`
- `lstm_PEMS-BAY_best.pt`

---

## PHASE 3 — Analysis (No Retraining)

### Step 7 — Robustness Experiment (~20 min per dataset)

Run only AFTER Phases 1 and 2 are fully complete.

```bash
python -u run_robustness.py --dataset METR-LA --n-seeds 5
python -u run_robustness.py --dataset PEMS-BAY --n-seeds 5
```

Saves to:
- `results/metrics/METR-LA_robustness.json`
- `results/metrics/PEMS-BAY_robustness.json`

---

### Step 8 — Graph Sparsity Analysis (~25 min)

```bash
python -u run_sparsity_analysis.py
```

Saves to: `results/metrics/METR-LA_sparsity_analysis.json`

---

### Step 8b — Graph Sparsity Ablation (~3–4 hrs)

This retrains STGCN and DCRNN at different graph sparsity thresholds (ε=0.1, 0.2, 0.3, 0.4, 0.5) to measure how graph density affects performance. The plot script in Step 10 reads from this.

```bash
python -u run_sparsity_ablation.py
```

Saves to: `results/metrics/METR-LA_sparsity_ablation.json`

> This is separate from Step 8. Step 8 is pure math (eigenvalues, spectral analysis).
> Step 8b actually retrains models — it takes longer but saves to a separate file.



## PHASE 4 — Ablation Studies (SAFE — separate files, production untouched)

### How save_tag works

```
--ablation identity  →  save_tag = "ablation_identity"  →  separate checkpoint file
--ablation random    →  save_tag = "ablation_random"    →  separate checkpoint file
no --ablation        →  production checkpoint (what you trained in Phase 1)
```

Production files are NEVER touched by ablation runs.

---

### Step 9a — Identity Graph Ablation (~5 hrs total)

Each sensor sees only itself — no spatial message passing at all.

```bash
python -u run_gnn.py --dataset METR-LA --ablation identity --model stgcn
python -u run_gnn.py --dataset METR-LA --ablation identity --model dcrnn
```

Saves to (separate from production):
```
results/models/gnn_models/stgcn/stgcn_METR-LA_ablation_identity_best.pt
results/models/gnn_models/dcrnn/dcrnn_METR-LA_ablation_identity_best.pt
results/metrics/METR-LA_gnn_ablation_identity.json
```

---

### Step 9b — Random Graph Ablation (~5 hrs total)

Same sparsity as learned graph, but random edges.

```bash
python -u run_gnn.py --dataset METR-LA --ablation random --model stgcn
python -u run_gnn.py --dataset METR-LA --ablation random --model dcrnn
```

Saves to (separate from production):
```
results/models/gnn_models/stgcn/stgcn_METR-LA_ablation_random_best.pt
results/models/gnn_models/dcrnn/dcrnn_METR-LA_ablation_random_best.pt
results/metrics/METR-LA_gnn_ablation_random.json
```

---

## PHASE 5 — Plots and Validation

### Step 10 — Generate Plots

```bash
python plot_robustness.py
python plot_sparsity_ablation.py
```

Saves to: `results/plots/`

---

### Step 10b — Regenerate Paper Assets (~1 min)

This regenerates all LaTeX tables and CSV summaries with the NEW metrics from retraining.  
**You MUST run this after retraining — otherwise the tables will have the old (wrong) numbers.**

```bash
python generate_paper_assets.py
```

Saves to: `results/paper_assets/`
- `table_main_results.tex` — updated model comparison table
- `table_robustness.tex` — updated robustness table (mean ± std)
- `main_results.csv` — machine-readable results
- `robustness_random_missing.csv` — updated robustness data
- `robustness_sensor_failure.csv` — updated sensor failure data
- `hyperparameters.md` — config documentation
- `reproducibility_checklist.md` — validation checklist

---

### Step 11 — Validate Everything

```bash
python validate.py --dataset METR-LA
python validate.py --dataset PEMS-BAY
```

All checkpoints should show as found. All metrics should load successfully.

---

### Step 12 — Push Results to GitHub

```bash
git add results/metrics/ results/plots/ results/paper_assets/
git commit -m "results: full clean retrain on METR-LA and PEMS-BAY"
git push
```

> Do NOT `git add results/models/` — model weights are gitignored.
> Only `results/metrics/`, `results/plots/`, and `results/paper_assets/` go to GitHub.

---

## Complete Time Estimate

| Step | Job | Est. Time |
|---|---|---|
| 1 | DCRNN METR-LA | ~2.5 hrs |
| 2 | STGCN METR-LA | ~2.5 hrs |
| 3 | Baselines METR-LA | ~45 min |
| 4 | DCRNN PEMS-BAY | ~4–5 hrs |
| 5 | STGCN PEMS-BAY | ~3–4 hrs |
| 6 | Baselines PEMS-BAY | ~45 min |
| 7 | Robustness (×2) | ~40 min |
| 8 | Sparsity analysis | ~25 min |
| 8b | Sparsity ablation | ~3–4 hrs |
| 9a | Identity ablation | ~5 hrs |
| 9b | Random ablation | ~5 hrs |
| 10b | **Paper assets** | **~1 min** |
| **Total** | | **~28–30 hrs** |

Steps 1–8b are the core results: **~17–20 hrs**  
Steps 9a–9b are ablations (optional but recommended): **+10 hrs**

---

## Quick Reference

| Script | Retrains? | Output location |
|---|---|---|
| `run_baselines.py` | Yes | `baseline_models/` |
| `run_gnn.py` | Yes | `gnn_models/stgcn/` or `dcrnn/` |
| `run_gnn.py --ablation X` | Yes (separate file) | `..._ablation_X_best.pt` |
| `run_robustness.py` | No | `metrics/*_robustness.json` |
| `run_sparsity_analysis.py` | No | `metrics/*_sparsity_analysis.json` |
| `run_sparsity_ablation.py` | Yes (retrains at each ε) | `metrics/*_sparsity_ablation.json` |
| `plot_robustness.py` | No | `plots/*_robustness_curves.png` |
| `plot_sparsity_ablation.py` | No | `plots/*_sparsity_ablation.png` |
| `generate_paper_assets.py` | No | `paper_assets/*.tex, *.csv, *.md` |
| `validate.py` | No | Console output only |

