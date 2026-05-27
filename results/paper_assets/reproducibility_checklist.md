# Reproducibility Checklist

- ✅ Fixed random seed (42)
- ✅ Deterministic PyTorch (when CUDNN_BENCHMARK=False)
- ✅ Chronological train/val/test splits (no shuffling)
- ✅ Normalization stats from training set only
- ✅ Graph built from training data only
- ✅ Per-chunk sequence creation (no boundary leakage)
- ✅ Metrics computed on de-normalized predictions
- ✅ All models share identical data pipeline
- ✅ Multi-seed robustness (5 seeds, reports mean ± std)
- ✅ Early stopping on validation loss
- ✅ Model checkpoints saved (gitignored for size)
- ✅ All hyperparameters documented in config.py
- ✅ Results saved as JSON (committed to git)