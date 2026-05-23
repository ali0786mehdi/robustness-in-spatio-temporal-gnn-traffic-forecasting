# Environment Setup Guide

## Requirements

- **Python:** 3.11 (recommended) or 3.10
- **GPU:** NVIDIA GPU with CUDA support (RTX 3060 or higher recommended)
- **OS:** Windows 10/11 or Linux (Ubuntu 20.04+)
- **Disk:** ~2 GB for datasets + models
- **RAM:** 16 GB minimum

---

## Windows Setup

### 1. Install Python 3.11

Download from: https://www.python.org/downloads/release/python-3119/

Choose: **Windows installer (64-bit)**

> ⚠️ During installation, CHECK the box **"Add python.exe to PATH"**. If you miss this, Python won't work from the command line.

Verify:
```cmd
python --version
```
Expected: `Python 3.11.x`

---

### 2. Install Git

Download from: https://git-scm.com/download/win

Use all default options during installation.

Verify:
```cmd
git --version
```

---

### 3. Clone the Repository

```cmd
git clone https://github.com/kasim672/robustness-in-spatio-temporal-gnn-traffic-forecasting.git
cd robustness-in-spatio-temporal-gnn-traffic-forecasting
```

---

### 4. Create Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your command prompt.

---

### 5. Install PyTorch with CUDA

Pick the command for your GPU:

**RTX 4090 / 4080 / 4070 / 4060 (Ada Lovelace — CUDA 12.6):**
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**RTX 3090 / 3080 / 3070 / 3060 (Ampere — CUDA 11.8):**
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**No GPU (CPU only — training will be very slow):**
```cmd
pip install torch
```

---

### 6. Install Other Dependencies

```cmd
pip install numpy pandas scikit-learn statsmodels matplotlib seaborn scipy tqdm
```

---

### 7. Verify Installation

```cmd
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output (example for RTX 4090):
```
PyTorch: 2.x.x
CUDA: True
GPU: NVIDIA GeForce RTX 4090
```

Full project verification:
```cmd
python -c "from src import config; config.set_seed(); print('Device:', config.get_device()); print('Setup complete!')"
```

---

## Linux Setup

### 1. Install Python 3.11

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip git
```

**Already have Python 3.10+:**
```bash
python3 --version   # If 3.10 or 3.11, you're good
```

---

### 2. Clone the Repository

```bash
git clone https://github.com/kasim672/robustness-in-spatio-temporal-gnn-traffic-forecasting.git
cd robustness-in-spatio-temporal-gnn-traffic-forecasting
```

---

### 3. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

---

### 4. Install PyTorch with CUDA

**RTX 40-series (CUDA 12.6):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**RTX 30-series (CUDA 11.8):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

### 5. Install Other Dependencies

```bash
pip install numpy pandas scikit-learn statsmodels matplotlib seaborn scipy tqdm
```

---

### 6. Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Full project verification:
```bash
python -c "from src import config; config.set_seed(); print('Device:', config.get_device()); print('Setup complete!')"
```

---

## Dataset Setup

The datasets should already be in the repository:

```
dataset/
├── METR-LA.csv     (~50 MB)
└── PEMS-BAY.csv    (~80 MB)
```

If missing, download from the original sources:
- METR-LA: https://github.com/liyaguang/DCRNN
- PEMS-BAY: https://github.com/liyaguang/DCRNN

Place CSV files in the `dataset/` directory.

---

## GPU Batch Size Guide

Adjust `BATCH_SIZE` in `src/config.py` based on your GPU memory:

| GPU | VRAM | Recommended BATCH_SIZE |
|---|---|---|
| RTX 3060 | 12 GB | 128 |
| RTX 4060 | 8 GB | 128 |
| RTX 4070 | 12 GB | 192 |
| RTX 4080 | 16 GB | 256 |
| RTX 4090 | 24 GB | 256–512 |
| CPU only | — | 32 |

The default is `128`, which is safe for 8 GB+ GPUs with AMP enabled.

---

## Troubleshooting

### "torch.cuda.is_available() returns False"

1. Check NVIDIA driver: `nvidia-smi` (should show your GPU)
2. If no driver: download from https://www.nvidia.com/drivers
3. Make sure you installed the CUDA version of PyTorch (not CPU)
4. Reinstall: `pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu126`

### "ModuleNotFoundError: No module named 'src'"

Make sure you are running commands from the project root directory:
```cmd
cd robustness-in-spatio-temporal-gnn-traffic-forecasting
```

### "CUDA out of memory"

Reduce `BATCH_SIZE` in `src/config.py`:
```python
BATCH_SIZE = 64   # reduce from 128
```

### Windows: "python is not recognized"

Python was installed without adding to PATH. Either:
- Reinstall Python and check "Add to PATH"
- Or use the full path: `C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe`

---

## Next Steps

Once setup is complete, follow `TRAINING_GUIDE.md` to run the full training pipeline.
