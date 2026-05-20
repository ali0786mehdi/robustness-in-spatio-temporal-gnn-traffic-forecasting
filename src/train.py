"""
Training module for deep learning models (LSTM, STGCN, DCRNN).
Handles training loop, validation, early stopping, and checkpointing.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience=15, min_delta=1e-4, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_model(model, train_loader, val_loader, config, model_name,
                dataset_name, graph_data=None):
    """
    Generic training loop for LSTM, STGCN, and DCRNN.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Config module with hyperparameters.
        model_name: 'lstm', 'stgcn', or 'dcrnn'.
        dataset_name: 'METR-LA' or 'PEMS-BAY'.
        graph_data: Dict with graph matrices (needed for STGCN/DCRNN).

    Returns:
        dict: Training history with train_loss and val_loss per epoch.
    """
    device = config.DEVICE
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
    )
    criterion = nn.MSELoss()

    save_path = os.path.join(
        config.RESULTS_DIR, "models", f"{model_name}_{dataset_name}_best.pt"
    )
    early_stopping = EarlyStopping(
        patience=config.PATIENCE, save_path=save_path
    )

    # Prepare graph tensors if needed
    cheb_polys_tensor = None
    supports_tensor = None

    if model_name == 'stgcn' and graph_data is not None:
        cheb_polys_tensor = [
            torch.FloatTensor(p).to(device) for p in graph_data['cheb_polys']
        ]
    elif model_name == 'dcrnn' and graph_data is not None:
        supports_tensor = [
            torch.FloatTensor(s).to(device) for s in graph_data['diffusion_supports']
        ]

    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on {dataset_name}")
    print(f"Device: {device} | Epochs: {config.EPOCHS} | LR: {config.LEARNING_RATE}")
    print(f"{'='*60}")

    for epoch in range(config.EPOCHS):
        # ---- Training ----
        model.train()
        train_losses = []
        t_start = time.time()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            if model_name == 'stgcn':
                pred = model(batch_x, cheb_polys_tensor)
            elif model_name == 'dcrnn':
                # Teacher forcing ratio decays over epochs
                tf_ratio = max(0.0, 1.0 - epoch / (config.EPOCHS * 0.5))
                pred = model(batch_x, supports_tensor, batch_y, tf_ratio)
            else:
                pred = model(batch_x)

            loss = criterion(pred, batch_y)
            loss.backward()

            if config.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # ---- Validation ----
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                if model_name == 'stgcn':
                    pred = model(batch_x, cheb_polys_tensor)
                elif model_name == 'dcrnn':
                    pred = model(batch_x, supports_tensor)
                else:
                    pred = model(batch_x)

                loss = criterion(pred, batch_y)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        elapsed = time.time() - t_start
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        if 'epoch_times' not in history:
            history['epoch_times'] = []
        history['epoch_times'].append(elapsed)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
        print(f"  Loaded best model from {save_path}")

    print(f"  Best val loss: {early_stopping.best_loss:.6f}")

    # --- Efficiency Metrics ---
    param_count = sum(p.numel() for p in model.parameters())
    peak_gpu = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0
    avg_epoch_time = np.mean(history.get('epoch_times', []))
    
    history['efficiency'] = {
        'param_count': param_count,
        'peak_gpu_mb': peak_gpu,
        'train_time_per_epoch_s': avg_epoch_time,
    }

    return history


def predict_model(model, test_loader, config, model_name, graph_data=None):
    """
    Generate predictions on test set.

    Args:
        model: Trained PyTorch model.
        test_loader: Test DataLoader.
        config: Config module.
        model_name: 'lstm', 'stgcn', or 'dcrnn'.
        graph_data: Graph matrices.

    Returns:
        tuple: (predictions, ground_truth) both as numpy arrays
               shape (num_test, pred_len, num_sensors)
    """
    device = config.DEVICE
    model = model.to(device)
    model.eval()

    cheb_polys_tensor = None
    supports_tensor = None

    if model_name == 'stgcn' and graph_data is not None:
        cheb_polys_tensor = [
            torch.FloatTensor(p).to(device) for p in graph_data['cheb_polys']
        ]
    elif model_name == 'dcrnn' and graph_data is not None:
        supports_tensor = [
            torch.FloatTensor(s).to(device) for s in graph_data['diffusion_supports']
        ]

    all_preds = []
    all_targets = []
    inference_times = []

    with torch.no_grad():
        # Warmup for more accurate GPU timing
        warmup_batches = 3
        for i, (batch_x, batch_y) in enumerate(test_loader):
            if i >= warmup_batches:
                break
            batch_x = batch_x.to(device)
            if model_name == 'stgcn':
                _ = model(batch_x, cheb_polys_tensor)
            elif model_name == 'dcrnn':
                _ = model(batch_x, supports_tensor)
            else:
                _ = model(batch_x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            
            t_start = time.time()
            if model_name == 'stgcn':
                pred = model(batch_x, cheb_polys_tensor)
            elif model_name == 'dcrnn':
                pred = model(batch_x, supports_tensor)
            else:
                pred = model(batch_x)
                
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_times.append(time.time() - t_start)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())

    predictions = np.concatenate(all_preds, axis=0)
    ground_truth = np.concatenate(all_targets, axis=0)
    
    avg_latency_ms = np.mean(inference_times) * 1000.0

    return predictions, ground_truth, avg_latency_ms
