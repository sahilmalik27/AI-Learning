"""Time series forecasting experiment runner.

Features:
- Model factory (RNN/LSTM/GRU) for time series prediction
- Device pick (CUDA→MPS→CPU)
- Config-driven training with synthetic, CSV, or Yahoo Finance data
- TensorBoard logging with prediction overlays
- Checkpointing to models/representations/exp_ts/<exp_name>
"""

import argparse
import os
import yaml
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from .data_timeseries import build_datasets
from .models_ts import build_ts_model
from ..exp.utils import seed_everything, count_parameters
import pandas as pd


def train_one_epoch(model, loader, opt, loss_fn, device, clip_grad=None):
    """Train one epoch for time series forecasting."""
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()
        total += float(loss.item()) * y.size(0)
        n += y.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Evaluate time series model."""
    model.eval()
    total, n = 0.0, 0
    y_true_list, y_pred_list = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        total += float(loss.item()) * y.size(0)
        n += y.size(0)
        y_true_list.append(y.cpu())
        y_pred_list.append(pred.cpu())
    
    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)
    return total / n, y_true, y_pred


def run_experiment(cfg):
    """Main time series experiment runner."""
    seed_everything(cfg.get('seed', 1337))
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    # Data
    train_ds, val_ds = build_datasets(cfg['data'])
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['train']['batch_size'], shuffle=False)

    # Model
    model = build_ts_model(cfg['model'])
    model.to(device)
    print(model)
    print(f"Parameters: {count_parameters(model):,}")

    # Loss/optim
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])

    # Logging
    run_name = cfg.get('run_name', f"{cfg['model']['type']}_ts")
    writer = SummaryWriter(log_dir=os.path.join('runs_ts', run_name))

    # Create output directories
    os.makedirs('models/representations/exp_ts', exist_ok=True)
    os.makedirs('predictions_ts', exist_ok=True)

    best_loss = float('inf')
    for epoch in range(1, cfg['train']['epochs']+1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device, cfg['train'].get('clip_grad'))
        va_loss, y_true, y_pred = evaluate(model, val_loader, loss_fn, device)

        writer.add_scalar('loss/train', tr_loss, epoch)
        writer.add_scalar('loss/val', va_loss, epoch)

        print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")
        
        if va_loss < best_loss:
            best_loss = va_loss
            # Create directory if it doesn't exist
            os.makedirs(f"models/representations/exp_ts/{run_name}", exist_ok=True)
            torch.save({
                'model_state': model.state_dict(), 
                'cfg': cfg,
                'best_loss': best_loss
            }, f"models/representations/exp_ts/{run_name}/best.pt")

    # Save predictions
    df = pd.DataFrame({
        'y_true': y_true.numpy().reshape(-1), 
        'y_pred': y_pred.numpy().reshape(-1)
    })
    df.to_csv(f'predictions_ts/{run_name}_val_preds.csv', index=False)
    
    writer.close()
    print(f"Best validation loss: {best_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train time series models")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    run_experiment(cfg)


if __name__ == '__main__':
    main()
