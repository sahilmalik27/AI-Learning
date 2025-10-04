"""Training engine for sequence models: optimizer/scheduler factories and train/eval loops.

Lightweight utilities for RNN/LSTM/GRU training on AG_NEWS. Designed to be device-agnostic
and work on CUDA, MPS and CPU. Includes gate introspection for LSTM/GRU.
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import pandas as pd

from .data_text import build_datasets
from .models.sequence_models import build_model, gate_probe_batch
from .utils import seed_everything, count_parameters


def train_one_epoch(model, loader, optimizer, criterion, device, clip_grad=None):
    """Train one epoch with gradient clipping."""
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for batch in loader:
        x, lengths, y = [t.to(device) for t in batch]
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)  # (B, C)
        loss = criterion(logits, y)
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            total_correct += (pred == y).sum().item()
            total_loss += float(loss.item()) * y.size(0)
            total_count += y.size(0)

    return total_loss/total_count, total_correct/total_count


def evaluate(model, loader, criterion, device):
    """Evaluate model and return predictions."""
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            x, lengths, y = [t.to(device) for t in batch]
            logits = model(x, lengths)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=-1)

            total_correct += (pred == y).sum().item()
            total_loss += float(loss.item()) * y.size(0)
            total_count += y.size(0)
            all_pred.append(pred.cpu())
            all_true.append(y.cpu())

    preds = torch.cat(all_pred).numpy()
    trues = torch.cat(all_true).numpy()
    return total_loss/total_count, total_correct/total_count, preds, trues


def log_gate_stats(model, batch_probe, writer, epoch, device):
    """Log gate statistics for LSTM/GRU introspection."""
    if batch_probe is None:
        return
    x, lengths, _ = [t.to(device) for t in batch_probe]
    stats = gate_probe_batch(model, x, lengths)
    # stats: dict of scalar means
    for k, v in stats.items():
        writer.add_scalar(k, v, epoch)


def run_experiment(cfg):
    """Main experiment runner."""
    seed_everything(cfg.get("seed", 123))

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    # Data
    train_ds, val_ds, vocab = build_datasets(cfg["data"])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              collate_fn=train_ds.collate_fn, num_workers=cfg["data"].get("num_workers",0))
    val_loader   = DataLoader(val_ds,   batch_size=cfg["train"]["batch_size"], shuffle=False,
                              collate_fn=val_ds.collate_fn, num_workers=cfg["data"].get("num_workers",0))

    # Model
    model = build_model(cfg["model"], vocab_size=len(vocab))
    model.to(device)
    print(model)
    print(f"Parameters: {count_parameters(model):,}")

    # Loss/optim
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"]) 

    # Logging
    run_name = cfg.get("run_name", f"{cfg['model']['type']}_ag_news")
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    best_acc = 0.0
    probe_batch = next(iter(val_loader)) if cfg.get("introspect", True) else None

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                          clip_grad=cfg["train"].get("clip_grad", None))
        va_loss, va_acc, preds, trues = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("acc/train", tr_acc, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("acc/val", va_acc, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("stats/epoch_time_sec", dt, epoch)

        # Introspection (gates, norms)
        log_gate_stats(model, probe_batch, writer, epoch, device)

        print(f"Epoch {epoch:03d} | train {tr_loss:.3f}/{tr_acc:.3f} | val {va_loss:.3f}/{va_acc:.3f} | {dt:.1f}s")

        if va_acc > best_acc:
            best_acc = va_acc
            # Create directory if it doesn't exist
            os.makedirs(f"models/representations/exp/{run_name}", exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'vocab': vocab.get_itos(),
                'cfg': cfg,
            }, f"models/representations/exp/{run_name}/best.pt")

    # Save predictions CSV
    df = pd.DataFrame({"y_true": trues, "y_pred": preds})
    df.to_csv(f"predictions/{run_name}_val_preds.csv", index=False)

    # Confusion matrix (printed only; view in notebook as needed)
    cm = confusion_matrix(trues, preds)
    print("Confusion Matrix:\n", cm)

    writer.close()
    print(f"Best validation accuracy: {best_acc:.3f}")
