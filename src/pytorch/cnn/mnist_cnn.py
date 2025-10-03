# mnist_cnn.py
# PyTorch CNN for MNIST with CLI: train / predict, plots, checkpoints
# Safe for macOS (spawn) and Jupyter-less environments.

import argparse
import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Use non-interactive backend by default (safer for scripts); can still show with --show-plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ----------------------------
# Utility: device pick + seed
# ----------------------------
class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    def __init__(self, patience=3, min_delta=0.0):
        """
        Args:
            patience: number of epochs to wait without improvement
            min_delta: minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        """Check if training should stop based on validation loss."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def pick_device(device_arg: str) -> str:
    if device_arg and device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ----------------------------
# Model
# ----------------------------
class CNN(nn.Module):
    """CNN for MNIST classification."""
    def __init__(self, dropout=0.2):
        super().__init__()
        # First conv block: in_channels=1 (grayscale), out_channels=32 filters, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # output: [32,28,28]
        self.pool1 = nn.MaxPool2d(2, 2)                          # output: [32,14,14]

        # Second conv block: input 32 → 64 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # output: [64,14,14]
        self.pool2 = nn.MaxPool2d(2, 2)                          # output: [64,7,7]

        # Fully connected layers
        self.fc1 = nn.Linear(64*7*7, 128)  # Flatten [64,7,7] → 3136
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 10)      # 10 logits

    def forward(self, x):
        # Convolutional feature extraction
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)   # [B, 64*7*7]

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)             # logits, no softmax here
        return x


# ----------------------------
# Data
# ----------------------------
def build_dataloaders(data_dir: str,
                      batch_train: int,
                      batch_test: int,
                      num_workers: int,
                      device: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    test_ds  = datasets.MNIST(data_dir, train=False, transform=transform, download=True)

    pin = (device != "cpu")
    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_test, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


# ----------------------------
# Train/Eval loops
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits  = model(xb)
        loss    = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        b = yb.size(0)
        total_loss   += loss.item() * b
        total_correct += (logits.argmax(1) == yb).sum().item()
        n += b

    return total_loss / n, total_correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss   = criterion(logits, yb)

        b = yb.size(0)
        total_loss   += loss.item() * b
        total_correct += (logits.argmax(1) == yb).sum().item()
        n += b

    return total_loss / n, total_correct / n


# ----------------------------
# Plotting
# ----------------------------
def plot_curves(train_loss, test_loss, train_acc, test_acc, out_path="training_curves.png", show=False):
    epochs = range(1, len(train_loss) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    ax[0].plot(epochs, test_loss, marker="s", label="Test Loss")
    ax[0].set_title("Loss over Epochs"); ax[0].set_xlabel("Epochs"); ax[0].set_ylabel("Loss")
    ax[0].grid(True, linestyle="--", alpha=0.6); ax[0].legend()

    # Accuracy
    ax[1].plot(epochs, train_acc, marker="o", label="Train Acc")
    ax[1].plot(epochs, test_acc, marker="s", label="Test Acc")
    ax[1].set_title("Accuracy over Epochs"); ax[1].set_xlabel("Epochs"); ax[1].set_ylabel("Accuracy")
    ax[1].grid(True, linestyle="--", alpha=0.6); ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        try:
            plt.show()
        except Exception:
            pass  # Ignore non-interactive backend warnings
    plt.close(fig)
    print(f"[saved] {out_path}")


@torch.no_grad()
def plot_confusion(model, loader, device, out_path="confusion_matrix.png", show=False):
    if not SKLEARN_OK:
        print("[warn] scikit-learn not available; skipping confusion matrix.")
        return
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("MNIST CNN Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        try:
            plt.show()
        except Exception:
            pass  # Ignore non-interactive backend warnings
    plt.close(fig)
    print(f"[saved] {out_path}")


# ----------------------------
# Predict
# ----------------------------
@torch.no_grad()
def predict_samples(model, loader, device, k=10, out_path="pred_samples.png", show=False):
    model.eval()
    xb, yb = next(iter(loader))
    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb[:k])
    preds = logits.argmax(1).cpu().numpy()
    truths = yb[:k].cpu().numpy()

    # Plot
    cols = k
    fig, axes = plt.subplots(1, cols, figsize=(1.8 * cols, 2))
    if cols == 1:
        axes = [axes]
    for i in range(cols):
        axes[i].imshow(xb[i].cpu().squeeze(), cmap="gray")
        axes[i].set_title(f"T:{truths[i]} P:{preds[i]}")
        axes[i].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        try:
            plt.show()
        except Exception:
            pass  # Ignore non-interactive backend warnings
    plt.close(fig)
    print(f"[saved] {out_path}")


@torch.no_grad()
def predict_image(model, img_path, device, out_path="pred_single.png", show=False):
    from PIL import Image

    # MNIST is 28x28 grayscale, normalized with given mean/std
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img = Image.open(img_path)
    x = transform(img).unsqueeze(0).to(device)  # [1,1,28,28]
    logits = model(x)
    pred = logits.argmax(1).item()

    # Visualize
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(x.cpu().squeeze().numpy(), cmap="gray")
    ax.axis("off")
    ax.set_title(f"Pred: {pred}")
    fig.savefig(out_path, dpi=150)
    if show:
        try:
            plt.show()
        except Exception:
            pass  # Ignore non-interactive backend warnings
    plt.close(fig)
    print(f"[pred] {img_path} -> {pred}  [saved plot: {out_path}]")
    return pred


# ----------------------------
# Checkpoint
# ----------------------------
def save_ckpt(model, path):
    torch.save({"model_state": model.state_dict()}, path)
    print(f"[saved] checkpoint -> {path}")


def load_ckpt(model_cls, path, device, dropout=0.2):
    state = torch.load(path, map_location=device, weights_only=True)
    model = model_cls(dropout=dropout).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


# ----------------------------
# Subcommands
# ----------------------------
def cmd_train(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"[info] device={device}")

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_train=args.batch_train,
        batch_test=args.batch_test,
        num_workers=args.num_workers,
        device=device,
    )

    model = CNN(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Add learning rate scheduler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, 
                                 patience=args.lr_patience, verbose=True)
    
    # Add early stopping
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(tr_loss); test_losses.append(te_loss)
        train_accs.append(tr_acc);    test_accs.append(te_acc)

        # Step the learning rate scheduler
        scheduler.step(te_loss)
        
        # Get current learning rate for logging
        current_lr = optimizer.param_groups[0]['lr']

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f} | "
              f"lr {current_lr:.2e} | {dt:.1f}s")

        # Early stopping check
        if early_stopper.step(te_loss):
            print(f"[early stopping] no improvement for {args.patience} epochs")
            break

    # Save checkpoint
    if args.checkpoint:
        save_ckpt(model, args.checkpoint)

    # Plots
    plot_curves(train_losses, test_losses, train_accs, test_accs,
                out_path=args.curves_path, show=args.show_plots)

    # Confusion matrix
    if args.confusion_matrix:
        plot_confusion(model, test_loader, device,
                       out_path=args.cm_path, show=args.show_plots)

    # Predict samples grid
    if args.predict_grid:
        predict_samples(model, test_loader, device, k=args.k_samples,
                        out_path=args.samples_path, show=args.show_plots)


def cmd_predict(args):
    device = pick_device(args.device)
    print(f"[info] device={device}")
    # Load model
    model = load_ckpt(CNN, args.checkpoint, device, dropout=args.dropout)

    if args.image:
        predict_image(model, args.image, device,
                      out_path=args.out_image, show=args.show_plots)
    else:
        # Predict a small grid from the test set
        _, test_loader = build_dataloaders(
            data_dir=args.data_dir, batch_train=256, batch_test=1024,
            num_workers=0, device=device
        )
        predict_samples(model, test_loader, device, k=args.k_samples,
                        out_path=args.samples_path, show=args.show_plots)


# ----------------------------
# CLI
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="PyTorch CNN on MNIST (train/predict with plots)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train
    t = sub.add_parser("train", help="Train a CNN on MNIST")
    t.add_argument("--data-dir", default="./data")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-train", type=int, default=256)
    t.add_argument("--batch-test", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--dropout", type=float, default=0.20)
    t.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    t.add_argument("--num-workers", type=int, default=0)   # macOS-safe default
    t.add_argument("--seed", type=int, default=42)

    t.add_argument("--checkpoint", default="mnist_cnn.pt")
    t.add_argument("--curves-path", default="training_curves.png")
    t.add_argument("--cm-path", default="confusion_matrix.png")
    t.add_argument("--samples-path", default="pred_samples.png")
    t.add_argument("--k-samples", type=int, default=10)

    t.add_argument("--confusion-matrix", action="store_true")
    t.add_argument("--predict-grid", action="store_true")
    t.add_argument("--show-plots", action="store_true")
    
    # Early stopping and LR scheduling
    t.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    t.add_argument("--min-delta", type=float, default=1e-4, help="Minimum change for early stopping")
    t.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor for scheduler")
    t.add_argument("--lr-patience", type=int, default=2, help="LR scheduler patience (epochs)")

    # Predict
    p_ = sub.add_parser("predict", help="Predict using a saved checkpoint")
    p_.add_argument("--checkpoint", required=True)
    p_.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p_.add_argument("--dropout", type=float, default=0.20)  # must match training architecture
    p_.add_argument("--data-dir", default="./data")
    p_.add_argument("--k-samples", type=int, default=10)
    p_.add_argument("--image", help="Path to a single image to classify")
    p_.add_argument("--out-image", default="pred_single.png")
    p_.add_argument("--samples-path", default="pred_samples.png")
    p_.add_argument("--show-plots", action="store_true")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    # macOS / Python 3.8+ prefers 'spawn'; be explicit to avoid surprises
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
