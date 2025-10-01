"""
CIFAR-10 (Flattened) MLP CLI (nn_core)

Demonstrates MLP limitations on CIFAR-10 (images require convolutions for best results).
This flattened baseline is still useful to test optimizers and schedulers quickly.
"""

import sys
sys.path.append('src')

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from nn_core.models.mlp import MLP
from nn_core.optim.optim import Optim
from nn_core.schedulers.lr import LRScheduler
from nn_core.training.loop import train_supervised
from nn_core.utils.cli import add_common_training_args, build_optim_and_scheduler


def load_cifar10_flattened():
    # Using OpenML CIFAR-10 (small) surrogate; otherwise user can plug a loader
    data = fetch_openml("CIFAR_10_small", version=1, as_frame=False)
    X = data["data"].astype(np.float32) / 255.0
    y = data["target"].astype(np.int64)
    return train_test_split(X, y, test_size=10000, random_state=42, stratify=y)


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 (flattened) MLP (nn_core)')
    parser = add_common_training_args(parser)
    parser.set_defaults(epochs=15, lr=0.001, opt='adamw', clip_grad=1.0, lr_sched='cosine', warmup_epochs=2, weight_decay=1e-2)

    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_cifar10_flattened()

    d = X_train.shape[1]  # expected 3072 if 32x32x3 flattened
    model = MLP(d=d, h=512, c=10, init=args.init, seed=args.seed)
    params = model.parameters()

    optim, sched = build_optim_and_scheduler(params, args, Optim, LRScheduler)

    def on_epoch_end(e, lr_e, tr_l, v_l, v_a):
        print(f"Epoch {e:02d} | lr={lr_e:.5f} | train_loss={tr_l:.4f} | test_loss={v_l:.4f} | test_acc={v_a:.4f}")

    train_losses, test_losses, test_accs = train_supervised(
        model,
        X_train, y_train,
        X_test, y_test,
        num_epochs=args.epochs, batch_size=args.batch_size,
        num_classes=10,
        optim=optim, scheduler=sched,
        seed=args.seed, clip_grad=args.clip_grad,
        on_epoch_end=on_epoch_end
    )

    print(f"Final test accuracy: {test_accs[-1]:.4f} (expect lower; MLP is limited for CIFAR-10)")


if __name__ == '__main__':
    main()
