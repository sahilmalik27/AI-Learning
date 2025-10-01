"""
Fashion-MNIST MLP CLI (nn_core)

Train a 2-layer MLP on Fashion-MNIST to handle harder patterns than MNIST while remaining fast.
Includes optimizers, schedulers with warmup, and gradient clipping.
"""

import sys
sys.path.append('src')

import argparse
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from nn_core.models.mlp import MLP
from nn_core.optim.optim import Optim
from nn_core.schedulers.lr import LRScheduler
from nn_core.training.loop import train_supervised
from nn_core.utils.cli import add_common_training_args, build_optim_and_scheduler


def load_fashion_mnist():
    data = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X = data["data"].astype(np.float32) / 255.0
    y = data["target"].astype(np.int64)
    return train_test_split(X, y, test_size=10000, random_state=42, stratify=y)


def main():
    parser = argparse.ArgumentParser(description='Fashion-MNIST MLP (nn_core)')
    parser = add_common_training_args(parser)
    parser.set_defaults(lr=0.05, opt='adamw', clip_grad=1.0, lr_sched='cosine', warmup_epochs=1)

    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_fashion_mnist()

    model = MLP(d=784, h=256, c=10, init=args.init, seed=args.seed)
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

    print(f"Final test accuracy: {test_accs[-1]:.4f}")


if __name__ == '__main__':
    main()
