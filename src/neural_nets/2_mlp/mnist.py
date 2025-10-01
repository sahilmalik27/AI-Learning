"""
MNIST MLP CLI (nn_core)

Train a 2-layer MLP on MNIST using the reusable nn_core modules.
Features:
- Optimizers: sgd/momentum/nesterov/adam/adamw
- Init: he/xavier/normal
- LR schedulers: none/step/cosine (+ warmup)
- Gradient clipping

Usage:
  python src/neural_nets/2_mlp/mnist_cli.py --epochs 3 --opt adamw --lr 3e-3 \
      --lr-sched cosine --warmup-epochs 1 --weight-decay 1e-2
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
from nn_core.training.loop import train_supervised, one_hot, accuracy
from nn_core.utils.cli import add_common_training_args, build_optim_and_scheduler


def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0
    y = mnist["target"].astype(np.int64)
    return train_test_split(X, y, test_size=10000, random_state=42, stratify=y)


def main():
    parser = argparse.ArgumentParser(description='MNIST MLP (nn_core)')
    parser = add_common_training_args(parser)

    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_mnist()

    model = MLP(d=784, h=128, c=10, init=args.init, seed=args.seed)
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
