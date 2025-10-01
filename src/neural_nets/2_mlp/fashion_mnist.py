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
from nn_core.utils.cli import add_common_training_args, build_optim_and_scheduler, add_image_prediction_args
from nn_core.utils.io import save_pickle, load_pickle, ensure_dir


def load_fashion_mnist():
    data = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X = data["data"].astype(np.float32) / 255.0
    y = data["target"].astype(np.int64)
    return train_test_split(X, y, test_size=10000, random_state=42, stratify=y)


def main():
    parser = argparse.ArgumentParser(description='Fashion-MNIST MLP (nn_core)')
    parser = add_common_training_args(parser)
    parser.set_defaults(lr=0.05, opt='adamw', clip_grad=1.0, lr_sched='cosine', warmup_epochs=1)

    parser = add_image_prediction_args(parser, model_name='fashion_mnist_mlp')
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_fashion_mnist()

    # Prediction-only / download samples
    if getattr(args, 'download_samples', False):
        from nn_core.utils.datasets import download_fashion_mnist_samples
        n = download_fashion_mnist_samples(10, 'data/raw/fashion_mnist_samples')
        print(f"Downloaded {n} Fashion-MNIST samples to data/raw/fashion_mnist_samples")
        return
    # Prediction-only
    if args.predict_image:
        import os, numpy as np
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}. Train first.")
            return
        model = load_pickle(args.model_path)
        x = np.load(args.predict_image).astype(np.float32)
        p = model.forward(x.reshape(1, -1))
        pred = int(np.argmax(p[0]))
        from nn_core.utils.datasets import fashion_mnist_label_names
        label = fashion_mnist_label_names()[pred]
        conf = float(p[0][pred])
        print(f"Predicted: {pred} ({label}) | Confidence: {conf:.4f}")
        return

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
    ensure_dir('models')
    save_pickle(model, args.model_path)
    print(f"Saved model to {args.model_path}")


if __name__ == '__main__':
    main()
