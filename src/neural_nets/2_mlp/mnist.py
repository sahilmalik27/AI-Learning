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
from nn_core.utils.cli import add_common_training_args, build_optim_and_scheduler, add_image_prediction_args
from nn_core.utils.io import save_pickle, load_pickle, ensure_dir


def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0
    y = mnist["target"].astype(np.int64)
    return train_test_split(X, y, test_size=10000, random_state=42, stratify=y)


def main():
    parser = argparse.ArgumentParser(description='MNIST MLP (nn_core)')
    parser = add_common_training_args(parser)

    parser = add_image_prediction_args(parser, model_name='mnist_mlp', include_list_samples=True, include_download=True)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_mnist()

    # Prediction-only / samples management
    if args.download_samples:
        from nn_core.utils.datasets import download_mnist_samples
        n = download_mnist_samples(10, 'data/raw/mnist_samples')
        print(f"Downloaded {n} MNIST samples to data/raw/mnist_samples")
        return
    if args.predict_image or args.list_samples:
        import glob, os
        samples_dir = 'data/raw/mnist_samples'
        if args.list_samples:
            files = sorted(glob.glob(os.path.join(samples_dir, '*.npy')))
            if not files:
                print('No samples found. Run download_mnist_samples.py first.')
                return
            for i, f in enumerate(files):
                print(f"{i:02d}: {os.path.basename(f)}")
            return
        # load model and predict
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}. Train first.")
            return
        model = load_pickle(args.model_path)
        import numpy as np
        x = np.load(args.predict_image).astype(np.float32)
        p = model.forward(x.reshape(1, -1))
        pred = int(np.argmax(p[0]))
        from nn_core.utils.datasets import mnist_label_names
        label = mnist_label_names()[pred]
        conf = float(p[0][pred])
        print(f"Predicted: {pred} ({label}) | Confidence: {conf:.4f}")
        return

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
    # Save model
    ensure_dir('models')
    save_pickle(model, args.model_path)
    print(f"Saved model to {args.model_path}")


if __name__ == '__main__':
    main()
