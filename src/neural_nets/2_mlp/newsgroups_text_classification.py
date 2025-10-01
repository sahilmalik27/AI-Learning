"""
20 Newsgroups Text Classification CLI (nn_core)

Text classification using TF-IDF features + 2-layer MLP.
Demonstrates preprocessing differences vs. vision tasks.
"""

import sys
sys.path.append('src')

import argparse
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from nn_core.models.mlp import MLP
from nn_core.optim.optim import Optim
from nn_core.schedulers.lr import LRScheduler
from nn_core.training.loop import train_supervised
from nn_core.utils.cli import add_common_training_args, build_optim_and_scheduler


def load_20newsgroups(max_features=20000):
    data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vectorizer.fit_transform(data.data)
    X = X.astype(np.float32).toarray()
    y = np.array(data.target, dtype=np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():
    parser = argparse.ArgumentParser(description='20 Newsgroups Text Classification (nn_core)')
    parser = add_common_training_args(parser)
    parser.set_defaults(lr=3e-3, opt='adamw', clip_grad=1.0, lr_sched='cosine', warmup_epochs=1, weight_decay=1e-2, init='xavier')
    parser.add_argument('--max-features', type=int, default=20000)

    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_20newsgroups(max_features=args.max_features)

    d = X_train.shape[1]
    c = len(np.unique(y_train))
    model = MLP(d=d, h=256, c=c, init=args.init, seed=args.seed)
    params = model.parameters()

    optim, sched = build_optim_and_scheduler(params, args, Optim, LRScheduler)

    def on_epoch_end(e, lr_e, tr_l, v_l, v_a):
        print(f"Epoch {e:02d} | lr={lr_e:.5f} | train_loss={tr_l:.4f} | val_loss={v_l:.4f} | val_acc={v_a:.4f}")

    train_losses, val_losses, val_accs = train_supervised(
        model,
        X_train, y_train,
        X_test, y_test,
        num_epochs=args.epochs, batch_size=args.batch_size,
        num_classes=c,
        optim=optim, scheduler=sched,
        seed=args.seed, clip_grad=args.clip_grad,
        on_epoch_end=on_epoch_end
    )

    print(f"Final val accuracy: {val_accs[-1]:.4f}")


if __name__ == '__main__':
    main()
