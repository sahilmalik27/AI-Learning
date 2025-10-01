import os
import numpy as np
from sklearn.datasets import fetch_openml


def download_mnist_samples(num_samples: int = 10, save_dir: str = 'data/raw/mnist_samples'):
    os.makedirs(save_dir, exist_ok=True)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0
    y = mnist["target"].astype(np.int64)
    used = set()
    count = 0
    for i in range(len(X)):
        label = int(y[i])
        if label in used:
            continue
        np.save(os.path.join(save_dir, f'sample_{count:02d}_digit_{label}.npy'), X[i])
        used.add(label)
        count += 1
        if count >= num_samples:
            break
    return count


def download_fashion_mnist_samples(num_samples: int = 10, save_dir: str = 'data/raw/fashion_mnist_samples'):
    os.makedirs(save_dir, exist_ok=True)
    data = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X = data["data"].astype(np.float32) / 255.0
    y = data["target"].astype(np.int64)
    used = set()
    count = 0
    for i in range(len(X)):
        label = int(y[i])
        if label in used:
            continue
        np.save(os.path.join(save_dir, f'sample_{count:02d}_class_{label}.npy'), X[i])
        used.add(label)
        count += 1
        if count >= num_samples:
            break
    return count


# ---------- label name helpers ----------
def mnist_label_names():
    return [str(i) for i in range(10)]


def fashion_mnist_label_names():
    # Standard Fashion-MNIST class names in label order 0..9
    return [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]


def cifar10_label_names():
    # Common CIFAR-10 class names
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
