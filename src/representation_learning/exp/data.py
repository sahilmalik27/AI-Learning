# exp/data.py
"""Data utilities for CIFAR-10 loaders with optional subset sizing.

Detects pre-downloaded data under ./data and avoids re-downloads. Provides
subset sampling via max_train/max_test for quick experiments.
"""

import os
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

def _cifar10_already_present(root: str) -> bool:
    data_dir = os.path.join(root, 'cifar-10-batches-py')
    expected = [
        'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5',
        'test_batch', 'batches.meta'
    ]
    if not os.path.isdir(data_dir):
        return False
    return all(os.path.exists(os.path.join(data_dir, f)) for f in expected)

def make_cifar10_loaders(root, batch_size, num_workers=2, max_train: int | None = None, max_test: int | None = None):
    # Force root under ./data by default semantics of config; if absolute path provided, honor it
    root = os.path.abspath(root)
    train_tf = T.Compose([T.RandomCrop(32,padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    test_tf  = T.Compose([T.ToTensor()])
    download = not _cifar10_already_present(root)
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,  download=download, transform=train_tf)
    testset  = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=test_tf)
    if max_train:
        trainset = Subset(trainset, list(range(min(max_train, len(trainset)))))
    if max_test:
        testset = Subset(testset, list(range(min(max_test, len(testset)))))
    return (DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True),
            DataLoader(testset, batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True))
