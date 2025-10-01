"""
Training Loop Module

This module provides the core training infrastructure:
- Supervised training loop with batching and evaluation
- Utility functions for one-hot encoding, loss computation, accuracy
- Gradient clipping for training stability
- Callback system for monitoring training progress

Author: AI Learning Project
"""

import numpy as np
from typing import Callable, Tuple, Dict


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding.
    
    Args:
        y: Integer labels (N,)
        num_classes: Number of classes
        
    Returns:
        One-hot encoded labels (N, num_classes)
    """
    oh = np.zeros((y.size, num_classes), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh


def cross_entropy(p: np.ndarray, y_onehot: np.ndarray) -> float:
    """
    Compute cross-entropy loss with numerical stability.
    
    Args:
        p: Predicted probabilities (N, C)
        y_onehot: True one-hot labels (N, C)
        
    Returns:
        Cross-entropy loss (scalar)
    """
    eps = 1e-12
    return float(-np.mean(np.sum(y_onehot * np.log(p + eps), axis=1)))


def accuracy(p: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        p: Predicted probabilities (N, C)
        y_true: True integer labels (N,)
        
    Returns:
        Accuracy (0-1)
    """
    return float((np.argmax(p, axis=1) == y_true).mean())


def clip_gradients_(grads: Dict[str, np.ndarray], max_norm: float) -> None:
    """
    Clip gradients by global norm for training stability.
    
    Args:
        grads: Gradients dictionary
        max_norm: Maximum gradient norm (0 = no clipping)
    """
    if max_norm <= 0:
        return
    total_sq = 0.0
    for k in grads:
        g = grads[k].ravel()
        total_sq += float(np.dot(g, g))
    norm = np.sqrt(total_sq) + 1e-12
    if norm > max_norm:
        scale = max_norm / norm
        for k in grads:
            grads[k] *= scale


def train_supervised(model,
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     num_epochs: int, batch_size: int,
                     num_classes: int,
                     optim,
                     scheduler,
                     seed: int = 42,
                     clip_grad: float = 0.0,
                     on_epoch_end: Callable[[int, float, float, float, float], None] = None
                     ) -> Tuple[list, list, list]:
    """
    Train a model using supervised learning with mini-batch SGD.
    
    Args:
        model: Model with forward() and backward() methods
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        num_epochs: Number of training epochs
        batch_size: Mini-batch size
        num_classes: Number of output classes
        optim: Optimizer instance
        scheduler: Learning rate scheduler (can be None)
        seed: Random seed for reproducibility
        clip_grad: Gradient clipping threshold (0 = no clipping)
        on_epoch_end: Callback function(epoch, lr, train_loss, val_loss, val_acc)
        
    Returns:
        Tuple of (train_losses, val_losses, val_accuracies) lists
    """
    rng = np.random.default_rng(seed)
    N = X_train.shape[0]
    steps = (N + batch_size - 1) // batch_size

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, num_epochs + 1):
        lr_epoch = scheduler.lr_at(epoch) if scheduler is not None else optim.lr
        optim.lr = lr_epoch

        idx = rng.permutation(N)
        Xs, ys = X_train[idx], y_train[idx]

        total_loss = 0.0
        for s in range(steps):
            a, b = s * batch_size, min(N, (s + 1) * batch_size)
            xb, yb = Xs[a:b], ys[a:b]
            yb_oh = one_hot(yb, num_classes)
            pb = model.forward(xb)
            loss = cross_entropy(pb, yb_oh)
            total_loss += loss * (b - a)
            grads = model.backward(yb_oh)
            clip_gradients_(grads, clip_grad)
            optim.step(grads)

        p_val = model.forward(X_val)
        v_acc = accuracy(p_val, y_val)
        v_loss = cross_entropy(p_val, one_hot(y_val, num_classes))

        tr_loss = total_loss / N
        train_losses.append(tr_loss)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        if on_epoch_end is not None:
            on_epoch_end(epoch, lr_epoch, tr_loss, v_loss, v_acc)

    return train_losses, val_losses, val_accs
