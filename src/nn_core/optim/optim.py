"""
Optimizer Module

This module provides various optimization algorithms for neural network training:
- SGD: Stochastic Gradient Descent
- Momentum: SGD with momentum
- Nesterov: Nesterov Accelerated Gradient
- Adam: Adaptive Moment Estimation
- AdamW: Adam with decoupled weight decay

All optimizers support L2 weight decay and maintain internal state for momentum/adaptive methods.

Author: AI Learning Project
"""

import numpy as np

class Optim:
    """
    Unified optimizer supporting multiple algorithms.
    
    Supported optimizers:
    - 'sgd': Vanilla SGD
    - 'momentum': SGD with momentum
    - 'nesterov': Nesterov Accelerated Gradient
    - 'adam': Adaptive Moment Estimation
    - 'adamw': Adam with decoupled weight decay
    
    Args:
        params (dict): Model parameters dictionary
        opt (str): Optimizer type
        lr (float): Learning rate
        beta1 (float): First moment decay (momentum/Adam)
        beta2 (float): Second moment decay (Adam)
        eps (float): Numerical stability term (Adam)
        weight_decay (float): L2 regularization strength
        nesterov (bool): Use Nesterov momentum (for 'nesterov' optimizer)
    """
    def __init__(self, params: dict, opt: str = 'sgd', lr: float = 0.1,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.0, nesterov: bool = False):
        self.opt = opt
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.v = {k: np.zeros_like(p) for k, p in params.items()}
        self.s = {k: np.zeros_like(p) for k, p in params.items()}
        self.t = 0
        self.params = params

    def step(self, grads: dict):
        """
        Perform one optimization step.
        
        Args:
            grads (dict): Gradients dictionary with same keys as params
        """
        if self.opt == 'adamw':
            for k in self.params:
                if k.startswith('W'):
                    self.params[k] -= self.lr * self.weight_decay * self.params[k]
        self.t += 1
        for k in self.params:
            g = grads[k]
            if self.opt in ['sgd', 'momentum', 'nesterov', 'adam'] and self.weight_decay > 0 and k.startswith('W'):
                g = g + self.weight_decay * self.params[k]
            if self.opt == 'sgd':
                self.params[k] -= self.lr * g
            elif self.opt in ['momentum', 'nesterov']:
                v = self.v[k] = self.beta1 * self.v[k] + (1 - self.beta1) * g
                update = self.beta1 * v + (1 - self.beta1) * g if self.opt == 'nesterov' else v
                self.params[k] -= self.lr * update
            elif self.opt in ['adam', 'adamw']:
                m = self.v[k] = self.beta1 * self.v[k] + (1 - self.beta1) * g
                v = self.s[k] = self.beta2 * self.s[k] + (1 - self.beta2) * (g * g)
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                self.params[k] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))
