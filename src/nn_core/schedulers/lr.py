"""
Learning Rate Scheduler Module

This module provides various learning rate scheduling strategies:
- None: Constant learning rate
- Step: Step decay at regular intervals
- Cosine: Cosine annealing with optional warmup

All schedulers support warmup periods for stable training initialization.

Author: AI Learning Project
"""

import math

class LRScheduler:
    """
    Learning rate scheduler with multiple decay strategies.
    
    Args:
        base_lr (float): Base learning rate
        kind (str): Scheduler type ('none', 'step', 'cosine')
        step (int): Step size for step decay (epochs)
        gamma (float): Decay factor for step scheduler
        total_epochs (int): Total training epochs
        warmup (int): Warmup epochs (cosine scheduler)
    """
    def __init__(self, base_lr: float, kind: str = 'none', step: int = 5,
                 gamma: float = 0.5, total_epochs: int = 10, warmup: int = 0):
        self.base_lr = base_lr
        self.kind = kind
        self.step = step
        self.gamma = gamma
        self.total_epochs = total_epochs
        self.warmup = warmup

    def lr_at(self, epoch: int) -> float:
        """
        Get learning rate for given epoch.
        
        Args:
            epoch (int): Current epoch (1-indexed)
            
        Returns:
            Learning rate for this epoch
        """
        if self.kind == 'none':
            return self.base_lr
        if self.kind == 'step':
            drops = (epoch - 1) // self.step
            return self.base_lr * (self.gamma ** drops)
        if self.kind == 'cosine':
            e = epoch - 1
            if self.warmup > 0 and e < self.warmup:
                return self.base_lr * (e + 1) / self.warmup
            T = max(1, self.total_epochs - self.warmup)
            t = min(T, max(0, e - self.warmup))
            return 0.5 * self.base_lr * (1 + math.cos(math.pi * t / T))
        return self.base_lr
