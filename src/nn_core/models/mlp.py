"""
Multi-Layer Perceptron (MLP) Model

This module provides a pure NumPy implementation of a 2-layer MLP with:
- Configurable initialization schemes (He, Xavier, Normal)
- ReLU activation in hidden layer
- Softmax activation in output layer
- Forward and backward propagation methods
- Parameter access for optimizer integration

Author: AI Learning Project
"""

import numpy as np

class MLP:
    """
    Multi-Layer Perceptron with configurable initialization and activation functions.
    
    Architecture:
        Input (d) → Hidden (h, ReLU) → Output (c, Softmax)
    
    Args:
        d (int): Input dimension (default: 784 for 28x28 images)
        h (int): Hidden layer size (default: 128)
        c (int): Output classes (default: 10)
        init (str): Weight initialization scheme ('he', 'xavier', 'normal')
        seed (int): Random seed for reproducibility
        
    Attributes:
        W1, b1: First layer weights and biases
        W2, b2: Second layer weights and biases
        cache: Forward pass activations for backpropagation
    """
    def __init__(self, d: int = 784, h: int = 128, c: int = 10, init: str = 'he', seed: int = 42):
        rng = np.random.default_rng(seed)

        def init_w(shape, kind):
            fan_in, fan_out = shape[0], shape[1]
            if kind == 'he':
                std = np.sqrt(2.0 / fan_in)
                return rng.normal(0, std, size=shape).astype(np.float32)
            elif kind == 'xavier':
                std = np.sqrt(2.0 / (fan_in + fan_out))
                return rng.normal(0, std, size=shape).astype(np.float32)
            else:
                return rng.normal(0, 0.01, size=shape).astype(np.float32)

        self.W1 = init_w((d, h), init)
        self.b1 = np.zeros((h,), dtype=np.float32)
        self.W2 = init_w((h, c), init)
        self.b2 = np.zeros((c,), dtype=np.float32)
        self.cache = {}

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function: max(0, x)
        
        Args:
            x: Input array
            
        Returns:
            ReLU applied element-wise
        """
        return np.maximum(0, x)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """
        Softmax activation function with numerical stability.
        
        Args:
            logits: Raw logits from final layer
            
        Returns:
            Probability distribution over classes (sums to 1 per sample)
        """
        z = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the MLP.
        
        Args:
            X: Input batch (N, d)
            
        Returns:
            Output probabilities (N, c)
        """
        a1 = X @ self.W1 + self.b1
        h  = self.relu(a1)
        z  = h @ self.W2 + self.b2
        p  = self.softmax(z)
        self.cache = {"X": X, "a1": a1, "h": h, "z": z, "p": p}
        return p

    def backward(self, y_onehot: np.ndarray) -> dict:
        """
        Backward pass computing gradients via chain rule.
        
        Args:
            y_onehot: One-hot encoded true labels (N, c)
            
        Returns:
            Dictionary of gradients: {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        """
        X, a1, h, z, p = (self.cache[k] for k in ["X","a1","h","z","p"])
        N = X.shape[0]
        dz  = (p - y_onehot) / N
        dW2 = h.T @ dz
        db2 = dz.sum(axis=0)
        dh  = dz @ self.W2.T
        da1 = dh * (a1 > 0).astype(np.float32)
        dW1 = X.T @ da1
        db1 = da1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def parameters(self) -> dict:
        """
        Get model parameters for optimizer integration.
        
        Returns:
            Dictionary of parameter arrays: {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        """
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}
