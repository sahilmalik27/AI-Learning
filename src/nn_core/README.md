# NN Core - Reusable Neural Network Components

A modular, pure NumPy implementation of neural network components for educational and research purposes.

## 🏗️ Architecture

```
src/nn_core/
├── models/          # Neural network architectures
├── optim/           # Optimization algorithms  
├── schedulers/      # Learning rate scheduling
├── training/        # Training loops and utilities
└── utils/           # Common utilities
```

## 📦 Modules

### Models (`src/nn_core/models/`)

**MLP**: Multi-Layer Perceptron
- Configurable initialization (He, Xavier, Normal)
- ReLU + Softmax activations
- Forward/backward propagation
- Parameter access for optimizers

```python
from nn_core.models.mlp import MLP

model = MLP(d=784, h=128, c=10, init='he', seed=42)
```

### Optimizers (`src/nn_core/optim/`)

**Optim**: Unified optimizer supporting:
- SGD: Vanilla stochastic gradient descent
- Momentum: SGD with momentum
- Nesterov: Nesterov Accelerated Gradient
- Adam: Adaptive Moment Estimation
- AdamW: Adam with decoupled weight decay

```python
from nn_core.optim.optim import Optim

optim = Optim(params, opt='adamw', lr=0.001, weight_decay=0.01)
```

### Schedulers (`src/nn_core/schedulers/`)

**LRScheduler**: Learning rate scheduling
- None: Constant learning rate
- Step: Step decay at intervals
- Cosine: Cosine annealing with warmup

```python
from nn_core.schedulers.lr import LRScheduler

scheduler = LRScheduler(base_lr=0.1, kind='cosine', warmup=5)
```

### Training (`src/nn_core/training/`)

**train_supervised**: Complete training loop
- Mini-batch SGD with shuffling
- Gradient clipping
- Validation evaluation
- Callback system

```python
from nn_core.training.loop import train_supervised

train_losses, val_losses, val_accs = train_supervised(
    model, X_train, y_train, X_val, y_val,
    num_epochs=10, batch_size=128, num_classes=10,
    optim=optim, scheduler=scheduler, clip_grad=1.0
)
```

## 🚀 Quick Start

### Basic Usage

```python
import sys
sys.path.append('src')

from nn_core.models.mlp import MLP
from nn_core.optim.optim import Optim
from nn_core.schedulers.lr import LRScheduler
from nn_core.training.loop import train_supervised

# Create model
model = MLP(d=784, h=128, c=10, init='he')

# Setup optimizer and scheduler
params = model.parameters()
optim = Optim(params, opt='adamw', lr=0.001, weight_decay=0.01)
scheduler = LRScheduler(0.001, kind='cosine', warmup=5)

# Train
train_losses, val_losses, val_accs = train_supervised(
    model, X_train, y_train, X_val, y_val,
    num_epochs=10, batch_size=128, num_classes=10,
    optim=optim, scheduler=scheduler
)
```

### Advanced Configuration

```python
# Custom optimizer settings
optim = Optim(params, 
    opt='nesterov', 
    lr=0.1, 
    beta1=0.9, 
    weight_decay=1e-4
)

# Custom scheduler
scheduler = LRScheduler(
    base_lr=0.1,
    kind='step',
    step=5,
    gamma=0.5
)

# Training with gradient clipping
train_losses, val_losses, val_accs = train_supervised(
    model, X_train, y_train, X_val, y_val,
    num_epochs=20, batch_size=64, num_classes=10,
    optim=optim, scheduler=scheduler,
    clip_grad=1.0,  # Gradient clipping
    on_epoch_end=lambda e, lr, tr, val, acc: print(f"Epoch {e}: {acc:.3f}")
)
```

## 🎯 Use Cases

### 1. MNIST Classification
```python
# Already implemented in src/neural_nets/2_mlp/mlp_mnist_numpy.py
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --opt adamw --lr 0.003
```

### 2. Fashion-MNIST
```python
# Swap dataset loader, keep everything else
from sklearn.datasets import fetch_openml
fashion = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
X = fashion["data"].astype(np.float32) / 255.0
y = fashion["target"].astype(np.int64)
```

### 3. CIFAR-10 (Flattened)
```python
# Flatten 32x32x3 = 3072 input dimensions
model = MLP(d=3072, h=256, c=10, init='he')
```

### 4. Tabular Data
```python
# UCI datasets (Iris, Wine, Adult)
model = MLP(d=feature_dim, h=64, c=num_classes, init='xavier')
```

### 5. Regression (MSE Loss)
```python
# Modify loss function in training loop
def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)
```

## 🔧 Extending the Core

### Adding New Models
```python
# src/nn_core/models/cnn.py
class CNN:
    def __init__(self, ...):
        # Implementation
    def forward(self, X):
        # Implementation
    def backward(self, y_onehot):
        # Implementation
    def parameters(self):
        # Implementation
```

### Adding New Optimizers
```python
# src/nn_core/optim/rmsprop.py
class RMSprop:
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8):
        # Implementation
    def step(self, grads):
        # Implementation
```

### Adding New Schedulers
```python
# src/nn_core/schedulers/exponential.py
class ExponentialLR:
    def __init__(self, base_lr, gamma):
        # Implementation
    def lr_at(self, epoch):
        # Implementation
```

## 📊 Performance Tips

### Optimizer Selection
- **AdamW**: Best general-purpose optimizer
- **Nesterov**: Good for convex problems
- **SGD**: Baseline, good for understanding dynamics

### Initialization
- **He**: Best for ReLU networks (default)
- **Xavier**: Good for tanh/sigmoid
- **Normal**: Simple baseline

### Learning Rate Scheduling
- **Cosine**: Smooth decay, good for long training
- **Step**: Simple, good for quick experiments
- **None**: Constant LR, good for short training

### Gradient Clipping
- Use `clip_grad=1.0` for stability
- Higher values for more aggressive clipping
- `0.0` to disable clipping

## 🧪 Experimentation

### Hyperparameter Sweeps
```python
# Compare optimizers
for opt in ['sgd', 'momentum', 'adam', 'adamw']:
    optim = Optim(params, opt=opt, lr=0.001)
    # Train and evaluate

# Compare initializations  
for init in ['he', 'xavier', 'normal']:
    model = MLP(init=init)
    # Train and evaluate

# Compare schedulers
for sched in ['none', 'step', 'cosine']:
    scheduler = LRScheduler(0.001, kind=sched)
    # Train and evaluate
```

### Multi-Dataset Testing
```python
datasets = {
    'MNIST': load_mnist(),
    'Fashion-MNIST': load_fashion_mnist(),
    'CIFAR-10': load_cifar10()
}

for name, (X_train, X_val, y_train, y_val) in datasets.items():
    model = MLP(d=X_train.shape[1], h=128, c=10)
    # Train and evaluate
```

## 📚 Educational Value

This core provides:
- **Mathematical Understanding**: All operations implemented from scratch
- **Modular Design**: Easy to understand and extend
- **Real-world Applicability**: Works on actual datasets
- **Research Flexibility**: Easy to experiment with new ideas

Perfect for:
- Learning neural network fundamentals
- Understanding optimization algorithms
- Experimenting with different architectures
- Research prototyping
- Educational demonstrations

## 🤝 Contributing

To extend this core:
1. Add new models in `src/nn_core/models/`
2. Add new optimizers in `src/nn_core/optim/`
3. Add new schedulers in `src/nn_core/schedulers/`
4. Add new training utilities in `src/nn_core/training/`
5. Update documentation and examples

---

**Happy Learning! 🚀**
