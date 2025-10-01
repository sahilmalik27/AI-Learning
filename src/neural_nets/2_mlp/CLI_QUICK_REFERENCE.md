# 2-Layer MLP CLI Quick Reference

## 🚀 Essential Commands

### Training
```bash
# Basic training
python src/neural_nets/2_mlp/mlp_mnist_numpy.py

# Custom training
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --epochs 20 --lr 0.05 --batch-size 64
```

### Sample Management
```bash
# Download samples
python src/neural_nets/2_mlp/download_mnist_samples.py

# List samples
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --list-samples
```

### Testing
```bash
# Test all samples
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --test-all-samples

# Test single image
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --predict-image data/raw/mnist_samples/sample_00_digit_0.npy
```

## 📋 All Options

| Command | Description |
|---------|-------------|
| `--epochs N` | Number of training epochs |
| `--batch-size N` | Batch size |
| `--lr FLOAT` | Learning rate |
| `--weight-decay FLOAT` | Weight decay |
| `--seed N` | Random seed |
| `--no-plots` | Disable plots |
| `--predict-image FILE` | Predict from image file |
| `--list-samples` | List available samples |
| `--test-all-samples` | Test all samples |

## 🎯 Common Workflows

### 1. First Time Setup
```bash
# Download samples
python src/neural_nets/2_mlp/download_mnist_samples.py

# Train model
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --epochs 10

# Test samples
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --test-all-samples
```

### 2. Quick Testing
```bash
# Fast training + testing
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --epochs 5 --no-plots --test-all-samples
```

### 3. Experimentation
```bash
# Different learning rates
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --lr 0.01 --epochs 10
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --lr 0.1 --epochs 10
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --lr 0.5 --epochs 10
```

## 📊 Expected Outputs

### Training
```
Epoch 01 | train_loss=0.7585 | test_loss=0.3443 | test_acc=0.9010 | time=0.14s
Epoch 02 | train_loss=0.3122 | test_loss=0.2819 | test_acc=0.9175 | time=0.13s
...
Final test accuracy: 0.9465
```

### Testing
```
✅ Sample  0: sample_00_digit_0.npy
    True: 0, Predicted: 0, Confidence: 1.000
✅ Sample  1: sample_01_digit_1.npy
    True: 1, Predicted: 1, Confidence: 0.988
...
Test Results: 10/10 correct (100.0%)
```

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No samples found" | Run `download_mnist_samples.py` first |
| "Image file not found" | Check file path with `--list-samples` |
| Slow training | Use `--no-plots` flag |
| Memory issues | Reduce `--batch-size` |

## 📁 File Locations

- **Samples**: `data/raw/mnist_samples/`
- **Plots**: `experiments/plots/mlp_training_history.png`
- **Scripts**: `src/neural_nets/2_mlp/`

---
**Quick Start**: `python src/neural_nets/2_mlp/mlp_mnist_numpy.py --help`
