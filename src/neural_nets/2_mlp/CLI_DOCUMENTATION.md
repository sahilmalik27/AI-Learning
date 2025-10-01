# 2-Layer MLP CLI Documentation

## 🧠 Overview

This document provides comprehensive documentation for the 2-layer Multi-Layer Perceptron (MLP) implementation in NumPy, including all CLI commands and usage examples.

## 📁 Files Structure

```
src/neural_nets/2_mlp/
├── mlp_mnist_numpy.py          # Main MLP implementation with CLI
├── download_mnist_samples.py   # Sample image downloader
├── mlp_summary.md              # Mathematical documentation
└── CLI_DOCUMENTATION.md        # This file
```

## 🚀 Quick Start

### 1. Basic Training
```bash
# Train with default parameters (10 epochs)
python src/neural_nets/2_mlp/mlp_mnist_numpy.py

# Train with custom parameters
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --epochs 20 --lr 0.05 --batch-size 64
```

### 2. Download Sample Images
```bash
# Download 10 sample images (one per digit)
python src/neural_nets/2_mlp/download_mnist_samples.py

# Download more samples
python src/neural_nets/2_mlp/download_mnist_samples.py --num-samples 20
```

### 3. Test with Images
```bash
# List available samples
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --list-samples

# Test all samples
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --test-all-samples

# Test specific image
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --predict-image data/raw/mnist_samples/sample_00_digit_0.npy
```

## 📋 Complete CLI Reference

### Main Script: `mlp_mnist_numpy.py`

#### **Training Options**
```bash
python src/neural_nets/2_mlp/mlp_mnist_numpy.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--epochs` | int | 10 | Number of training epochs |
| `--batch-size` | int | 128 | Batch size for training |
| `--lr` | float | 0.1 | Learning rate |
| `--weight-decay` | float | 1e-4 | L2 weight decay |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--no-plots` | flag | False | Disable plot generation |

#### **Prediction Options**
```bash
# Predict from pixel values (784 values)
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --predict [784 pixel values]

# Predict from image file
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --predict-image <path_to_npy_file>

# List available sample images
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --list-samples

# Test all available samples
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --test-all-samples
```

### Sample Downloader: `download_mnist_samples.py`

```bash
python src/neural_nets/2_mlp/download_mnist_samples.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--num-samples` | int | 10 | Number of samples to download |
| `--save-dir` | str | `data/raw/mnist_samples` | Directory to save samples |

## 🎯 Usage Examples

### Example 1: Basic Training and Evaluation
```bash
# Train the model
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --epochs 10

# Expected output:
# Loading MNIST dataset...
# Training set: 60000 samples
# Test set: 10000 samples
# Model initialized with 101770 parameters
# 
# Starting training for 10 epochs...
# ============================================================
# Epoch 01 | train_loss=0.7585 | test_loss=0.3443 | test_acc=0.9010 | time=0.14s
# Epoch 02 | train_loss=0.3122 | test_loss=0.2819 | test_acc=0.9175 | time=0.13s
# ...
# ============================================================
# Training completed in 1.32 seconds
# Final test accuracy: 0.9465
```

### Example 2: Download and Test Sample Images
```bash
# Step 1: Download sample images
python src/neural_nets/2_mlp/download_mnist_samples.py --num-samples 10

# Step 2: List available samples
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --list-samples

# Output:
# Available sample images:
# ==================================================
#  0. sample_00_digit_0.npy (digit: 0)
#  1. sample_01_digit_1.npy (digit: 1)
#  2. sample_02_digit_2.npy (digit: 2)
#  ...

# Step 3: Test all samples
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --test-all-samples

# Output:
# 🧪 Testing 10 sample images...
# ============================================================
# ✅ Sample  0: sample_00_digit_0.npy
#     True: 0, Predicted: 0, Confidence: 1.000
# ✅ Sample  1: sample_01_digit_1.npy
#     True: 1, Predicted: 1, Confidence: 0.988
# ...
# ============================================================
# Test Results: 10/10 correct (100.0%)
```

### Example 3: Single Image Prediction
```bash
# Predict from specific image
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --predict-image data/raw/mnist_samples/sample_00_digit_0.npy

# Output:
# 🔢 Digit Prediction from Image
# ========================================
# Image file: data/raw/mnist_samples/sample_00_digit_0.npy
# Predicted digit: 0
# Confidence: 0.9995 (100.0%)
# 
# Class probabilities:
#   0: 0.9995 (100.0%)
#   1: 0.0000 (0.0%)
#   2: 0.0003 (0.0%)
#   ...
```

### Example 4: Custom Training Parameters
```bash
# Train with custom parameters
python src/neural_nets/2_mlp/mlp_mnist_numpy.py \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.05 \
    --weight-decay 1e-3 \
    --seed 123

# Train without plots (faster)
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --epochs 5 --no-plots
```

## 📊 Output Files

### Training Outputs
- **Plots**: `experiments/plots/mlp_training_history.png`
  - Training and test loss curves
  - Test accuracy progression

### Sample Images
- **Location**: `data/raw/mnist_samples/`
- **Files**: 
  - `sample_XX_digit_Y.npy` - NumPy arrays (784 pixels)
  - `sample_XX_digit_Y.png` - Visual images
  - `metadata.txt` - Sample information

## 🔧 Advanced Usage

### Custom Sample Directory
```bash
# Download to custom directory
python src/neural_nets/2_mlp/download_mnist_samples.py \
    --num-samples 20 \
    --save-dir data/custom_samples

# Use custom directory for testing
python src/neural_nets/2_mlp/mlp_mnist_numpy.py \
    --predict-image data/custom_samples/sample_00_digit_0.npy
```

### Batch Testing with Different Seeds
```bash
# Test with different random seeds
for seed in 42 123 456; do
    echo "Testing with seed $seed"
    python src/neural_nets/2_mlp/mlp_mnist_numpy.py \
        --test-all-samples \
        --seed $seed
done
```

### Performance Comparison
```bash
# Compare different learning rates
for lr in 0.01 0.05 0.1 0.2; do
    echo "Testing with learning rate $lr"
    python src/neural_nets/2_mlp/mlp_mnist_numpy.py \
        --epochs 5 \
        --lr $lr \
        --no-plots
done
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "No samples found" Error
```bash
# Solution: Download samples first
python src/neural_nets/2_mlp/download_mnist_samples.py
```

#### 2. "Image file not found" Error
```bash
# Check if file exists
ls data/raw/mnist_samples/

# List available samples
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --list-samples
```

#### 3. Memory Issues with Large Batches
```bash
# Use smaller batch size
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --batch-size 32
```

#### 4. Slow Training
```bash
# Disable plots for faster training
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --no-plots
```

## 📈 Performance Benchmarks

### Typical Results
- **Training Time**: ~0.7 seconds for 5 epochs
- **Final Accuracy**: 94-96% on test set
- **Model Parameters**: 101,770 parameters
- **Memory Usage**: ~50MB during training

### Hardware Requirements
- **RAM**: Minimum 2GB, Recommended 4GB
- **CPU**: Any modern processor
- **Storage**: ~100MB for samples and plots

## 🔬 Mathematical Background

The implementation includes:
- **Forward Pass**: Linear layers + ReLU + Softmax
- **Backward Pass**: Manual backpropagation with chain rule
- **Loss Function**: Cross-entropy with numerical stability
- **Optimization**: SGD with weight decay

See `mlp_summary.md` for detailed mathematical documentation.

## 🎓 Learning Objectives

This implementation demonstrates:
- ✅ Neural network fundamentals
- ✅ Backpropagation from scratch
- ✅ Activation functions (ReLU, Softmax)
- ✅ Loss functions (Cross-entropy)
- ✅ Optimization (SGD with weight decay)
- ✅ Multi-class classification
- ✅ NumPy proficiency
- ✅ CLI development
- ✅ Data visualization
- ✅ Model evaluation

## 📚 Next Steps

After mastering this implementation:
1. **Week 3**: Move to PyTorch implementation
2. **Week 4**: Add more layers and advanced architectures
3. **Week 5**: Implement attention mechanisms
4. **Week 6**: Build transformer models

## 🤝 Contributing

To extend this implementation:
1. Add more activation functions
2. Implement different optimizers
3. Add regularization techniques
4. Create interactive visualizations
5. Add model saving/loading functionality

---

**Happy Learning! 🚀**
