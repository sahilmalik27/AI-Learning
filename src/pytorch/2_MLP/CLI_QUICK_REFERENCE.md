# PyTorch 2-Layer MLP - CLI Quick Reference

## 🚀 Quick Start

### Basic Training
```bash
# Train with default settings (10 epochs, early stopping)
python src/pytorch/2_MLP/mnist/mnist.py train --data-dir ./data

# Train with custom parameters
python src/pytorch/2_MLP/mnist/mnist.py train --data-dir ./data --epochs 20 --lr 0.001 --batch-train 128
```

### Prediction
```bash
# Predict from test set samples
python src/pytorch/2_MLP/mnist/mnist.py predict --checkpoint mnist_mlp.pt --data-dir ./data

# Predict single image
python src/pytorch/2_MLP/mnist/mnist.py predict --checkpoint mnist_mlp.pt --image path/to/image.png
```

## 📊 Training Options

### Core Training Parameters
```bash
--epochs EPOCHS              # Max epochs (default: 10)
--batch-train BATCH_TRAIN    # Training batch size (default: 256)
--batch-test BATCH_TEST      # Test batch size (default: 1024)
--lr LR                      # Learning rate (default: 0.001)
--weight-decay WEIGHT_DECAY  # L2 regularization (default: 0.0001)
--dropout DROPOUT            # Dropout rate (default: 0.2)
```

### Early Stopping & LR Scheduling
```bash
--patience PATIENCE          # Early stopping patience (default: 5)
--min-delta MIN_DELTA        # Min improvement for early stopping (default: 0.0001)
--lr-factor LR_FACTOR        # LR reduction factor (default: 0.5)
--lr-patience LR_PATIENCE    # LR scheduler patience (default: 2)
```

### Device & Performance
```bash
--device {auto,cpu,cuda,mps} # Device selection (default: auto)
--num-workers NUM_WORKERS    # DataLoader workers (default: 0)
--seed SEED                  # Random seed (default: 42)
```

## 🎯 Common Training Scenarios

### Conservative Training (Recommended)
```bash
python src/pytorch/2_MLP/mnist/mnist.py train \
  --data-dir ./data \
  --epochs 20 \
  --lr 0.001 \
  --patience 5 \
  --lr-patience 2 \
  --confusion-matrix \
  --predict-grid
```

### Fast Training
```bash
python src/pytorch/2_MLP/mnist/mnist.py train \
  --data-dir ./data \
  --epochs 15 \
  --lr 0.003 \
  --batch-train 512 \
  --patience 3 \
  --lr-factor 0.3
```

### High Accuracy Training
```bash
python src/pytorch/2_MLP/mnist/mnist.py train \
  --data-dir ./data \
  --epochs 30 \
  --lr 0.0005 \
  --weight-decay 0.001 \
  --dropout 0.3 \
  --patience 7 \
  --lr-patience 3
```

## 📈 Visualization Options

### Generate All Plots
```bash
python src/pytorch/2_MLP/mnist/mnist.py train \
  --data-dir ./data \
  --confusion-matrix \
  --predict-grid \
  --show-plots
```

### Custom Output Paths
```bash
python src/pytorch/2_MLP/mnist/mnist.py train \
  --data-dir ./data \
  --curves-path my_training_curves.png \
  --cm-path my_confusion_matrix.png \
  --samples-path my_predictions.png \
  --checkpoint my_model.pt
```

## 🔍 Prediction Examples

### Test Set Predictions
```bash
# Predict 10 random samples from test set
python src/pytorch/2_MLP/mnist/mnist.py predict \
  --checkpoint mnist_mlp.pt \
  --data-dir ./data \
  --k-samples 10

# Predict 5 samples and save to custom path
python src/pytorch/2_MLP/mnist/mnist.py predict \
  --checkpoint mnist_mlp.pt \
  --data-dir ./data \
  --k-samples 5 \
  --samples-path my_predictions.png
```

### Single Image Prediction
```bash
# Predict from existing sample
python src/pytorch/2_MLP/mnist/mnist.py predict \
  --checkpoint mnist_mlp.pt \
  --image ./data/raw/mnist_samples/sample_00_digit_0.png

# Predict with custom output
python src/pytorch/2_MLP/mnist/mnist.py predict \
  --checkpoint mnist_mlp.pt \
  --image path/to/your/image.png \
  --out-image my_prediction.png
```

## ⚡ Performance Tips

### For Apple Silicon (M1/M2/M3)
```bash
# Use MPS acceleration (automatic with --device auto)
python src/pytorch/2_MLP/mnist/mnist.py train --data-dir ./data --device mps
```

### For GPU Training
```bash
# Use CUDA if available
python src/pytorch/2_MLP/mnist/mnist.py train --data-dir ./data --device cuda
```

### For CPU Training
```bash
# Force CPU usage
python src/pytorch/2_MLP/mnist/mnist.py train --data-dir ./data --device cpu --num-workers 4
```

## 🎛️ Advanced Configuration

### Custom Model Architecture
```bash
# Higher dropout for regularization
python src/pytorch/2_MLP/mnist/mnist.py train \
  --data-dir ./data \
  --dropout 0.4 \
  --weight-decay 0.01

# Lower learning rate with more patience
python src/pytorch/2_MLP/mnist/mnist.py train \
  --data-dir ./data \
  --lr 0.0001 \
  --patience 10 \
  --lr-patience 5
```

### Experiment Tracking
```bash
# Save with descriptive names
python src/pytorch/2_MLP/mnist/mnist.py train \
  --data-dir ./data \
  --checkpoint experiments/exp_001_high_lr.pt \
  --curves-path experiments/exp_001_curves.png \
  --lr 0.01 \
  --patience 3
```

## 📁 Output Files

After training, you'll get:
- `mnist_mlp.pt` - Model checkpoint
- `training_curves.png` - Loss and accuracy plots
- `confusion_matrix.png` - Confusion matrix (if --confusion-matrix)
- `pred_samples.png` - Sample predictions (if --predict-grid)

After prediction:
- `pred_single.png` - Single image prediction
- `pred_samples.png` - Batch predictions

## 🐛 Troubleshooting

### Common Issues
```bash
# If you get "device not found" errors
python src/pytorch/2_MLP/mnist/mnist.py train --device cpu

# If training is too slow
python src/pytorch/2_MLP/mnist/mnist.py train --batch-train 512 --num-workers 2

# If you get memory errors
python src/pytorch/2_MLP/mnist/mnist.py train --batch-train 64 --batch-test 128
```

### Check Available Options
```bash
# See all training options
python src/pytorch/2_MLP/mnist/mnist.py train --help

# See all prediction options
python src/pytorch/2_MLP/mnist/mnist.py predict --help
```
