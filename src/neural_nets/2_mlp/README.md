# 2-Layer MLP Implementation (Week 2)

## 🧠 Overview

This directory contains a complete implementation of a 2-layer Multi-Layer Perceptron (MLP) in pure NumPy, trained on MNIST digit classification. This is part of Week 2 of the AI Learning Journey.

## 📁 Files

| File | Description |
|------|-------------|
| `mlp_mnist_numpy.py` | Main MLP implementation with CLI |
| `download_mnist_samples.py` | Sample image downloader |
| `mlp_summary.md` | Mathematical documentation |
| `CLI_DOCUMENTATION.md` | Complete CLI reference |
| `CLI_QUICK_REFERENCE.md` | Quick command reference |
| `README.md` | This file |

## 🚀 Quick Start

### 1. Download Sample Images
```bash
python src/neural_nets/2_mlp/download_mnist_samples.py
```

### 2. Train the Model
```bash
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --epochs 10
```

### 3. Test with Images
```bash
python src/neural_nets/2_mlp/mlp_mnist_numpy.py --test-all-samples
```

## 🎯 Learning Objectives

This implementation demonstrates:

### **Mathematical Concepts**
- ✅ Forward propagation (Linear + ReLU + Softmax)
- ✅ Backpropagation with chain rule
- ✅ Cross-entropy loss function
- ✅ Gradient computation from scratch
- ✅ SGD optimization with weight decay

### **Programming Skills**
- ✅ Pure NumPy implementation
- ✅ Object-oriented design
- ✅ CLI development
- ✅ Data visualization
- ✅ Model evaluation

### **Practical Applications**
- ✅ MNIST digit classification
- ✅ Multi-class classification
- ✅ Image-based testing
- ✅ Performance analysis

## 📊 Performance

- **Model Size**: 101,770 parameters
- **Training Time**: ~0.7 seconds for 5 epochs
- **Test Accuracy**: 94-96%
- **Sample Test Accuracy**: 100% (10/10 samples)

## 🔬 Architecture

```
Input (784) → Hidden (128, ReLU) → Output (10, Softmax)
```

- **Input Layer**: 784 units (28×28 flattened image)
- **Hidden Layer**: 128 units with ReLU activation
- **Output Layer**: 10 units with Softmax activation
- **Loss Function**: Cross-entropy
- **Optimizer**: SGD with weight decay

## 📚 Documentation

### **For Learning**
- `mlp_summary.md` - Mathematical foundations and equations
- `README.md` - This overview

### **For Usage**
- `CLI_DOCUMENTATION.md` - Complete CLI reference
- `CLI_QUICK_REFERENCE.md` - Quick command reference

## 🎓 Week 2 Progress

### **Completed**
- [x] Neural network fundamentals
- [x] Backpropagation implementation
- [x] Activation functions (ReLU, Softmax)
- [x] Loss functions (Cross-entropy)
- [x] Optimization (SGD)
- [x] Multi-class classification
- [x] NumPy proficiency
- [x] CLI development
- [x] Data visualization
- [x] Model evaluation

### **Key Achievements**
- **Pure NumPy**: No deep learning frameworks used
- **Mathematical Rigor**: All derivatives computed manually
- **Practical Testing**: Real MNIST images for evaluation
- **Professional CLI**: Comprehensive command-line interface
- **Visual Analysis**: Training curves and prediction plots

## 🚀 Next Steps (Week 3)

After mastering this NumPy implementation:
1. **PyTorch Introduction**: Learn tensor operations
2. **Automatic Differentiation**: Use `autograd`
3. **Higher-Level APIs**: `nn.Module`, optimizers
4. **GPU Acceleration**: CUDA support
5. **Modern Architectures**: CNNs, RNNs

## 🔧 Development

### **Extending the Implementation**
- Add more activation functions (Tanh, Sigmoid)
- Implement different optimizers (Adam, RMSprop)
- Add regularization techniques (Dropout, BatchNorm)
- Create interactive visualizations
- Add model saving/loading

### **Testing Different Architectures**
- 3-layer MLP
- Different hidden sizes
- Various activation functions
- Different optimization strategies

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 94.65% |
| **Training Time** | 0.66s (5 epochs) |
| **Model Parameters** | 101,770 |
| **Sample Accuracy** | 100% (10/10) |
| **Average Confidence** | 97.2% |

## 🎉 Success Criteria Met

- [x] **Mathematical Understanding**: All equations derived and implemented
- [x] **Code Quality**: Clean, documented, modular code
- [x] **Practical Application**: Real-world testing with MNIST
- [x] **User Experience**: Intuitive CLI with comprehensive options
- [x] **Visualization**: Training curves and prediction analysis
- [x] **Performance**: Competitive accuracy with fast training

---

**Ready for Week 3: PyTorch Mastery! 🚀**
