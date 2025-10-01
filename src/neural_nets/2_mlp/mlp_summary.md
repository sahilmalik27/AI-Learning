# 2-Layer MLP: Concepts & Backpropagation

This document summarizes the formulas and intuitions we discussed for a simple **2-layer Multi-Layer Perceptron (MLP)** applied to MNIST. The current codebase decouples the MLP and training logic into a reusable core (`src/nn_core`) with:

- Model: `MLP` (He/Xavier/Normal init)
- Optimizers: SGD, Momentum, Nesterov, Adam, AdamW
- Schedulers: None, Step, Cosine (with warmup)
- Training: Mini-batch loop with gradient clipping and callbacks

---

## 1. Network Architecture

- **Input layer**: 784 units (28×28 pixel images flattened).
- **Hidden layer**: 128 units with **ReLU** activation.
- **Output layer**: 10 units with **Softmax** activation (digits 0–9).

Diagram:

```
Input (784) → Hidden (128, ReLU) → Output (10, Softmax)
```

---

## 2. Forward Pass

1. **First linear transformation**  
\[ a_1 = XW_1 + b_1 \]
- Shapes:  
  - \(X \in \mathbb{R}^{N \times d}\)  
  - \(W_1 \in \mathbb{R}^{d \times h}\)  
  - \(b_1 \in \mathbb{R}^h\)  

2. **ReLU activation**  
\[ h = \mathrm{ReLU}(a_1) = \max(0, a_1) \]

3. **Second linear transformation**  
\[ z = hW_2 + b_2 \]

4. **Softmax output**  
\[ p = \mathrm{softmax}(z) \]
\[ p_i = \frac{e^{z_i}}{\sum_j e^{z_j}} \]

---

## 3. Loss Function (Cross-Entropy)

For one-hot labels \(Y \in \mathbb{R}^{N \times C}\):  

\[ L = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C Y_{ic} \log P_{ic} \]

---

## 4. Backpropagation

### Output Layer (Softmax + Cross-Entropy)
\[ \delta z = \frac{P - Y}{N} \]

### Gradients for Second Layer
- Weights: \( \nabla_{W_2} = h^T \delta z \)  
- Bias: \( \nabla_{b_2} = \sum_i \delta z_{i:} \)  
- Backprop to hidden: \( \delta h = \delta z W_2^T \)

### ReLU Backpropagation
\[ \delta a_1 = \delta h \odot 1[a_1 > 0] \]

### Gradients for First Layer
- Weights: \( \nabla_{W_1} = X^T \delta a_1 \)  
- Bias: \( \nabla_{b_1} = \sum_i \delta a_{1,i:} \)

---

## 5. Parameter Update (Optimizers)

For each parameter \( \theta \in \{W_1, b_1, W_2, b_2\} \):  
\[ \theta \leftarrow \theta - \eta \, \nabla_\theta \]

Where \( \eta \) is the learning rate. In practice, we often use advanced optimizers (e.g., Momentum, Adam/AdamW) and learning-rate schedules (Step/Cosine with warmup) for faster and more stable convergence. L2 regularization (weight decay) and gradient clipping are also used to improve generalization and stability.

---

## 6. Key Intuitions

- **ReLU** passes gradients only when input > 0 (otherwise derivative = 0).  
- **Softmax + Cross-Entropy** simplifies gradient at output: \( \delta z = p - y \).  
- MLP learns **non-linear boundaries**, unlike logistic regression.  
- Gradients flow backwards layer by layer using the chain rule.

---

## 7. Sanity Check

- Shapes of gradients:  
  - \( \nabla_{W_2} \in \mathbb{R}^{h \times C} \)  
  - \( \nabla_{W_1} \in \mathbb{R}^{d \times h} \)  
- If \(a_1 \le 0\), ReLU output = 0 and gradient = 0 (dead unit).

---

**End of Summary**
