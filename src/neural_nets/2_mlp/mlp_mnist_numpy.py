# mlp_mnist_numpy.py
# 2-layer MLP (784 -> 128 -> 10) with ReLU + Softmax CE, pure NumPy

"""
2-Layer MLP (MNIST) — Documentation
===================================

This script implements a simple 2-layer Multi-Layer Perceptron (MLP) in NumPy,
trained on MNIST. It includes both the code and the math concepts for forward
and backward propagation.

--------------------------------------------------------
1. Network Architecture
--------------------------------------------------------
- Input: 784 units (28x28 image flattened)
- Hidden: 128 units with ReLU activation
- Output: 10 units with Softmax activation (digits 0–9)

Diagram:
    Input (784) → Hidden (128, ReLU) → Output (10, Softmax)

--------------------------------------------------------
2. Forward Pass
--------------------------------------------------------
- First linear layer:
    a1 = X W1 + b1      # shape: (N, h)
- ReLU activation:
    h = ReLU(a1) = max(0, a1)
- Second linear layer:
    z = h W2 + b2       # shape: (N, c)
- Softmax probabilities:
    p = softmax(z) = exp(z) / sum(exp(z))

--------------------------------------------------------
3. Loss Function (Cross-Entropy)
--------------------------------------------------------
Given true one-hot labels Y and predictions P:

    L = - (1/N) Σ Σ Y_ic log(P_ic)

This measures how far predicted probabilities are from the true labels.

--------------------------------------------------------
4. Backpropagation
--------------------------------------------------------
Output layer (Softmax + Cross-Entropy):
    δz = (p - Y) / N

Gradients for second layer:
    ∂L/∂W2 = h^T δz
    ∂L/∂b2 = Σ δz
    δh = δz W2^T

ReLU backprop:
    δa1 = δh ⊙ 1[a1 > 0]

Gradients for first layer:
    ∂L/∂W1 = X^T δa1
    ∂L/∂b1 = Σ δa1

--------------------------------------------------------
5. SGD Update
--------------------------------------------------------
For each parameter θ ∈ {W1, b1, W2, b2}:
    θ ← θ - η * ∂L/∂θ

where η is the learning rate.

--------------------------------------------------------
6. Key Intuitions
--------------------------------------------------------
- ReLU only passes gradient if input > 0 (otherwise derivative = 0).
- Softmax + CE simplifies gradient at output: δz = p - y.
- MLP learns nonlinear decision boundaries, unlike logistic regression.
- Backprop = applying the chain rule layer by layer.

--------------------------------------------------------
7. Sanity Checks
--------------------------------------------------------
- ∂L/∂W2 has shape (h, c)
- ∂L/∂W1 has shape (d, h)
- If a1 ≤ 0, ReLU output = 0 and gradient = 0 (dead neuron).
"""


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import time
import os
import glob

# ---------- helpers ----------
def one_hot(y, num_classes):
    oh = np.zeros((y.size, num_classes), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh

def relu(x):
    return np.maximum(0, x)

def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)   # stability
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def cross_entropy(p, y_onehot):
    eps = 1e-12
    return -np.mean(np.sum(y_onehot * np.log(p + eps), axis=1))

def accuracy(p, y_true):
    return (np.argmax(p, axis=1) == y_true).mean()

# ---------- model ----------
class MLP:
    def __init__(self, d=784, h=128, c=10, weight_scale=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, weight_scale, size=(d, h)).astype(np.float32)
        self.b1 = np.zeros((h,), dtype=np.float32)
        self.W2 = rng.normal(0, weight_scale, size=(h, c)).astype(np.float32)
        self.b2 = np.zeros((c,), dtype=np.float32)
        self.cache = {}

    def forward(self, X):
        a1 = X @ self.W1 + self.b1          # (N, h)
        h  = relu(a1)                        # (N, h)
        z  = h @ self.W2 + self.b2           # (N, c)
        p  = softmax(z)                      # (N, c)
        self.cache = {"X": X, "a1": a1, "h": h, "z": z, "p": p}
        return p

    def backward(self, y_onehot):
        X, a1, h, z, p = (self.cache[k] for k in ["X","a1","h","z","p"])
        N = X.shape[0]

        dz  = (p - y_onehot) / N            # (N, c)
        dW2 = h.T @ dz                       # (h, c)
        db2 = dz.sum(axis=0)                 # (c,)

        dh  = dz @ self.W2.T                 # (N, h)
        da1 = dh * (a1 > 0).astype(np.float32)

        dW1 = X.T @ da1                      # (d, h)
        db1 = da1.sum(axis=0)                # (h,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads, lr=0.1, weight_decay=0.0):
        # L2 weight decay on weights (not biases)
        if weight_decay:
            grads["W1"] += weight_decay * self.W1
            grads["W2"] += weight_decay * self.W2
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

# ---------- data ----------
def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0
    y = mnist["target"].astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# ---------- train loop ----------
def train(num_epochs=10, batch_size=128, lr=0.1, weight_decay=1e-4, seed=42, save_plots=True):
    print("Loading MNIST dataset...")
    X_train, X_test, y_train, y_test = load_mnist()
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    model = MLP(seed=seed)
    print(f"Model initialized with {sum(p.size for p in [model.W1, model.b1, model.W2, model.b2])} parameters")

    rng = np.random.default_rng(seed)
    N = X_train.shape[0]
    steps = (N + batch_size - 1) // batch_size

    # Training history
    train_losses = []
    test_losses = []
    test_accuracies = []

    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        idx = rng.permutation(N)
        Xs, ys = X_train[idx], y_train[idx]

        total_loss = 0.0
        for s in range(steps):
            a, b = s * batch_size, min(N, (s + 1) * batch_size)
            xb, yb = Xs[a:b], ys[a:b]
            yb_oh = one_hot(yb, 10)

            pb = model.forward(xb)
            loss = cross_entropy(pb, yb_oh)
            total_loss += loss * (b - a)

            grads = model.backward(yb_oh)
            model.step(grads, lr=lr, weight_decay=weight_decay)

        # Evaluation
        p_test = model.forward(X_test)
        test_acc = accuracy(p_test, y_test)
        test_loss = cross_entropy(p_test, one_hot(y_test, 10))
        
        train_loss = total_loss / N
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
              f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f} | "
              f"time={epoch_time:.2f}s")

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Final test accuracy: {test_accuracies[-1]:.4f}")
    
    if save_plots:
        plot_training_history(train_losses, test_losses, test_accuracies)
    
    return model, train_losses, test_losses, test_accuracies

def plot_training_history(train_losses, test_losses, test_accuracies):
    """Plot training history"""
    os.makedirs('experiments/plots', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(test_losses, label='Test Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(test_accuracies, label='Test Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/plots/mlp_training_history.png', dpi=300, bbox_inches='tight')
    print("Training plots saved to experiments/plots/mlp_training_history.png")
    plt.show()

def predict_digit(model, X_sample):
    """Predict a single digit"""
    p = model.forward(X_sample.reshape(1, -1))
    predicted_class = np.argmax(p[0])
    confidence = p[0][predicted_class]
    return predicted_class, confidence, p[0]

def load_sample_image(sample_path):
    """Load a sample image from file"""
    if sample_path.endswith('.npy'):
        return np.load(sample_path)
    else:
        raise ValueError("Sample file must be .npy format")

def predict_from_image(model, image_path, show_image=True):
    """Predict digit from image file"""
    # Load image
    X_sample = load_sample_image(image_path)
    
    # Predict
    predicted_class, confidence, probabilities = predict_digit(model, X_sample)
    
    # Display results
    print(f"\n🔢 Digit Prediction from Image")
    print("=" * 40)
    print(f"Image file: {image_path}")
    print(f"Predicted digit: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence:.1%})")
    
    if show_image:
        # Show the image
        plt.figure(figsize=(6, 3))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(X_sample.reshape(28, 28), cmap='gray')
        plt.title(f'Input Image\nPredicted: {predicted_class}')
        plt.axis('off')
        
        # Probability distribution
        plt.subplot(1, 2, 2)
        plt.bar(range(10), probabilities, color='skyblue', alpha=0.7)
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title('Prediction Probabilities')
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print(f"\nClass probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  {i}: {prob:.4f} ({prob:.1%})")
    
    return predicted_class, confidence, probabilities

def list_available_samples(samples_dir="data/raw/mnist_samples"):
    """List available sample images"""
    if not os.path.exists(samples_dir):
        print(f"No samples found in {samples_dir}")
        print("Run: python src/neural_nets/2_mlp/download_mnist_samples.py")
        return []
    
    npy_files = glob.glob(os.path.join(samples_dir, "*.npy"))
    samples = []
    
    for file_path in sorted(npy_files):
        filename = os.path.basename(file_path)
        # Extract digit from filename (sample_XX_digit_Y.npy)
        parts = filename.split('_')
        if len(parts) >= 4:
            digit = parts[3].split('.')[0]
            samples.append({
                'file': file_path,
                'digit': digit,
                'filename': filename
            })
    
    return samples

def main():
    parser = argparse.ArgumentParser(description='2-Layer MLP on MNIST (NumPy)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--predict', nargs=784, type=float, help='Predict digit from 784 pixel values')
    parser.add_argument('--predict-image', type=str, help='Predict digit from image file (.npy)')
    parser.add_argument('--list-samples', action='store_true', help='List available sample images')
    parser.add_argument('--test-all-samples', action='store_true', help='Test all available samples')
    
    args = parser.parse_args()
    
    if args.list_samples:
        # List available samples
        samples = list_available_samples()
        if samples:
            print("Available sample images:")
            print("=" * 50)
            for i, sample in enumerate(samples):
                print(f"{i:2d}. {sample['filename']} (digit: {sample['digit']})")
        return
    
    if args.test_all_samples:
        # Test all available samples
        samples = list_available_samples()
        if not samples:
            return
        
        print("Training a quick model for testing...")
        model, _, _, _ = train(num_epochs=5, batch_size=128, lr=0.1, 
                              weight_decay=1e-4, seed=args.seed, save_plots=False)
        
        print(f"\n🧪 Testing {len(samples)} sample images...")
        print("=" * 60)
        
        correct = 0
        for i, sample in enumerate(samples):
            predicted_class, confidence, _ = predict_from_image(model, sample['file'], show_image=False)
            true_digit = int(sample['digit'])
            is_correct = predicted_class == true_digit
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} Sample {i:2d}: {sample['filename']}")
            print(f"    True: {true_digit}, Predicted: {predicted_class}, Confidence: {confidence:.3f}")
        
        accuracy = correct / len(samples)
        print("=" * 60)
        print(f"Test Results: {correct}/{len(samples)} correct ({accuracy:.1%})")
        return
    
    if args.predict_image:
        # Predict from image file
        print("Training a quick model for prediction...")
        model, _, _, _ = train(num_epochs=5, batch_size=128, lr=0.1, 
                              weight_decay=1e-4, seed=args.seed, save_plots=False)
        
        if not os.path.exists(args.predict_image):
            print(f"Error: Image file '{args.predict_image}' not found")
            return
        
        predict_from_image(model, args.predict_image)
        return
    
    if args.predict:
        # Predict from pixel values
        print("Training a quick model for prediction...")
        model, _, _, _ = train(num_epochs=5, batch_size=128, lr=0.1, 
                              weight_decay=1e-4, seed=args.seed, save_plots=False)
        
        if len(args.predict) != 784:
            print("Error: Must provide exactly 784 pixel values (28x28 image)")
            return
        
        X_sample = np.array(args.predict, dtype=np.float32)
        predicted_class, confidence, probabilities = predict_digit(model, X_sample)
        
        print(f"\n🔢 Digit Prediction")
        print("=" * 30)
        print(f"Predicted digit: {predicted_class}")
        print(f"Confidence: {confidence:.4f} ({confidence:.1%})")
        print(f"\nClass probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"  {i}: {prob:.4f} ({prob:.1%})")
    else:
        # Training mode
        train(num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              weight_decay=args.weight_decay, seed=args.seed, save_plots=not args.no_plots)

if __name__ == "__main__":
    main()
