#!/usr/bin/env python3
"""
Download MNIST sample images for testing the MLP model
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import os
import argparse

def download_mnist_samples(num_samples=10, save_dir="data/raw/mnist_samples"):
    """Download and save MNIST sample images"""
    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0
    y = mnist["target"].astype(np.int64)
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get unique samples (one per digit)
    unique_digits = []
    for digit in range(10):
        indices = np.where(y == digit)[0]
        if len(indices) > 0:
            # Take the first sample for each digit
            unique_digits.append(indices[0])
    
    # If we need more samples, add random ones
    if num_samples > 10:
        remaining = num_samples - 10
        other_indices = np.random.choice(len(X), remaining, replace=False)
        unique_digits.extend(other_indices)
    
    # Save samples
    samples = []
    for i, idx in enumerate(unique_digits[:num_samples]):
        image = X[idx].reshape(28, 28)
        label = y[idx]
        
        # Save as numpy array
        sample_file = os.path.join(save_dir, f"sample_{i:02d}_digit_{label}.npy")
        np.save(sample_file, X[idx])
        
        # Save as image
        img_file = os.path.join(save_dir, f"sample_{i:02d}_digit_{label}.png")
        plt.figure(figsize=(2, 2))
        plt.imshow(image, cmap='gray')
        plt.title(f'Digit: {label}')
        plt.axis('off')
        plt.savefig(img_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        samples.append({
            'index': i,
            'digit': label,
            'npy_file': sample_file,
            'img_file': img_file,
            'pixels': X[idx]
        })
        
        print(f"Saved sample {i:02d}: digit {label}")
    
    # Save metadata
    metadata_file = os.path.join(save_dir, "metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write("MNIST Sample Images\n")
        f.write("==================\n\n")
        for sample in samples:
            f.write(f"Sample {sample['index']:02d}: digit {sample['digit']}\n")
            f.write(f"  NPY file: {sample['npy_file']}\n")
            f.write(f"  Image file: {sample['img_file']}\n\n")
    
    print(f"\nSaved {len(samples)} samples to {save_dir}")
    print(f"Metadata saved to {metadata_file}")
    
    return samples

def main():
    parser = argparse.ArgumentParser(description='Download MNIST sample images')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to download')
    parser.add_argument('--save-dir', default='data/raw/mnist_samples', help='Directory to save samples')
    
    args = parser.parse_args()
    
    samples = download_mnist_samples(args.num_samples, args.save_dir)
    
    print(f"\nSample files created:")
    for sample in samples:
        print(f"  {sample['npy_file']}")
        print(f"  {sample['img_file']}")

if __name__ == "__main__":
    main()
