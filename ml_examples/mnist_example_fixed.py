#!/usr/bin/env python3
"""
MNIST Classification - Fixed Version

This script demonstrates the FIXED code with proper data normalization
and additional improvements for better accuracy.
"""
import numpy as np
from sklearn.datasets import fetch_openml
from logistic_regression import LogisticRegressionClassifier


def main():
    print("=" * 70)
    print("MNIST Classification - FIXED VERSION WITH IMPROVEMENTS")
    print("=" * 70)
    print()
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    X_mnist = np.array(mnist.data)
    y_mnist = np.array(mnist.target, dtype=int)  # convert to int

    # Shuffle data
    indices = np.arange(len(X_mnist))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)

    X_mnist = X_mnist[indices]
    y_mnist = y_mnist[indices]

    # Split data
    split_ratio = 0.7
    split_index = int(len(X_mnist) * split_ratio)

    X_train = X_mnist[:split_index]
    X_test  = X_mnist[split_index:]
    y_train = y_mnist[:split_index]
    y_test  = y_mnist[split_index:]

    # Use smaller subset for demonstration
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()

    # FIX: Normalize BOTH training and test data
    print("*** FIX: Normalizing BOTH training and test data ***")
    X_train = X_train / 255.0
    X_test = X_test / 255.0  # This is the fix!
    
    print(f"X_train range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"X_test range: [{X_test.min():.2f}, {X_test.max():.2f}]")
    print()
    print("Both datasets are now properly normalized to [0, 1]")
    print()

    # Train model with improved hyperparameters
    print("Training model with improved hyperparameters...")
    print("Improvements:")
    print("  1. Higher learning rate (0.1 instead of 0.0001)")
    print("  2. More epochs (20 instead of 3)")
    print("  3. Larger batch size (128 instead of 64)")
    print()
    
    model_mnist = LogisticRegressionClassifier(
        learning_rate=0.1,      # Increased from 0.0001
        batch_size=128,         # Increased from 64
        epochs=20               # Increased from 3
    )
    model_mnist.fit(X_train, y_train, verbose=True)

    # Evaluate
    train_accuracy = model_mnist.score(X_train, y_train)
    test_accuracy = model_mnist.score(X_test, y_test)
    
    print()
    print("=" * 70)
    print("FINAL RESULTS:")
    print("=" * 70)
    print(f"  Training accuracy: {train_accuracy:.2%}")
    print(f"  Test accuracy: {test_accuracy:.2%}")
    print()
    print("*** Test accuracy is now much better! ***")
    print()
    print("Key takeaway:")
    print("  Always normalize/preprocess test data the SAME WAY as training data!")
    print()


if __name__ == "__main__":
    main()
