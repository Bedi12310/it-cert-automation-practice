#!/usr/bin/env python3
"""
MNIST Classification - Demonstrating the Problem

This script demonstrates the ORIGINAL code with the BUG:
The test data is NOT normalized, which causes poor accuracy.
"""
import numpy as np
from sklearn.datasets import fetch_openml
from logistic_regression import LogisticRegressionClassifier


def main():
    print("=" * 70)
    print("MNIST Classification - ORIGINAL CODE WITH BUG")
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

    # BUG: Only training data is normalized!
    print("*** BUG: Normalizing ONLY training data ***")
    X_train = X_train / 255
    # X_test is NOT normalized - this is the bug!
    
    print(f"X_train range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"X_test range: [{X_test.min():.2f}, {X_test.max():.2f}]")
    print()
    print("Notice: X_train is in [0, 1] but X_test is in [0, 255]")
    print("This mismatch will cause poor test accuracy!")
    print()

    # Train model
    print("Training model...")
    model_mnist = LogisticRegressionClassifier(
        learning_rate=0.1,
        batch_size=64,
        epochs=10
    )
    model_mnist.fit(X_train, y_train, verbose=False)

    # Evaluate
    train_accuracy = model_mnist.score(X_train, y_train)
    test_accuracy = model_mnist.score(X_test, y_test)
    
    print()
    print("Results:")
    print(f"  Training accuracy: {train_accuracy:.2%}")
    print(f"  Test accuracy: {test_accuracy:.2%}")
    print()
    print("*** Test accuracy is very poor due to the normalization bug! ***")
    print()


if __name__ == "__main__":
    main()
