#!/usr/bin/env python3
"""
MNIST Training Script for Logistic Regression Classifier.

This script demonstrates how to train the LogisticRegressionClassifier on the MNIST dataset
with proper data preprocessing to achieve good accuracy.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from logistic_regression_classifier import LogisticRegressionClassifier


def main():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    X_mnist = np.array(mnist.data)
    y_mnist = np.array(mnist.target, dtype=int)  # convert to int

    print(f"Dataset shape: X={X_mnist.shape}, y={y_mnist.shape}")

    # Shuffle the data
    indices = np.arange(len(X_mnist))
    np.random.shuffle(indices)

    X_mnist = X_mnist[indices]
    y_mnist = y_mnist[indices]

    # Split into train and test sets
    split_ratio = 0.7
    split_index = int(len(X_mnist) * split_ratio)

    X_train = X_mnist[:split_index]
    X_test = X_mnist[split_index:]
    y_train = y_mnist[:split_index]
    y_test = y_mnist[split_index:]

    # IMPORTANT FIX: Only normalize X (features), NOT y (labels)!
    # Labels should remain as integers (0-9)
    print("\nNormalizing features...")
    X_train = X_train / 255.0
    X_test = X_test / 255.0  # FIX: Also normalize test data!

    print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"Label range: {y_train.min()} to {y_train.max()}")

    # Train the model
    print("\nTraining model...")
    model_mnist = LogisticRegressionClassifier(
        learning_rate=0.001,
        batch_size=1024,
        epochs=10
    )
    model_mnist.fit(X_train, y_train, verbose=True)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = model_mnist.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.2%}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
