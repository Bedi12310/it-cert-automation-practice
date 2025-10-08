#!/usr/bin/env python3
"""
Synthetic Data Classification - Demonstrating Data Normalization Importance

This script demonstrates the impact of proper data normalization using synthetic data.
"""
import numpy as np
from logistic_regression import LogisticRegressionClassifier


def generate_synthetic_data(n_samples=1000, n_features=20, n_classes=3, scale=255.0, seed=42):
    """Generate synthetic classification data."""
    np.random.seed(seed)
    
    # Generate data for each class
    X_list = []
    y_list = []
    
    samples_per_class = n_samples // n_classes
    for class_idx in range(n_classes):
        # Generate features with different means for each class
        # Use scale-dependent means to make normalization important
        mean = np.ones(n_features) * (class_idx * scale / n_classes + scale / 6)
        cov = np.eye(n_features) * (scale / 10)
        X_class = np.random.multivariate_normal(mean, cov, samples_per_class)
        
        # Scale to [0, scale] range (simulating image pixel values)
        X_class = np.clip(X_class, 0, scale)
        
        X_list.append(X_class)
        y_list.append(np.ones(samples_per_class) * class_idx)
    
    X = np.vstack(X_list)
    y = np.hstack(y_list).astype(int)
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    return X, y


def demonstrate_problem():
    """Demonstrate the problem with unnormalized test data."""
    print("=" * 70)
    print("PROBLEM: Test data not normalized")
    print("=" * 70)
    print()
    
    # Generate training and test data
    X_train, y_train = generate_synthetic_data(n_samples=600, scale=255.0, seed=42)
    X_test, y_test = generate_synthetic_data(n_samples=300, scale=255.0, seed=123)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # BUG: Only normalize training data
    print("*** BUG: Normalizing ONLY training data ***")
    X_train_normalized = X_train / 255.0
    X_test_not_normalized = X_test  # NOT normalized!
    
    print(f"X_train range: [{X_train_normalized.min():.2f}, {X_train_normalized.max():.2f}]")
    print(f"X_test range: [{X_test_not_normalized.min():.2f}, {X_test_not_normalized.max():.2f}]")
    print()
    
    # Train model
    print("Training model...")
    model = LogisticRegressionClassifier(
        learning_rate=0.1,
        batch_size=32,
        epochs=10
    )
    model.fit(X_train_normalized, y_train, verbose=False)
    
    # Evaluate
    train_accuracy = model.score(X_train_normalized, y_train)
    test_accuracy = model.score(X_test_not_normalized, y_test)
    
    print()
    print("Results with BUG:")
    print(f"  Training accuracy: {train_accuracy:.2%}")
    print(f"  Test accuracy: {test_accuracy:.2%}")
    print()
    print("*** Test accuracy is poor due to normalization mismatch! ***")
    print()


def demonstrate_solution():
    """Demonstrate the solution with properly normalized data."""
    print("=" * 70)
    print("SOLUTION: Both train and test data normalized")
    print("=" * 70)
    print()
    
    # Generate training and test data
    X_train, y_train = generate_synthetic_data(n_samples=600, scale=255.0, seed=42)
    X_test, y_test = generate_synthetic_data(n_samples=300, scale=255.0, seed=123)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # FIX: Normalize BOTH training and test data
    print("*** FIX: Normalizing BOTH training and test data ***")
    X_train_normalized = X_train / 255.0
    X_test_normalized = X_test / 255.0  # This is the fix!
    
    print(f"X_train range: [{X_train_normalized.min():.2f}, {X_train_normalized.max():.2f}]")
    print(f"X_test range: [{X_test_normalized.min():.2f}, {X_test_normalized.max():.2f}]")
    print()
    
    # Train model
    print("Training model...")
    model = LogisticRegressionClassifier(
        learning_rate=0.1,
        batch_size=32,
        epochs=10
    )
    model.fit(X_train_normalized, y_train, verbose=True)
    
    # Evaluate
    train_accuracy = model.score(X_train_normalized, y_train)
    test_accuracy = model.score(X_test_normalized, y_test)
    
    print()
    print("=" * 70)
    print("Results with FIX:")
    print("=" * 70)
    print(f"  Training accuracy: {train_accuracy:.2%}")
    print(f"  Test accuracy: {test_accuracy:.2%}")
    print()
    print("*** Test accuracy is now much better! ***")
    print()


def main():
    print()
    print("Synthetic Data Classification Example")
    print("Demonstrating the importance of consistent data preprocessing")
    print()
    
    demonstrate_problem()
    print()
    print()
    demonstrate_solution()
    
    print()
    print("=" * 70)
    print("KEY TAKEAWAY:")
    print("=" * 70)
    print("Always preprocess test data the SAME WAY as training data!")
    print("Normalization must be applied consistently to avoid accuracy issues.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
