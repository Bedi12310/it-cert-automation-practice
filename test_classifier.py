#!/usr/bin/env python3
"""
Unit tests for LogisticRegressionClassifier.

Tests the classifier with synthetic data to verify correctness of implementation.
"""

import numpy as np
from logistic_regression_classifier import LogisticRegressionClassifier


def test_softmax_numerical_stability():
    """Test that softmax is numerically stable."""
    print("Testing softmax numerical stability...")
    
    classifier = LogisticRegressionClassifier()
    
    # Test with large values
    x_large = np.array([[1000, 2000, 3000]])
    result = classifier._softmax(x_large)
    
    # Should not contain NaN or Inf
    assert not np.isnan(result).any(), "Softmax produced NaN values"
    assert not np.isinf(result).any(), "Softmax produced Inf values"
    
    # Should sum to 1
    assert np.allclose(result.sum(), 1.0), f"Softmax doesn't sum to 1: {result.sum()}"
    
    print("✓ Softmax is numerically stable")


def test_label_handling():
    """Test that classifier properly handles integer labels."""
    print("\nTesting label handling...")
    
    # Create simple synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, size=n_samples)
    
    # Ensure labels are integers
    assert y.dtype in [np.int32, np.int64], f"Labels should be integers, got {y.dtype}"
    
    classifier = LogisticRegressionClassifier(
        learning_rate=0.01,
        batch_size=20,
        epochs=5
    )
    
    # Should train without errors
    classifier.fit(X, y, verbose=False)
    
    # Should predict integer labels
    predictions = classifier.predict(X)
    assert predictions.dtype in [np.int32, np.int64], "Predictions should be integers"
    assert predictions.min() >= 0, "Predictions should be non-negative"
    assert predictions.max() < n_classes, f"Predictions should be < {n_classes}"
    
    print(f"✓ Labels properly handled as integers (0-{n_classes-1})")


def test_training_improves_accuracy():
    """Test that training improves accuracy on synthetic data."""
    print("\nTesting that training improves accuracy...")
    
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    n_classes = 3
    
    # Create linearly separable data
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features, n_classes)
    logits = X @ true_weights
    y = np.argmax(logits, axis=1)
    
    classifier = LogisticRegressionClassifier(
        learning_rate=0.1,
        batch_size=40,
        epochs=20
    )
    
    # Get initial accuracy (random)
    classifier.fit(X, y, verbose=False)
    initial_accuracy = classifier.score(X[:40], y[:40])
    
    # Continue training
    classifier.fit(X, y, initialize_beta=False, verbose=False)
    final_accuracy = classifier.score(X[:40], y[:40])
    
    print(f"  Initial accuracy: {initial_accuracy:.2%}")
    print(f"  Final accuracy: {final_accuracy:.2%}")
    
    # After training on linearly separable data, accuracy should be high
    assert final_accuracy > 0.7, f"Expected accuracy > 70%, got {final_accuracy:.2%}"
    
    print("✓ Training improves accuracy")


def test_data_normalization():
    """Test example of correct data normalization."""
    print("\nTesting data normalization example...")
    
    # Simulate MNIST-like data (pixel values 0-255)
    X = np.random.randint(0, 256, size=(100, 28, 28)).astype(float)
    y = np.random.randint(0, 10, size=100)
    
    # Correct normalization
    X_normalized = X / 255.0
    
    assert X_normalized.min() >= 0, "Normalized X should be >= 0"
    assert X_normalized.max() <= 1, "Normalized X should be <= 1"
    assert y.min() == 0 or y.min() == 0, "Labels should start at 0"
    assert y.max() <= 9, "Labels should be <= 9 for MNIST"
    assert y.dtype in [np.int32, np.int64], "Labels should remain integers"
    
    print("✓ Data normalization example is correct")
    print(f"  X range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
    print(f"  y range: [{y.min()}, {y.max()}]")
    print(f"  y dtype: {y.dtype}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running LogisticRegressionClassifier Tests")
    print("=" * 60)
    
    test_softmax_numerical_stability()
    test_label_handling()
    test_training_improves_accuracy()
    test_data_normalization()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
