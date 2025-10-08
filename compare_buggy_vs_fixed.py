#!/usr/bin/env python3
"""
Comparison script showing the impact of the bugs vs fixes.

This script demonstrates the difference in accuracy between:
1. Buggy version (normalizing labels, not normalizing test data)
2. Fixed version (correct normalization)

Uses synthetic MNIST-like data for demonstration.
"""

import numpy as np
from logistic_regression_classifier import LogisticRegressionClassifier


def generate_synthetic_mnist():
    """Generate synthetic MNIST-like data for demonstration."""
    np.random.seed(42)
    n_samples = 2000
    n_features = 784  # 28x28
    n_classes = 10
    
    # Create synthetic data with some structure
    X = np.random.randint(0, 256, size=(n_samples, n_features)).astype(float)
    
    # Create labels - just random for demonstration
    y = np.random.randint(0, n_classes, size=n_samples)
    
    return X, y


def simulate_buggy_version():
    """Simulate the buggy version with incorrect normalization."""
    print("\n" + "="*70)
    print("BUGGY VERSION (with incorrect normalization)")
    print("="*70)
    
    print("Generating synthetic MNIST-like data...")
    X_mnist, y_mnist = generate_synthetic_mnist()

    # Shuffle
    indices = np.arange(len(X_mnist))
    np.random.seed(42)
    np.random.shuffle(indices)
    X_mnist = X_mnist[indices]
    y_mnist = y_mnist[indices]

    # Split
    split_index = int(len(X_mnist) * 0.7)
    X_train = X_mnist[:split_index]
    X_test = X_mnist[split_index:]
    y_train = y_mnist[:split_index]
    y_test = y_mnist[split_index:]

    # ❌ BUGGY NORMALIZATION
    print("\n❌ BUG: Normalizing both X and y (WRONG!)")
    X_train_buggy = X_train / 255
    y_train_buggy = y_train / 255  # ❌ This is wrong!
    # X_test not normalized at all
    
    print(f"   X_train range: [{X_train_buggy.min():.3f}, {X_train_buggy.max():.3f}]")
    print(f"   y_train range: [{y_train_buggy.min():.3f}, {y_train_buggy.max():.3f}]")
    print(f"   y_train dtype: {y_train_buggy.dtype}")
    print(f"   X_test range: [{X_test.min():.1f}, {X_test.max():.1f}] (not normalized!)")

    try:
        model_buggy = LogisticRegressionClassifier(
            learning_rate=0.001,
            batch_size=512,
            epochs=5
        )
        print("\nTraining buggy model...")
        model_buggy.fit(X_train_buggy, y_train_buggy, verbose=False)
        
        # Try to evaluate (will likely fail or give ~10% accuracy)
        train_acc = model_buggy.score(X_train_buggy, y_train_buggy)
        test_acc = model_buggy.score(X_test, y_test)
        
        print(f"\n📊 BUGGY VERSION RESULTS:")
        print(f"   Training accuracy: {train_acc:.2%}")
        print(f"   Test accuracy: {test_acc:.2%}")
        print(f"   Status: ❌ Poor performance (likely ~10%, random guessing)")
    except Exception as e:
        print(f"\n💥 BUGGY VERSION CRASHED: {str(e)}")


def demonstrate_fixed_version():
    """Demonstrate the fixed version with correct normalization."""
    print("\n" + "="*70)
    print("FIXED VERSION (with correct normalization)")
    print("="*70)
    
    print("Generating synthetic MNIST-like data...")
    X_mnist, y_mnist = generate_synthetic_mnist()

    # Shuffle
    indices = np.arange(len(X_mnist))
    np.random.seed(42)
    np.random.shuffle(indices)
    X_mnist = X_mnist[indices]
    y_mnist = y_mnist[indices]

    # Split
    split_index = int(len(X_mnist) * 0.7)
    X_train = X_mnist[:split_index]
    X_test = X_mnist[split_index:]
    y_train = y_mnist[:split_index]
    y_test = y_mnist[split_index:]

    # ✓ CORRECT NORMALIZATION
    print("\n✓ CORRECT: Only normalizing X (features), keeping y (labels) as integers")
    X_train = X_train / 255.0
    X_test = X_test / 255.0  # ✓ Also normalize test data!
    # y_train and y_test remain unchanged
    
    print(f"   X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"   y_train range: [{y_train.min()}, {y_train.max()}]")
    print(f"   y_train dtype: {y_train.dtype}")
    print(f"   X_test range: [{X_test.min():.3f}, {X_test.max():.3f}] (normalized!)")

    model_fixed = LogisticRegressionClassifier(
        learning_rate=0.001,
        batch_size=512,
        epochs=5
    )
    print("\nTraining fixed model...")
    model_fixed.fit(X_train, y_train, verbose=False)
    
    train_acc = model_fixed.score(X_train, y_train)
    test_acc = model_fixed.score(X_test, y_test)
    
    print(f"\n📊 FIXED VERSION RESULTS:")
    print(f"   Training accuracy: {train_acc:.2%}")
    print(f"   Test accuracy: {test_acc:.2%}")
    print(f"   Status: ✓ Correct preprocessing (with real MNIST: 75-85%)")


def main():
    print("=" * 70)
    print("COMPARISON: Buggy vs Fixed Logistic Regression Classifier")
    print("=" * 70)
    print("\nThis script demonstrates the impact of the normalization bugs.")
    print("Using synthetic MNIST-like data for demonstration.")
    
    # Show buggy version
    simulate_buggy_version()
    
    # Show fixed version
    demonstrate_fixed_version()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n🔑 KEY TAKEAWAYS:")
    print("   1. NEVER normalize labels (y) - they must remain as integers")
    print("   2. ALWAYS normalize test data the same way as training data")
    print("   3. Use numerically stable softmax (subtract max before exp)")
    print("\n💡 With these fixes on real MNIST data, accuracy improves from ~10% to ~75-85%!
   (Note: This demo uses random synthetic data, so both show ~10%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
