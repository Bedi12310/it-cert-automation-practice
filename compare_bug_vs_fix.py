#!/usr/bin/env python3
"""
Comparison script showing the impact of the bug vs the fix on a small dataset.
"""

import numpy as np

print("=" * 70)
print("COMPARISON: Buggy Code vs Fixed Code")
print("=" * 70)

# Simulate a small MNIST-like dataset
np.random.seed(42)
n_samples = 10

# Create sample data (simulating pixel values 0-255)
X_train = np.random.randint(0, 256, size=(n_samples, 784))
X_test = np.random.randint(0, 256, size=(n_samples, 784))
y_train = np.random.randint(0, 10, size=n_samples)
y_test = np.random.randint(0, 10, size=n_samples)

print(f"\nOriginal data shapes:")
print(f"  X_train: {X_train.shape}, values in [{X_train.min()}, {X_train.max()}]")
print(f"  X_test:  {X_test.shape}, values in [{X_test.min()}, {X_test.max()}]")
print(f"  y_train: {y_train.shape}, values: {y_train[:10]}")
print(f"  y_test:  {y_test.shape}, values: {y_test[:10]}")

print("\n" + "-" * 70)
print("BUGGY CODE:")
print("-" * 70)

# Buggy preprocessing
X_train_buggy = X_train / 255
y_train_buggy = y_train / 255  # ❌ BUG!
X_test_buggy = X_test  # Not normalized!
y_test_buggy = y_test

print(f"\nAfter buggy preprocessing:")
print(f"  X_train: values in [{X_train_buggy.min():.3f}, {X_train_buggy.max():.3f}] ✓ (normalized)")
print(f"  X_test:  values in [{X_test_buggy.min()}, {X_test_buggy.max()}] ❌ (NOT normalized!)")
print(f"  y_train: {y_train_buggy[:10]} ❌ (corrupted labels!)")
print(f"  y_test:  {y_test_buggy[:10]} ✓ (integers preserved)")

print("\n  Problems:")
print("  1. ❌ y_train labels corrupted (should be 0-9, not 0.0-0.035)")
print("  2. ❌ X_test not normalized (values 0-255 vs X_train 0-1)")
print("  3. ❌ Train/test distribution mismatch")
print("  4. ❌ Model will learn wrong patterns from corrupted labels")

print("\n" + "-" * 70)
print("FIXED CODE:")
print("-" * 70)

# Fixed preprocessing
X_train_fixed = X_train / 255.0
X_test_fixed = X_test / 255.0  # ✓ Now normalized!
y_train_fixed = y_train  # ✓ Keep as integers!
y_test_fixed = y_test  # ✓ Keep as integers!

print(f"\nAfter correct preprocessing:")
print(f"  X_train: values in [{X_train_fixed.min():.3f}, {X_train_fixed.max():.3f}] ✓ (normalized)")
print(f"  X_test:  values in [{X_test_fixed.min():.3f}, {X_test_fixed.max():.3f}] ✓ (normalized)")
print(f"  y_train: {y_train_fixed[:10]} ✓ (integers preserved)")
print(f"  y_test:  {y_test_fixed[:10]} ✓ (integers preserved)")

print("\n  Benefits:")
print("  1. ✓ All labels are correct integers (0-9)")
print("  2. ✓ Both X_train and X_test normalized to [0, 1]")
print("  3. ✓ Consistent data distribution between train/test")
print("  4. ✓ Model can learn correct class boundaries")
print("  5. ✓ Proper gradient calculations")

print("\n" + "=" * 70)
print("IMPACT ON MODEL TRAINING:")
print("=" * 70)

print("\nBuggy code issues:")
print("  • One-hot encoding fails with corrupted labels")
print("  • Gradient calculations are wrong")
print("  • Model sees different scales in train vs test")
print("  • Poor generalization and low accuracy")

print("\nFixed code benefits:")
print("  • Correct one-hot encoding")
print("  • Proper gradient descent")
print("  • Consistent feature scaling")
print("  • Improved accuracy and generalization")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("\nThe fix ensures:")
print("  1. Features (X) are normalized: divide by 255")
print("  2. Labels (y) stay as integers: no division")
print("  3. Train and test data are preprocessed identically")
print("\nThis will significantly improve your model's accuracy!")
print("=" * 70)
