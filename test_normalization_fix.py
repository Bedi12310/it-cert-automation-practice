#!/usr/bin/env python3
"""
Test script to demonstrate the normalization bug fix.
"""

import numpy as np

print("Demonstrating the bug fix for data normalization:")
print("=" * 60)

# Simulating what was happening in the buggy code
print("\n1. BUGGY CODE (Original):")
print("   X_train = X_train / 255")
print("   y_train = y_train / 255  # BUG: Normalizing labels!")
print("\n   Problem: Labels should be integers (0-9), not normalized floats")

# Example with sample data
X_train_sample = np.array([[200, 150], [100, 50]])
X_test_sample = np.array([[180, 160], [90, 40]])
y_train_sample = np.array([5, 3])
y_test_sample = np.array([7, 2])

print(f"\n   Original y_train: {y_train_sample}")
y_train_buggy = y_train_sample / 255
print(f"   After buggy normalization: {y_train_buggy}")
print(f"   → Labels corrupted! Should be integers, not {y_train_buggy}")

print("\n" + "=" * 60)
print("\n2. FIXED CODE:")
print("   X_train = X_train / 255")
print("   X_test = X_test / 255   # FIX: Normalize test features instead!")
print("\n   Solution: Keep labels as integers, normalize both train and test features")

X_train_fixed = X_train_sample / 255
X_test_fixed = X_test_sample / 255
y_train_fixed = y_train_sample  # Keep as integers
y_test_fixed = y_test_sample    # Keep as integers

print(f"\n   X_train normalized: {X_train_fixed[0][:5]}... (scaled to [0,1])")
print(f"   X_test normalized: {X_test_fixed[0][:5]}... (scaled to [0,1])")
print(f"   y_train: {y_train_fixed} (integers preserved)")
print(f"   y_test: {y_test_fixed} (integers preserved)")

print("\n" + "=" * 60)
print("\n3. WHY THIS FIXES ACCURACY:")
print("   - Labels must be integers for classification (0-9 for MNIST)")
print("   - Test data must be normalized same way as training data")
print("   - The model expects consistent feature scaling between train/test")
print("   - Corrupted labels would cause incorrect gradient calculations")
print("\n" + "=" * 60)
