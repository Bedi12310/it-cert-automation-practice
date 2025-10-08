# Logistic Regression Classifier - Accuracy Improvements

## Issues Found and Fixed

### Critical Bug #1: Normalizing Labels (MAJOR ISSUE)
**Problem:**
```python
y_train = y_train / 255  # WRONG!
```

This is a **critical bug**! Labels should NEVER be normalized. The labels represent class indices (0-9 for MNIST digits), not pixel values. Dividing by 255 turns them into small decimal values (0.0, 0.004, 0.008, etc.), which breaks the one-hot encoding and gradient computation.

**Fix:**
```python
# Only normalize features (X), NOT labels (y)
X_train = X_train / 255.0
X_test = X_test / 255.0
# y_train and y_test remain as integers
```

### Bug #2: Not Normalizing Test Data
**Problem:**
```python
X_train = X_train / 255
# X_test was not normalized!
```

The test data must be normalized the same way as training data. Without this, the model will perform poorly on test data because the feature scales don't match.

**Fix:**
```python
X_train = X_train / 255.0
X_test = X_test / 255.0  # Added this line
```

### Bug #3: Softmax Numerical Stability
**Problem:**
```python
def _softmax(self, x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)  # Not using x_max!
    out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return out
```

The code computes `x_max` but doesn't use it, defeating the purpose of numerical stability.

**Fix:**
```python
def _softmax(self, x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)  # Subtract x_max for stability
    out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return out
```

## Expected Accuracy Improvement

With these fixes, you should expect:
- **Before fixes:** ~10% accuracy (essentially random guessing)
- **After fixes:** ~85-92% accuracy on MNIST with the given hyperparameters

## Additional Tips for Better Accuracy

If you want to improve accuracy further, consider:

1. **Increase epochs:** Try 20-50 epochs instead of 10
2. **Tune learning rate:** Try values between 0.0001 and 0.01
3. **Add regularization:** Implement L2 regularization to prevent overfitting
4. **Feature engineering:** Add polynomial features or use PCA
5. **Better initialization:** Use Xavier/He initialization instead of random * 0.01
6. **Learning rate schedule:** Decrease learning rate as training progresses
7. **Use more training data:** Use full MNIST dataset instead of 70% split

## Usage

```python
from logistic_regression_classifier import LogisticRegressionClassifier
import numpy as np

# Load and preprocess data (see mnist_train.py for complete example)
X_train = X_train / 255.0  # Normalize features only!
X_test = X_test / 255.0    # Normalize test data too!
# Keep y_train and y_test as integers!

# Train model
model = LogisticRegressionClassifier(
    learning_rate=0.001,
    batch_size=1024,
    epochs=10
)
model.fit(X_train, y_train, verbose=True)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")
```

## Summary

The main issue preventing good accuracy was **normalizing the labels**. Labels are categorical class indices and should remain as integers (0, 1, 2, ..., 9), not be converted to decimals. This single fix should improve your accuracy from ~10% to ~85-92%.
