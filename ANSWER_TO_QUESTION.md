# How to Improve Your Logistic Regression Classifier Accuracy

## TL;DR - The Main Problem

Your code has **3 critical bugs** that are causing ~10% accuracy (random guessing):

```python
# ❌ WRONG CODE (from your problem statement)
X_train = X_train / 255
y_train = y_train / 255  # ❌ BUG #1: Don't normalize labels!

model_mnist.fit(X_train, y_train, verbose=True)

# X_test was never normalized! ❌ BUG #2
```

## ✅ The Solution

```python
# ✓ CORRECT CODE
X_train = X_train / 255.0  # Normalize features
X_test = X_test / 255.0    # ✓ FIX #1: Normalize test data too!
# y_train and y_test stay as integers - DON'T normalize them!

model_mnist.fit(X_train, y_train, verbose=True)
```

## 🐛 Bug #1: Normalizing Labels (CRITICAL!)

### What You Did Wrong:
```python
y_train = y_train / 255  # ❌ WRONG!
```

### Why It's Wrong:
- Labels represent class indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Dividing by 255 makes them: 0.0, 0.004, 0.008, 0.012, 0.016, 0.020, 0.024, 0.027, 0.031, 0.035
- The one-hot encoding expects **integers**, not decimals
- This completely breaks the training process!

### The Fix:
```python
# Only normalize X (pixel values 0-255), NOT y (class labels 0-9)
X_train = X_train / 255.0
# y_train stays as integers!
```

## 🐛 Bug #2: Not Normalizing Test Data

### What You Did Wrong:
```python
X_train = X_train / 255
# X_test was not normalized ❌
```

### Why It's Wrong:
- Training data: values between 0.0 and 1.0
- Test data: values between 0 and 255
- The model learned on normalized data, but you're testing on unnormalized data
- This causes a huge distribution mismatch!

### The Fix:
```python
X_train = X_train / 255.0
X_test = X_test / 255.0  # ✓ Normalize test data the same way!
```

## 🐛 Bug #3: Softmax Numerical Instability

### What You Did Wrong:
```python
def _softmax(self, x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)  # ❌ Not using x_max!
    out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return out
```

You computed `x_max` but never used it! This defeats the purpose of numerical stability.

### The Fix:
```python
def _softmax(self, x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)  # ✓ Subtract x_max for stability
    out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return out
```

## 📊 Expected Accuracy Improvement

| Version | Training Accuracy | Test Accuracy |
|---------|------------------|---------------|
| **Before (buggy)** | ~10% | ~10% |
| **After (fixed)** | ~90% | ~85-92% |

**That's an 8x improvement!** 🎉

## ✅ Complete Fixed Code

Here's your complete training script with all fixes:

```python
import numpy as np
from sklearn.datasets import fetch_openml
from logistic_regression_classifier import LogisticRegressionClassifier

# Load MNIST
mnist = fetch_openml('mnist_784', version=1)
X_mnist = np.array(mnist.data)
y_mnist = np.array(mnist.target, dtype=int)

# Shuffle
indices = np.arange(len(X_mnist))
np.random.shuffle(indices)
X_mnist = X_mnist[indices]
y_mnist = y_mnist[indices]

# Split
split_ratio = 0.7
split_index = int(len(X_mnist) * split_ratio)

X_train = X_mnist[:split_index]
X_test = X_mnist[split_index:]
y_train = y_mnist[:split_index]
y_test = y_mnist[split_index:]

# ✓ CORRECT NORMALIZATION
X_train = X_train / 255.0  # Normalize X
X_test = X_test / 255.0    # Normalize X_test too!
# y_train and y_test remain as integers

# Train
model_mnist = LogisticRegressionClassifier(
    learning_rate=0.001,
    batch_size=1024,
    epochs=10
)
model_mnist.fit(X_train, y_train, verbose=True)

# Evaluate
test_accuracy = model_mnist.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2%}")
```

## 🚀 How to Use the Fixed Code

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run tests to verify:**
   ```bash
   python test_classifier.py
   ```

3. **Train on MNIST:**
   ```bash
   python mnist_train.py
   ```

## 💡 Additional Tips for Even Better Accuracy

If you want to push beyond 92% accuracy:

1. **More epochs**: Try 20-50 instead of 10
2. **Learning rate tuning**: Experiment with 0.0001 to 0.01
3. **Regularization**: Add L2 penalty (weight decay)
4. **Better initialization**: Use Xavier/He initialization
5. **Learning rate schedule**: Decay the learning rate over time
6. **More training data**: Use all 70,000 MNIST samples
7. **Data augmentation**: Rotate, shift, or scale images slightly

## 📁 Files Provided

- `logistic_regression_classifier.py` - Fixed classifier with all bugs resolved
- `mnist_train.py` - Training script with correct preprocessing
- `test_classifier.py` - Unit tests (run with `python test_classifier.py`)
- `IMPROVEMENTS.md` - Detailed technical explanation
- `README_CLASSIFIER.md` - Complete usage guide
- `compare_buggy_vs_fixed.py` - Side-by-side comparison demonstration

## 🎯 Summary

The **#1 reason** for low accuracy was normalizing the labels. Labels are categorical indices (0-9) and should **never** be normalized. Only normalize the features (pixel values).

With these three simple fixes, your accuracy will jump from ~10% to ~85-92%! 🎉

---

**Questions?** Check `README_CLASSIFIER.md` for more details or run `python test_classifier.py` to verify the fixes work.
