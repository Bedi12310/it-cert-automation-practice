# Before and After: Code Comparison

## 🔴 BEFORE (Buggy Code - ~10% Accuracy)

```python
import numpy as np
from sklearn.datasets import fetch_openml
from logistic_regression_classifier import LogisticRegressionClassifier

mnist = fetch_openml('mnist_784', version=1)
X_mnist = np.array(mnist.data)
y_mnist = np.array(mnist.target, dtype=int)

indices = np.arange(len(X_mnist))
np.random.shuffle(indices)

X_mnist = X_mnist[indices]
y_mnist = y_mnist[indices]

split_ratio = 0.7
split_index = int(len(X_mnist) * split_ratio)

X_train = X_mnist[:split_index]
X_test  = X_mnist[split_index:]
y_train = y_mnist[:split_index]
y_test  = y_mnist[split_index:]

# ❌ BUG #1: Normalizing labels!
X_train = X_train / 255
y_train = y_train / 255  # ❌ WRONG! Labels should NOT be normalized

# ❌ BUG #2: Not normalizing test data!
# X_test is never normalized

model_mnist = LogisticRegressionClassifier(
    learning_rate=0.001,
    batch_size=1024,
    epochs=10
)
model_mnist.fit(X_train, y_train, verbose=True)
# Result: ~10% accuracy (random guessing)
```

### Buggy Softmax Function

```python
def _softmax(self, x):
    """Numerically stable softmax function."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)  # ❌ BUG #3: Not using x_max!
    out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return out
```

---

## 🟢 AFTER (Fixed Code - ~85-92% Accuracy)

```python
import numpy as np
from sklearn.datasets import fetch_openml
from logistic_regression_classifier import LogisticRegressionClassifier

mnist = fetch_openml('mnist_784', version=1)
X_mnist = np.array(mnist.data)
y_mnist = np.array(mnist.target, dtype=int)

indices = np.arange(len(X_mnist))
np.random.shuffle(indices)

X_mnist = X_mnist[indices]
y_mnist = y_mnist[indices]

split_ratio = 0.7
split_index = int(len(X_mnist) * split_ratio)

X_train = X_mnist[:split_index]
X_test  = X_mnist[split_index:]
y_train = y_mnist[:split_index]
y_test  = y_mnist[split_index:]

# ✅ FIX #1 & #2: Only normalize X (features), not y (labels)
X_train = X_train / 255.0  # Normalize training features
X_test  = X_test / 255.0   # ✅ FIXED: Normalize test features too!
# y_train and y_test remain as integers (0-9)

model_mnist = LogisticRegressionClassifier(
    learning_rate=0.001,
    batch_size=1024,
    epochs=10
)
model_mnist.fit(X_train, y_train, verbose=True)
# Result: ~85-92% accuracy!
```

### Fixed Softmax Function

```python
def _softmax(self, x):
    """Numerically stable softmax function."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)  # ✅ FIXED: Subtract x_max for stability
    out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return out
```

---

## 📊 Visual Comparison of Data Preprocessing

### 🔴 BEFORE (Wrong)

```
X_train: [0, 255] → [0.0, 1.0] ✓
y_train: [0, 9]   → [0.0, 0.035] ❌ WRONG!
X_test:  [0, 255] → [0, 255] ❌ NOT NORMALIZED!
y_test:  [0, 9]   → [0, 9] ✓
```

**Problem:**
- `y_train` becomes decimals instead of integers
- `X_test` has different scale than `X_train`
- Model can't learn properly

### 🟢 AFTER (Correct)

```
X_train: [0, 255] → [0.0, 1.0] ✓
y_train: [0, 9]   → [0, 9] ✓ INTEGERS!
X_test:  [0, 255] → [0.0, 1.0] ✓ NORMALIZED!
y_test:  [0, 9]   → [0, 9] ✓
```

**Solution:**
- `y_train` stays as integers for one-hot encoding
- `X_test` normalized same as `X_train`
- Model learns correctly

---

## 🎯 Key Differences Summary

| Aspect | Before (Buggy) | After (Fixed) |
|--------|---------------|---------------|
| **X_train normalization** | ✓ Correct | ✓ Correct |
| **y_train normalization** | ❌ Divided by 255 | ✅ Stays as int |
| **X_test normalization** | ❌ Not normalized | ✅ Divided by 255 |
| **y_test normalization** | ✓ Stays as int | ✓ Stays as int |
| **Softmax stability** | ❌ Doesn't use x_max | ✅ Uses x_max |
| **Training accuracy** | ~10% | ~90% |
| **Test accuracy** | ~10% | ~85-92% |

---

## 💡 Remember These Rules

### ✅ DO:
- Normalize **features** (X) by dividing by 255 (or max value)
- Keep **labels** (y) as integers (0, 1, 2, ..., K-1)
- Normalize training and test data the **same way**
- Use numerically stable implementations (subtract max before exp)

### ❌ DON'T:
- Don't normalize labels (they're categorical, not continuous)
- Don't forget to normalize test data
- Don't use different normalization for train vs test
- Don't ignore numerical stability (exp can overflow)

---

## 🚀 Quick Start with Fixed Code

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests to verify fixes
python test_classifier.py

# Train with fixed code
python mnist_train.py

# Compare buggy vs fixed
python compare_buggy_vs_fixed.py
```

---

**Bottom Line:** The main issue was normalizing the labels (y_train / 255). Labels are class indices and should **always remain as integers**. Only normalize the features (pixel values)!
