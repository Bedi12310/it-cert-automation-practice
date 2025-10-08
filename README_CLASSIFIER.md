# Logistic Regression Classifier for MNIST

A from-scratch implementation of multi-class logistic regression using stochastic gradient descent for the MNIST handwritten digits dataset.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
python test_classifier.py
```

### Train on MNIST

```bash
python mnist_train.py
```

## The Problem: Low Accuracy Issue

If you're experiencing low accuracy (~10%) with your logistic regression classifier on MNIST, it's likely due to one or more of these common bugs:

### Bug #1: Normalizing Labels ❌

**WRONG:**
```python
X_train = X_train / 255
y_train = y_train / 255  # ❌ CRITICAL BUG!
```

**CORRECT:**
```python
X_train = X_train / 255.0
# Do NOT normalize y_train - labels must remain as integers!
```

**Why?** Labels represent class indices (0-9), not pixel values. Normalizing them breaks the one-hot encoding and makes training impossible.

### Bug #2: Not Normalizing Test Data ❌

**WRONG:**
```python
X_train = X_train / 255
X_test  = X_test          # ❌ Not normalized!
```

**CORRECT:**
```python
X_train = X_train / 255.0
X_test  = X_test / 255.0  # ✓ Must normalize test data too!
```

**Why?** The model was trained on normalized data, so test data must be normalized the same way.

### Bug #3: Softmax Numerical Instability ❌

**WRONG:**
```python
def _softmax(self, x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)  # ❌ Not using x_max!
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

**CORRECT:**
```python
def _softmax(self, x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)  # ✓ Subtract max for stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

**Why?** Without subtracting the max, large logit values cause numerical overflow.

## Expected Accuracy

With the fixes applied:
- **Training accuracy**: ~85-92% (10 epochs)
- **Test accuracy**: ~82-90%

## Complete Working Example

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
split_index = int(len(X_mnist) * 0.7)
X_train = X_mnist[:split_index]
X_test = X_mnist[split_index:]
y_train = y_mnist[:split_index]
y_test = y_mnist[split_index:]

# ✓ CORRECT NORMALIZATION
X_train = X_train / 255.0  # Normalize features
X_test = X_test / 255.0    # Normalize test features too
# y_train and y_test remain as integers!

# Train
model = LogisticRegressionClassifier(
    learning_rate=0.001,
    batch_size=1024,
    epochs=10
)
model.fit(X_train, y_train, verbose=True)

# Evaluate
test_accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2%}")
```

## Tips for Better Accuracy

1. **More epochs**: Increase from 10 to 20-50 epochs
2. **Learning rate tuning**: Try 0.0001 to 0.01
3. **Larger batch size**: Try 2048 or 4096
4. **More data**: Use the full MNIST dataset (70,000 samples)
5. **Better initialization**: Use Xavier or He initialization
6. **Regularization**: Add L2 penalty to prevent overfitting

## Architecture Details

- **Model**: Multi-class logistic regression (softmax regression)
- **Loss**: Negative log-likelihood (cross-entropy)
- **Optimization**: Stochastic gradient descent (SGD)
- **Classes**: 10 (digits 0-9)
- **Features**: 784 (28x28 pixels, flattened)

## Files

- `logistic_regression_classifier.py` - Main classifier implementation
- `mnist_train.py` - Training script with correct preprocessing
- `test_classifier.py` - Unit tests
- `IMPROVEMENTS.md` - Detailed explanation of fixes
- `requirements.txt` - Dependencies

## License

Apache 2.0 (same as parent repository)
