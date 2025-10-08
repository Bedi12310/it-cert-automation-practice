# How to Improve Your MNIST Classifier Accuracy

## 🔴 Critical Bug in Your Code

Your code has a **critical bug** that severely impacts test accuracy:

```python
# Training data is normalized
X_train = X_train / 255

model_mnist = LogisticRegressionClassifier(...)
model_mnist.fit(X_train, y_train, verbose=True)
```

**Problem:** You normalized `X_train` but forgot to normalize `X_test`! 

## 🔧 The Fix (Most Important!)

```python
# Normalize BOTH training and test data
X_train = X_train / 255.0
X_test = X_test / 255.0   # ← ADD THIS LINE!

model_mnist.fit(X_train, y_train, verbose=True)
```

### Why This Matters

Your model learns weights optimized for pixel values in range [0, 1], but when you test it with pixel values in range [0, 255], the predictions are completely wrong. This is like training a person to recognize centimeters but then asking them to work with inches without conversion!

**Expected Impact:**
- Without fix: Test accuracy ~10% (random guessing)
- With fix: Test accuracy ~85-92%

## ⚙️ Additional Improvements

After fixing the normalization bug, you can further improve accuracy:

### 1. Better Hyperparameters

Your current settings are too conservative:

```python
# Current (slow convergence)
model_mnist = LogisticRegressionClassifier(
    learning_rate=0.0001,  # Too small
    batch_size=64,
    epochs=50
)

# Better settings
model_mnist = LogisticRegressionClassifier(
    learning_rate=0.1,      # 1000x larger!
    batch_size=128,         # Larger batches
    epochs=20               # Fewer epochs needed
)
```

**Why:**
- **Learning rate 0.0001 is too small** - the model learns very slowly
- With proper normalization, you can use learning rate ~0.1 safely
- Larger batch size = more stable gradients
- With higher learning rate, you need fewer epochs

### 2. Standardization Instead of Normalization

For even better results, use standardization (zero mean, unit variance):

```python
# Calculate mean and std from training data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8  # Small constant to avoid division by zero

# Apply to both train and test
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std   # Use training mean/std!
```

**Important:** Always calculate statistics (mean, std) from training data only, then apply to both train and test.

### 3. Use More Training Data

```python
# Instead of using 70% of data
split_ratio = 0.7

# Use more data for training
split_ratio = 0.85  # 85% train, 15% test
```

More training data generally leads to better models.

### 4. Early Stopping

Stop training when validation accuracy stops improving:

```python
# Split training into train/validation
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

# Train with validation monitoring
best_val_accuracy = 0
patience = 3
patience_counter = 0

for epoch in range(max_epochs):
    model_mnist.fit(X_train, y_train, initialize_beta=False)
    val_accuracy = model_mnist.score(X_val, y_val)
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

### 5. Learning Rate Decay

Reduce learning rate over time:

```python
initial_lr = 0.1
for epoch in range(epochs):
    # Decay learning rate
    current_lr = initial_lr / (1 + 0.01 * epoch)
    model_mnist.learning_rate = current_lr
    # Train...
```

### 6. Add L2 Regularization

Prevent overfitting by penalizing large weights. You would need to modify the `_compute_gradient_loss` method:

```python
def _compute_gradient_loss(self, X, y, lambda_reg=0.01):
    X_flat = self._flatten(X)
    logits, X_flat_with_bias = self._compute_logits(X_flat)
    p = self._softmax(logits)
    y_onehot = self._one_hot(y)
    grad = (X_flat_with_bias.T @ (p - y_onehot)) / X_flat.shape[0]
    
    # Add L2 regularization (don't regularize bias term)
    grad[1:] += (lambda_reg / X_flat.shape[0]) * self.beta[1:]
    
    return grad
```

## 📊 Complete Example

Here's a complete, improved version:

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegressionClassifier

# Load data
mnist = fetch_openml('mnist_784', version=1)
X = np.array(mnist.data)
y = np.array(mnist.target, dtype=int)

# Split data (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Normalize BOTH train and test data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Or use standardization (better)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Train with better hyperparameters
model = LogisticRegressionClassifier(
    learning_rate=0.1,
    batch_size=128,
    epochs=20
)

model.fit(X_train, y_train, verbose=True)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\nTraining Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")
```

## 🎯 Expected Results

| Configuration | Train Acc | Test Acc |
|--------------|-----------|----------|
| Original (with bug) | ~85% | ~10% ❌ |
| Fixed normalization | ~90% | ~85% ✅ |
| + Better hyperparameters | ~92% | ~88% ✅ |
| + Standardization | ~93% | ~89% ✅ |

## 🚀 Advanced Techniques (Beyond This Implementation)

For even better results (>95% accuracy), consider:

1. **Use a Neural Network** with hidden layers
2. **CNN (Convolutional Neural Network)** - designed for images
3. **Data Augmentation**: rotate, shift, scale images
4. **Ensemble Methods**: combine multiple models
5. **Use a library**: scikit-learn, TensorFlow, PyTorch

## 📝 Summary

The most important fixes in order of impact:

1. ✅ **Normalize test data** - Critical! (+75% test accuracy)
2. ✅ **Increase learning rate** to 0.1 (+3-5% test accuracy)
3. ✅ **Use standardization** instead of normalization (+1-2%)
4. ✅ **More training data** (+1-2%)
5. ⚙️ **Add regularization** (helps prevent overfitting)
6. ⚙️ **Learning rate decay** (fine-tuning)

**Start with fixes 1-4, they'll give you the biggest improvements!**
