# Machine Learning Examples - MNIST Logistic Regression

## Overview

This directory contains a logistic regression classifier implementation and examples demonstrating common pitfalls and improvements for MNIST digit classification.

## The Problem

The original code had a critical bug that caused poor test accuracy:

```python
# Training data is normalized
X_train = X_train / 255

# Test data is NOT normalized - BUG!
# X_test should also be normalized!
```

This causes a **feature distribution mismatch**:
- Training data: pixel values in range [0, 1]
- Test data: pixel values in range [0, 255]

The model learns weights for normalized features but receives unnormalized features during testing, resulting in poor accuracy.

## How to Improve Accuracy

### 1. **Fix Data Preprocessing (CRITICAL)**

The most important fix is to normalize test data the same way as training data:

```python
# Normalize both training and test data
X_train = X_train / 255.0
X_test = X_test / 255.0   # Don't forget this!
```

**Why this matters:** 
- Neural networks and linear models are sensitive to feature scales
- The model's weights are optimized for the scale of training data
- Mismatched scales between train/test lead to incorrect predictions

### 2. **Tune Hyperparameters**

Improve model performance by adjusting hyperparameters:

```python
model = LogisticRegressionClassifier(
    learning_rate=0.1,      # Increased from 0.0001 (100x larger)
    batch_size=128,         # Increased from 64
    epochs=20               # Increased from 3
)
```

**Guidelines:**
- **Learning rate**: Too small = slow convergence; too large = instability
- **Batch size**: Larger batches = more stable gradients but slower per-epoch
- **Epochs**: More epochs = better fit but risk of overfitting

### 3. **Additional Improvements** (For Further Study)

These improvements can be explored for even better results:

#### a) **Add L2 Regularization**
```python
# In _compute_gradient_loss method
grad = (X_flat_with_bias.T @ (p - y_onehot)) / X_flat.shape[0]
grad += (lambda_reg / X_flat.shape[0]) * self.beta  # L2 regularization
```

#### b) **Better Weight Initialization**
```python
# Xavier/He initialization instead of random
self.beta = np.random.randn(n_features + 1, self.K) * np.sqrt(2.0 / n_features)
```

#### c) **Learning Rate Decay**
```python
# Decrease learning rate over time
current_lr = self.learning_rate / (1 + decay_rate * epoch)
```

#### d) **Data Augmentation**
- Random rotations (±5 degrees)
- Small translations
- Elastic deformations

#### e) **Feature Engineering**
- Add polynomial features
- Extract edge features
- Use PCA for dimensionality reduction

#### f) **Use a Better Model**
- Multi-layer neural network
- Convolutional Neural Network (CNN)
- Ensemble methods

## Files

- `classifier_base.py` - Base class for all classifiers
- `logistic_regression.py` - Logistic regression implementation with SGD
- `mnist_example_problem.py` - Demonstrates the original bug
- `mnist_example_fixed.py` - Shows the fixed version with improvements
- `README.md` - This file

## Usage

### Install Dependencies

```bash
pip install numpy scikit-learn
```

### Run the Examples

**See the problem:**
```bash
cd ml_examples
python mnist_example_problem.py
```

**See the solution:**
```bash
python mnist_example_fixed.py
```

## Expected Results

### With Bug (Original Code)
- Training accuracy: ~85-90%
- Test accuracy: ~10% (random guessing!)

### With Fix
- Training accuracy: ~90-92%
- Test accuracy: ~85-88%

## Key Takeaways

1. **Always preprocess test data the same way as training data**
2. **Normalize features** to bring them to a similar scale
3. **Tune hyperparameters** to improve model performance
4. **Monitor both training and test accuracy** to detect issues
5. **Start simple, then add complexity** only if needed

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- Logistic Regression: https://en.wikipedia.org/wiki/Logistic_regression
- Stochastic Gradient Descent: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
