#!/usr/bin/env python3
"""
Logistic Regression Classifier for MNIST dataset.
This module implements a binary logistic regression model using stochastic gradient descent.
"""

import numpy as np
from sklearn.datasets import fetch_openml


class ClassifierBase:
    """Base class for classifiers."""
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters
        ----------
        X : np.ndarray
            Test data.
        y : np.ndarray
            True labels.
            
        Returns
        -------
        float
            Accuracy score.
        """
        y_pred = self.predict(X)
        y_flat = y.flatten() if len(y.shape) > 1 else y
        y_pred_flat = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
        return np.mean(y_pred_flat == y_flat)


class LogisticRegressionClassifier(ClassifierBase):
    """
    Binary logistic regression classifier using stochastic gradient descent.

    This class implements a binary logistic regression model for classification tasks,
    using negative log likelihood as the loss function and SGD for optimization.

    Parameters
    ----------
    learning_rate : float, default=0.0001
        Learning rate for gradient descent.
    batch_size : int, default=20
        Number of samples per batch for SGD.
    epochs : int, default=3
        Number of passes over the training data.
    """

    def __init__(self, learning_rate=0.0001, batch_size=20, epochs=3):
        """
        Initialize the classifier.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = None
        self.K = None

    def _flatten(self, X):
        """
        Flatten input images to 2D array.

        Returns
        -------
        np.ndarray
            Flattened data of shape (n_samples, n_features).
        """
        return X.reshape(X.shape[0], -1)
    
    def _batchify(self, X, y):
        """
        Split data into batches.

        Returns
        -------
        tuple
            Batched X and y.
        """
        n_batches = X.shape[0] // self.batch_size
        X = X[:n_batches*self.batch_size].reshape(n_batches, self.batch_size, *X.shape[1:])
        y = y[:n_batches*self.batch_size].reshape(n_batches, self.batch_size, 1)
        return X, y
    
    def _softmax(self, x):
        """
        Numerically stable softmax function.
        """
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return out
    

    def _compute_logits(self, X_flat):
        """
        Compute logits for input data.
        """
        X_flat = np.hstack((np.ones((X_flat.shape[0], 1)), X_flat))
        return X_flat @ self.beta, X_flat

    def _compute_gradient_loss(self, X, y):
        """
        Compute gradient of the loss function.
        """
        X_flat = self._flatten(X)
        logits, X_flat_with_bias = self._compute_logits(X_flat)
        p = self._softmax(logits)
        y_onehot = self._one_hot(y)
        grad = (X_flat_with_bias.T @ (p - y_onehot)) / X_flat.shape[0]
        return grad
    
    
    def _one_hot(self, y):
        """
        One-hot encode target labels.
        """
        y_flat = y.flatten().astype(int)
        y_onehot = np.zeros((y_flat.size, self.K))
        y_onehot[np.arange(y_flat.size), y_flat] = 1
        return y_onehot

    def fit(self, X, y, initialize_beta=True, verbose=False):
        """
        Fit the logistic regression model.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Training labels.
        initialize_beta : bool, default=True
            Whether to re-initialize model parameters.
        verbose : bool, default=False
            If True, print training accuracy after each epoch.
        """
        self.K = np.unique(y).shape[0]
        X_batched, y_batched = self._batchify(X, y)
        n_features = np.prod(X_batched.shape[2:]).item()
        if initialize_beta or self.beta is None:
            self.beta = np.random.randn(n_features + 1, self.K) * 0.01

        if verbose == True:
            print(f"Epoch {0}/{self.epochs}, Training accuracy: {self.score(X, y):.2%}")
        for epoch in range(self.epochs):
            for j, X_batch in enumerate(X_batched):
                y_batch = y_batched[j]
                gradient = self._compute_gradient_loss(X_batch, y_batch)
                self.beta -= self.learning_rate * gradient


            if verbose == True:
                print(f"Epoch {epoch+1}/{self.epochs}, Training accuracy: {self.score(X, y):.2%}")

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        logits, _ = self._compute_logits(self._flatten(X))
        p = self._softmax(logits)
        y_pred = np.argmax(p, axis=1)
        return y_pred.reshape(-1, 1)


def main():
    """Main function to train and evaluate the classifier on MNIST."""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    X_mnist = np.array(mnist.data)
    y_mnist = np.array(mnist.target, dtype=int)  # convert to int

    # Shuffle the data
    indices = np.arange(len(X_mnist))
    np.random.seed(42)  # Set seed for reproducibility
    np.random.shuffle(indices)

    X_mnist = X_mnist[indices]
    y_mnist = y_mnist[indices]

    # Split into train and test sets
    split_ratio = 0.7
    split_index = int(len(X_mnist) * split_ratio)

    X_train = X_mnist[:split_index]
    X_test = X_mnist[split_index:]
    y_train = y_mnist[:split_index]
    y_test = y_mnist[split_index:]

    # Normalize the features (divide by 255 to scale to [0, 1])
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    # Train the model
    print("\nTraining logistic regression classifier...")
    model_mnist = LogisticRegressionClassifier(
        learning_rate=0.001,
        batch_size=1024,
        epochs=10
    )
    model_mnist.fit(X_train, y_train, verbose=True)

    # Evaluate on test set
    test_accuracy = model_mnist.score(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.2%}")


if __name__ == "__main__":
    main()
