#!/usr/bin/env python3
"""
Base class for classifiers.
"""
import numpy as np


class ClassifierBase:
    """
    Base class for all classifiers.
    
    This class provides the basic interface that all classifiers should implement.
    """
    
    def fit(self, X, y):
        """
        Fit the classifier to the training data.
        
        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Training labels.
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : np.ndarray
            Data to predict.
            
        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters
        ----------
        X : np.ndarray
            Data to predict.
        y : np.ndarray
            True labels.
            
        Returns
        -------
        float
            Accuracy score.
        """
        y_pred = self.predict(X)
        y_true = y.reshape(-1, 1) if y.ndim == 1 else y
        return np.mean(y_pred == y_true)
