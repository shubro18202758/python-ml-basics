"""Bayesian Inference - Probabilistic Fraud Detection

Implements Bayesian methods for fraud detection including prior probabilities,
likelihood estimation, and posterior probability calculation using Bayes' theorem.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy import stats


class BayesianFraudDetector:
    """Bayesian probabilistic fraud detection model"""
    
    def __init__(self, fraud_prior: float = 0.05):
        """Initialize Bayesian fraud detector
        
        Parameters:
        - fraud_prior: Prior probability of fraud (P(Fraud))
        """
        self.fraud_prior = fraud_prior
        self.legitimate_prior = 1 - fraud_prior
        self.fraud_likelihood = None
        self.legitimate_likelihood = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Bayesian model to training data
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - y: Binary labels (fraud=1, legitimate=0)
        """
        fraud_data = X[y == 1]
        legitimate_data = X[y == 0]
        
        # Estimate feature distributions for each class
        self.fraud_means = fraud_data.mean(axis=0)
        self.fraud_stds = fraud_data.std(axis=0) + 1e-6  # Avoid division by zero
        
        self.legitimate_means = legitimate_data.mean(axis=0)
        self.legitimate_stds = legitimate_data.std(axis=0) + 1e-6
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior probability of fraud using Bayes' theorem
        
        P(Fraud|X) = P(X|Fraud) * P(Fraud) / P(X)
        
        Returns:
        - Array of fraud probabilities for each sample
        """
        n_samples = X.shape[0]
        probabilities = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Calculate likelihood P(X|Fraud) using Gaussian assumption
            fraud_likelihood = np.prod(
                stats.norm.pdf(X[i], self.fraud_means, self.fraud_stds)
            )
            
            # Calculate likelihood P(X|Legitimate)
            legitimate_likelihood = np.prod(
                stats.norm.pdf(X[i], self.legitimate_means, self.legitimate_stds)
            )
            
            # Apply Bayes' theorem
            fraud_posterior = (
                fraud_likelihood * self.fraud_prior /
                (fraud_likelihood * self.fraud_prior + 
                 legitimate_likelihood * self.legitimate_prior + 1e-10)
            )
            
            probabilities[i] = fraud_posterior
        
        return probabilities
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict fraud labels based on posterior probability
        
        Parameters:
        - X: Feature matrix
        - threshold: Decision threshold
        
        Returns:
        - Binary predictions (1=fraud, 0=legitimate)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


if __name__ == '__main__':
    print('Bayesian Fraud Detection using probabilistic inference')
    print('Based on Bayes theorem for posterior probability estimation')
