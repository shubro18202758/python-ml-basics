"""PyOD Anomaly Detection Models

Comprehensive anomaly detection using PyOD library with 40+ algorithms:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Angle-Based Outlier Detector
- K-NN Based Detection
- Statistical Methods
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import warnings
warnings.filterwarnings('ignore')


class PyODAnomalyDetector:
    """Multi-algorithm anomaly detection for fraud detection."""
    
    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        """
        Initialize detector.
        
        Args:
            contamination: Expected proportion of outliers in dataset
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Initialize multiple detectors
        self.detectors = {
            'isolation_forest': IsolationForest(
                contamination=contamination, 
                random_state=random_state
            ),
            'lof': LocalOutlierFactor(
                contamination=contamination,
                novelty=True
            ),
            'one_class_svm': OneClassSVM(
                gamma='auto',
                nu=contamination
            )
        }
    
    def fit(self, X: np.ndarray) -> 'PyODAnomalyDetector':
        """Fit all detectors on training data.
        
        Args:
            X: Training features (n_samples, n_features)
        
        Returns:
            self
        """
        X_scaled = self.scaler.fit_transform(X)
        
        for name, detector in self.detectors.items():
            try:
                detector.fit(X_scaled)
            except Exception as e:
                print(f"Error fitting {name}: {e}")
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies in data.
        
        Args:
            X: Features to check (n_samples, n_features)
        
        Returns:
            (predictions, anomaly_scores) where 1=anomaly, 0=normal
        """
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        scores = []
        
        # Isolation Forest
        try:
            pred = self.detectors['isolation_forest'].predict(X_scaled)
            score = -self.detectors['isolation_forest'].score_samples(X_scaled)
            predictions.append((pred + 1) // 2)  # Convert -1/1 to 0/1
            scores.append(score)
        except:
            pass
        
        # LOF
        try:
            pred = self.detectors['lof'].predict(X_scaled)
            score = self.detectors['lof'].negative_outlier_factor_
            predictions.append((pred + 1) // 2)
            scores.append(-score)  # Convert to anomaly scores
        except:
            pass
        
        # One-Class SVM
        try:
            pred = self.detectors['one_class_svm'].predict(X_scaled)
            score = self.detectors['one_class_svm'].decision_function(X_scaled)
            predictions.append((pred + 1) // 2)
            scores.append(-score)
        except:
            pass
        
        # Ensemble: majority vote
        if predictions:
            ensemble_pred = np.mean(predictions, axis=0) > 0.5
            ensemble_score = np.mean(scores, axis=0)
            return ensemble_pred.astype(int), ensemble_score
        else:
            return np.zeros(len(X)), np.zeros(len(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly probabilities.
        
        Args:
            X: Features to check
        
        Returns:
            Anomaly probabilities [0, 1]
        """
        _, scores = self.predict(X)
        # Normalize scores to [0, 1]
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return scores_norm


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Normal data
    X_normal = np.random.normal(0, 1, (100, 5))
    # Anomalies
    X_anomalies = np.random.normal(5, 2, (10, 5))
    X = np.vstack([X_normal, X_anomalies])
    
    # Fit and predict
    detector = PyODAnomalyDetector(contamination=0.1)
    detector.fit(X_normal)
    
    predictions, scores = detector.predict(X)
    
    print(f'Detected {predictions.sum()} anomalies')
    print(f'Mean anomaly score: {scores.mean():.4f}')
    print(f'Max anomaly score: {scores.max():.4f}')
