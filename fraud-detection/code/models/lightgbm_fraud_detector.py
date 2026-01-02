"""LightGBM Fraud Detection Model

Gradient boosting alternative to XGBoost with faster training.
Optimized for imbalanced classification with built-in features.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class LightGBMFraudDetector:
    """LightGBM-based fraud detection with class weight handling."""
    
    def __init__(self, scale_pos_weight: float = 99.0, n_estimators: int = 200,
                 learning_rate: float = 0.05, max_depth: int = 7):
        """
        Initialize LightGBM detector.
        
        Args:
            scale_pos_weight: Weight for positive class (fraud)
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
        """
        if lgb is None:
            raise ImportError("lightgbm not installed. Install with: pip install lightgbm")
        
        self.scale_pos_weight = scale_pos_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None
        self.feature_names = None
    
    def build_model(self):
        """Build LightGBM model with parameters."""
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=31,
            is_unbalance=True,
            scale_pos_weight=self.scale_pos_weight,
            random_state=42,
            verbose=-1
        )
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
        """
        if self.model is None:
            self.build_model()
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.model.fit(X, y, verbose=False)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict fraud.
        
        Args:
            X: Features
            threshold: Classification threshold
        
        Returns:
            Binary predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get fraud probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict_proba(X)[:, 1]
