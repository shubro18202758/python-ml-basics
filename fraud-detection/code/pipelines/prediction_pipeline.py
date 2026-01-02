"""Prediction Pipeline - Real-time Fraud Scoring & Detection

Implements the complete inference pipeline for scoring transactions in production.
Includes model loading, feature transformation, and fraud probability calculation.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple, List
from datetime import datetime


class PredictionPipeline:
    """Real-time fraud detection scoring pipeline"""
    
    def __init__(self, model_path: str, feature_scaler_path: str):
        """Initialize prediction pipeline with trained model
        
        Parameters:
        - model_path: Path to pickled trained fraud detection model
        - feature_scaler_path: Path to fitted feature scaler
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(feature_scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.prediction_threshold = 0.5
    
    def preprocess_transaction(self, transaction: Dict) -> pd.DataFrame:
        """Convert transaction dict to feature dataframe
        
        Parameters:
        - transaction: Transaction data dictionary
        
        Returns:
        - Preprocessed DataFrame ready for model input
        """
        # Create single-row dataframe
        df = pd.DataFrame([transaction])
        
        # Feature engineering
        df['transaction_hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['transaction_day'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['transaction_day'].isin([5, 6]).astype(int)
        
        # Drop non-numeric columns
        df = df.select_dtypes(include=[np.number])
        
        return df
    
    def score_transaction(self, transaction: Dict) -> float:
        """Score a single transaction for fraud probability
        
        Parameters:
        - transaction: Transaction data dictionary
        
        Returns:
        - Fraud probability score (0-1)
        """
        # Preprocess
        X = self.preprocess_transaction(transaction)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction probability
        fraud_probability = self.model.predict_proba(X_scaled)[0, 1]
        
        return float(fraud_probability)
    
    def predict_batch(self, transactions: List[Dict]) -> pd.DataFrame:
        """Score multiple transactions efficiently
        
        Parameters:
        - transactions: List of transaction dictionaries
        
        Returns:
        - DataFrame with original data + fraud_score and fraud_flag columns
        """
        results = []
        
        for transaction in transactions:
            X = self.preprocess_transaction(transaction)
            X_scaled = self.scaler.transform(X)
            
            fraud_prob = self.model.predict_proba(X_scaled)[0, 1]
            fraud_flag = 1 if fraud_prob >= self.prediction_threshold else 0
            
            result = transaction.copy()
            result['fraud_score'] = fraud_prob
            result['fraud_flag'] = fraud_flag
            result['prediction_timestamp'] = datetime.now().isoformat()
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def set_threshold(self, threshold: float):
        """Adjust fraud detection threshold
        
        Parameters:
        - threshold: New classification threshold (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold must be between 0 and 1')
        self.prediction_threshold = threshold


if __name__ == '__main__':
    print('Prediction Pipeline for Real-time Fraud Detection')
    print('Provides transaction scoring and fraud probability estimation')
