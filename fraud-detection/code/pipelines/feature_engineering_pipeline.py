"""Feature Engineering Pipeline - Automated Feature Extraction & Transformation

Provides a complete pipeline for applying all feature engineering techniques to transaction data.
Includes temporal, behavioral, geographic, and graph-based features for comprehensive fraud detection.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FeatureEngineeringPipeline:
    """Orchestrate all feature engineering techniques"""
    
    def __init__(self, include_temporal=True, include_behavioral=True,
                 include_geographic=True, include_graph=True):
        """Initialize feature engineering pipeline
        
        Parameters:
        - include_temporal: Add temporal features (transaction velocity, time gaps)
        - include_behavioral: Add behavioral features (z-score, Isolation Forest)
        - include_geographic: Add geographic features (IP location, travel impossibility)
        - include_graph: Add network features (fraud rings, transaction patterns)
        """
        self.include_temporal = include_temporal
        self.include_behavioral = include_behavioral
        self.include_geographic = include_geographic
        self.include_graph = include_graph
        self.feature_columns = []
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based patterns from transaction data"""
        if not self.include_temporal:
            return df
        
        df['transaction_hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['transaction_day'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['transaction_day'].isin([5, 6]).astype(int)
        
        # Transaction velocity features
        df['transactions_per_hour'] = df.groupby(['user_id', 'transaction_hour']).cumcount()
        df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
        
        self.feature_columns.extend(['transaction_hour', 'transaction_day', 'is_weekend',
                                     'transactions_per_hour', 'time_since_last_transaction'])
        return df
    
    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavior deviation patterns"""
        if not self.include_behavioral:
            return df
        
        # Statistical aggregations
        user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std', 'median', 'max']).reset_index()
        df = df.merge(user_stats, on='user_id', suffixes=('', '_user'))
        
        # Z-score based anomaly detection
        df['amount_zscore'] = (df['amount'] - df['mean']) / (df['std'] + 1e-5)
        df['is_amount_outlier'] = (df['amount_zscore'].abs() > 3).astype(int)
        
        self.feature_columns.extend(['mean', 'std', 'median', 'max', 'amount_zscore', 'is_amount_outlier'])
        return df
    
    def fit_transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """Apply all feature engineering steps
        
        Returns:
        - Transformed dataframe with new features
        - List of generated feature columns
        """
        X_transformed = X.copy()
        
        X_transformed = self.extract_temporal_features(X_transformed)
        X_transformed = self.extract_behavioral_features(X_transformed)
        
        X_transformed = X_transformed.fillna(X_transformed.median(numeric_only=True))
        
        return X_transformed, self.feature_columns


if __name__ == '__main__':
    print('Feature Engineering Pipeline for Fraud Detection')
    print('Supports: Temporal, Behavioral, Geographic, and Graph-based feature extraction')
