"""
Behavioral Deviation Detection for Fraud Detection

This module detects deviations from normal user behavior patterns using
statistical methods like z-scores, isolation forests, and distance metrics.

Key Concepts:
- Z-score based anomaly detection
- Isolation Forest for multivariate anomaly detection
- Mahalanobis distance for accounting feature correlations
- Local Outlier Factor (LOF) for density-based anomaly detection
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import mahalanobis


class BehavioralDeviationDetector:
    """Detect deviations from normal user behavior."""
    
    def __init__(self, z_score_threshold=3):
        """
        Initialize the behavioral deviation detector.
        
        Parameters:
        -----------
        z_score_threshold : float
            Number of standard deviations to flag as anomaly
        """
        self.z_score_threshold = z_score_threshold
        self.scaler = StandardScaler()
    
    def z_score_anomalies(self, df, features, group_by='user_id'):
        """
        Detect anomalies using z-score method.
        
        Formula: z = (x - mean) / std
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        features : list
            Feature columns to compute z-scores on
        group_by : str
            Column to group by (user-specific statistics)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with z-scores and anomaly flags
        """
        df = df.copy()
        
        for feature in features:
            # Calculate z-score per user
            z_scores = df.groupby(group_by)[feature].transform(
                lambda x: np.abs(stats.zscore(x, nan_policy='omit'))
            )
            df[f'{feature}_zscore'] = z_scores
            df[f'{feature}_anomaly'] = (z_scores > self.z_score_threshold).astype(int)
        
        # Overall anomaly flag
        anomaly_cols = [col for col in df.columns if col.endswith('_anomaly')]
        df['is_anomaly_zscore'] = (df[anomaly_cols].sum(axis=1) > 0).astype(int)
        
        return df
    
    def isolation_forest_anomalies(self, df, features, contamination=0.05):
        """
        Detect anomalies using Isolation Forest (multivariate approach).
        
        Isolation Forest works by randomly selecting features and split values,
        isolating anomalies which require fewer splits to separate.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        features : list
            Feature columns for anomaly detection
        contamination : float
            Expected proportion of anomalies in the dataset
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with anomaly predictions and scores
        """
        df = df.copy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(df[features].fillna(0))
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Predict (-1 for anomalies, 1 for normal)
        predictions = iso_forest.fit_predict(X_scaled)
        df['is_anomaly_iforest'] = (predictions == -1).astype(int)
        df['anomaly_score_iforest'] = iso_forest.score_samples(X_scaled)
        
        return df
    
    def mahalanobis_distance_anomalies(self, df, features, threshold_percentile=95):
        """
        Detect anomalies using Mahalanobis distance.
        
        Accounts for correlation between features.
        Formula: MD = sqrt((x - mean).T * S^-1 * (x - mean))
        where S is the covariance matrix
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        features : list
            Feature columns
        threshold_percentile : float
            Percentile threshold for anomaly detection
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with Mahalanobis distances
        """
        df = df.copy()
        X = df[features].fillna(df[features].mean()).values
        
        # Calculate mean and covariance
        mean = np.mean(X, axis=0)
        cov = np.cov(X.T)
        
        # Handle singular covariance matrix
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        
        # Calculate Mahalanobis distance for each sample
        md = []
        for i in range(len(X)):
            diff = X[i] - mean
            distance = np.sqrt(diff.dot(inv_cov).dot(diff.T))
            md.append(distance)
        
        df['mahalanobis_distance'] = md
        threshold = np.percentile(md, threshold_percentile)
        df['is_anomaly_mahal'] = (df['mahalanobis_distance'] > threshold).astype(int)
        
        return df
    
    def local_outlier_factor(self, df, features, n_neighbors=20):
        """
        Detect anomalies using Local Outlier Factor (density-based).
        
        LOF measures the local density deviation of a point compared to
        its neighbors. Points with significantly lower density are anomalies.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        features : list
            Feature columns
        n_neighbors : int
            Number of neighbors to consider
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with LOF scores
        """
        df = df.copy()
        X_scaled = self.scaler.fit_transform(df[features].fillna(0))
        
        # Fit LOF
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
        predictions = lof.fit_predict(X_scaled)
        
        df['is_anomaly_lof'] = (predictions == -1).astype(int)
        df['lof_score'] = lof.negative_outlier_factor_
        
        return df


def example_behavioral_deviation():
    """Example usage of behavioral deviation detection."""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 200
    df = pd.DataFrame({
        'user_id': np.repeat(['user_1', 'user_2', 'user_3'], n_samples // 3),
        'transaction_amount': np.random.gamma(shape=2, scale=500, size=n_samples),
        'merchant_diversity': np.random.poisson(lam=5, size=n_samples),
        'time_of_day': np.random.uniform(0, 24, n_samples),
    })
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, 20, replace=False)
    df.loc[anomaly_indices, 'transaction_amount'] = np.random.uniform(50000, 100000, 20)
    
    # Detect anomalies
    detector = BehavioralDeviationDetector(z_score_threshold=2.5)
    
    df = detector.z_score_anomalies(df, ['transaction_amount', 'merchant_diversity'])
    df = detector.isolation_forest_anomalies(df, ['transaction_amount', 'merchant_diversity'])
    df = detector.mahalanobis_distance_anomalies(df, ['transaction_amount', 'merchant_diversity'])
    df = detector.local_outlier_factor(df, ['transaction_amount', 'merchant_diversity'])
    
    print("\nBehavioral Deviation Detection Results:")
    anomaly_summary = df[['is_anomaly_zscore', 'is_anomaly_iforest', 'is_anomaly_mahal', 'is_anomaly_lof']].sum()
    print(f"Anomalies detected:\n{anomaly_summary}")
    
    return df


if __name__ == '__main__':
    result_df = example_behavioral_deviation()
    print(f"\nProcessed {len(result_df)} transactions")
