"""
Temporal Feature Engineering for Fraud Detection

This module implements temporal features that capture transaction patterns
over time, which are critical indicators of fraudulent behavior.

Key Features:
- Transaction velocity (frequency per hour/day)
- Time gaps between consecutive transactions
- Peak transaction times
- Recurring transaction patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TemporalFeatureEngineer:
    """Extract temporal features from transaction data."""
    
    def __init__(self, time_column='transaction_time'):
        """
        Initialize the temporal feature engineer.
        
        Parameters:
        -----------
        time_column : str
            Name of the datetime column in the DataFrame
        """
        self.time_column = time_column
    
    def extract_temporal_features(self, df, user_id_column='user_id'):
        """
        Extract all temporal features from transaction data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe with datetime column
        user_id_column : str
            Column name for user identifiers
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with temporal features added
        """
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        
        # Group by user
        df = df.sort_values([user_id_column, self.time_column])
        
        # Calculate time-based features
        df['hour_of_day'] = df[self.time_column].dt.hour
        df['day_of_week'] = df[self.time_column].dt.dayofweek
        df['day_of_month'] = df[self.time_column].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
        
        # Calculate transaction velocity per user
        df['velocity_1h'] = df.groupby([user_id_column, 
                                         pd.Grouper(key=self.time_column, freq='H')])[self.time_column].transform('count')
        df['velocity_1d'] = df.groupby([user_id_column, 
                                         pd.Grouper(key=self.time_column, freq='D')])[self.time_column].transform('count')
        
        return df
    
    def calculate_time_gaps(self, df, user_id_column='user_id'):
        """
        Calculate time gaps between consecutive transactions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe
        user_id_column : str
            Column name for user identifiers
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with time gap features
        """
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df = df.sort_values([user_id_column, self.time_column])
        
        # Time gap in minutes from previous transaction
        df['time_gap_minutes'] = df.groupby(user_id_column)[self.time_column].diff().dt.total_seconds() / 60
        df['time_gap_hours'] = df['time_gap_minutes'] / 60
        
        # Unusual time gap (sudden change in transaction pattern)
        df['unusual_time_gap'] = (df.groupby(user_id_column)['time_gap_minutes'].transform(
            lambda x: (np.abs(x - x.rolling(3, center=True).mean()) > 2 * x.rolling(3, center=True).std()).astype(int)
        ))
        
        return df
    
    def detect_rush_hours(self, df, threshold=5, user_id_column='user_id'):
        """
        Detect sudden rush of transactions (potential fraud indicator).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe
        threshold : int
            Minimum number of transactions in 10-minute window
        user_id_column : str
            Column name for user identifiers
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with rush detection feature
        """
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        
        # Count transactions per user in 10-minute windows
        df['time_window'] = df.groupby(user_id_column)[self.time_column].transform(
            lambda x: x.dt.round('10min')
        )
        
        df['transactions_in_window'] = df.groupby([user_id_column, 'time_window']).cumcount()
        df['is_rush_hour'] = (df['transactions_in_window'] >= threshold).astype(int)
        
        return df


def example_temporal_features():
    """Example usage of temporal feature engineering."""
    
    # Create sample transaction data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'user_id': np.random.choice(['user_1', 'user_2', 'user_3'], 100),
        'transaction_time': np.random.choice(dates, 100),
        'amount': np.random.uniform(10, 5000, 100)
    })
    
    # Extract features
    engineer = TemporalFeatureEngineer(time_column='transaction_time')
    
    df_with_features = engineer.extract_temporal_features(df)
    df_with_gaps = engineer.calculate_time_gaps(df_with_features)
    df_final = engineer.detect_rush_hours(df_with_gaps)
    
    print("\nTemporal Features Sample:")
    print(df_final[['user_id', 'transaction_time', 'hour_of_day', 
                     'velocity_1h', 'is_weekend', 'is_night']].head(10))
    
    return df_final


if __name__ == '__main__':
    result_df = example_temporal_features()
    print("\nFeature extraction complete!")
    print(f"Shape: {result_df.shape}")
    print(f"\nColumns: {list(result_df.columns)}")
