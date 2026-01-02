"""
Aggregation Features for Fraud Detection

This module creates user-level and time-window aggregations that capture
behavioral patterns and statistical summaries of transaction history.

Key Features:
- Rolling statistics (mean, std, median, min, max)
- User-level aggregates (total spend, transaction count)
- Decayed aggregates (recent transactions weighted higher)
- Cumulative statistics
- Percentile-based features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class AggregationFeatureEngineer:
    """Create aggregation features from transaction data."""
    
    def __init__(self, user_column='user_id', amount_column='amount', time_column='transaction_time'):
        """
        Initialize aggregation feature engineer.
        
        Parameters:
        -----------
        user_column : str
            User identifier column
        amount_column : str
            Transaction amount column
        time_column : str
            Datetime column
        """
        self.user_column = user_column
        self.amount_column = amount_column
        self.time_column = time_column
    
    def user_level_aggregates(self, df):
        """
        Calculate user-level aggregate statistics.
        
        Returns user's total spend, average transaction size, transaction frequency, etc.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe
            
        Returns:
        --------
        pd.DataFrame
            User aggregates with columns merged to original df
        """
        df = df.copy()
        
        # User aggregates
        user_agg = df.groupby(self.user_column)[self.amount_column].agg([
            ('total_spend', 'sum'),
            ('avg_transaction', 'mean'),
            ('std_transaction', 'std'),
            ('max_transaction', 'max'),
            ('min_transaction', 'min'),
            ('median_transaction', 'median'),
            ('transaction_count', 'count'),
            ('unique_merchants', 'nunique')  # Assuming merchant column exists
        ]).reset_index()
        
        # Merge back
        df = df.merge(user_agg, on=self.user_column, how='left')
        
        # Calculate transaction size ratio to user average
        df['amount_to_user_mean'] = df[self.amount_column] / df['avg_transaction']
        
        return df
    
    def rolling_aggregates(self, df, windows=[7, 30, 90]):
        """
        Calculate rolling window aggregates.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe (must be sorted by time)
        windows : list
            Window sizes in days
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with rolling features
        """
        df = df.copy().sort_values(self.time_column)
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        
        for window in windows:
            # Rolling sum
            df[f'spend_last_{window}d'] = df.groupby(self.user_column)[self.amount_column].rolling(
                f'{window}D', on=self.time_column
            ).sum().reset_index(0, drop=True)
            
            # Rolling count
            df[f'txn_count_last_{window}d'] = df.groupby(self.user_column)[self.amount_column].rolling(
                f'{window}D', on=self.time_column
            ).count().reset_index(0, drop=True)
            
            # Rolling mean
            df[f'avg_spend_last_{window}d'] = df.groupby(self.user_column)[self.amount_column].rolling(
                f'{window}D', on=self.time_column
            ).mean().reset_index(0, drop=True)
        
        return df
    
    def decayed_aggregates(self, df, decay_rate=0.95):
        """
        Calculate exponentially decayed aggregates.
        Recent transactions have higher weight.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe (sorted by time)
        decay_rate : float
            Decay factor (0-1)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with decayed features
        """
        df = df.copy().sort_values(self.time_column)
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        
        # Days since most recent transaction
        max_date = df[self.time_column].max()
        df['days_since'] = (max_date - df[self.time_column]).dt.total_seconds() / (24 * 3600)
        
        # Exponential weights
        df['decay_weight'] = decay_rate ** (df['days_since'] / 30)  # 30-day half-life
        
        # Decayed sum
        def decayed_sum(group):
            return (group[self.amount_column] * group['decay_weight']).sum()
        
        df['decayed_spend'] = df.groupby(self.user_column).apply(decayed_sum).reindex(df.index)
        
        df.drop(['days_since', 'decay_weight'], axis=1, inplace=True)
        return df
    
    def percentile_features(self, df, percentiles=[25, 50, 75, 95]):
        """
        Calculate percentile-based features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe
        percentiles : list
            Percentile values to compute
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with percentile features
        """
        df = df.copy()
        
        for p in percentiles:
            percentile_val = df.groupby(self.user_column)[self.amount_column].transform(
                lambda x: x.quantile(p / 100)
            )
            df[f'amount_p{p}'] = percentile_val
        
        # Deviation from percentiles
        df['amount_to_p50'] = df[self.amount_column] / df['amount_p50']
        df['amount_to_p95'] = df[self.amount_column] / df['amount_p95']
        
        return df


def example_aggregation_features():
    """Example usage of aggregation features."""
    
    # Create sample transaction data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    n_transactions = 1000
    
    df = pd.DataFrame({
        'user_id': np.random.choice(['user_1', 'user_2', 'user_3', 'user_4'], n_transactions),
        'transaction_time': np.random.choice(dates, n_transactions),
        'amount': np.random.gamma(shape=2, scale=500, size=n_transactions),
        'merchant': np.random.choice(['merchant_1', 'merchant_2', 'merchant_3'], n_transactions)
    }).sort_values('transaction_time')
    
    # Engineer features
    engineer = AggregationFeatureEngineer(
        user_column='user_id',
        amount_column='amount',
        time_column='transaction_time'
    )
    
    df = engineer.user_level_aggregates(df)
    df = engineer.rolling_aggregates(df, windows=[7, 30])
    df = engineer.decayed_aggregates(df)
    df = engineer.percentile_features(df)
    
    print("\nAggregation Features Sample:")
    feature_cols = [col for col in df.columns if col not in 
                   ['user_id', 'transaction_time', 'amount', 'merchant']]
    print(df[['user_id', 'amount'] + feature_cols[:10]].head(10))
    
    return df


if __name__ == '__main__':
    result_df = example_aggregation_features()
    print(f"\nTotal columns: {len(result_df.columns)}")
