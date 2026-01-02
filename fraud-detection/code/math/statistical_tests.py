"""Statistical Hypothesis Tests for Fraud Detection

Implements Z-tests, Chi-square tests, and Kolmogorov-Smirnov tests for
detecting statistical anomalies in transaction behavior.
"""

import numpy as np
from scipy import stats
from typing import Tuple


class StatisticalTests:
    """Suite of statistical tests for fraud anomaly detection"""
    
    @staticmethod
    def z_score_test(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Z-score based anomaly detection
        
        Detects values beyond threshold standard deviations from mean.
        
        Parameters:
        - data: 1D array of values
        - threshold: Number of standard deviations (default 3 = 99.7%)
        
        Returns:
        - Boolean array indicating anomalies
        """
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / (std + 1e-10))
        return z_scores > threshold
    
    @staticmethod
    def chi_square_test(observed: np.ndarray, expected: np.ndarray) -> Tuple[float, float]:
        """Chi-square test for categorical feature independence
        
        Tests if observed distribution differs significantly from expected.
        
        Parameters:
        - observed: Observed frequency distribution
        - expected: Expected frequency distribution
        
        Returns:
        - (chi2_statistic, p_value)
        """
        chi2_stat = np.sum((observed - expected) ** 2 / (expected + 1e-10))
        # Degrees of freedom = number of categories - 1
        df = len(observed) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        return chi2_stat, p_value
    
    @staticmethod
    def ks_test(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test for distribution comparison
        
        Compares two samples to detect distribution shifts in fraud patterns.
        
        Parameters:
        - sample1: First data sample (before fraud event)
        - sample2: Second data sample (after/during fraud event)
        
        Returns:
        - (ks_statistic, p_value)
        """
        ks_stat, p_value = stats.ks_2samp(sample1, sample2)
        return ks_stat, p_value
    
    @staticmethod
    def t_test(control: np.ndarray, treatment: np.ndarray) -> Tuple[float, float]:
        """Independent samples t-test
        
        Compares mean transaction amounts between groups.
        
        Returns:
        - (t_statistic, p_value)
        """
        t_stat, p_value = stats.ttest_ind(control, treatment)
        return t_stat, p_value


if __name__ == '__main__':
    print('Statistical Tests for Fraud Detection')
    print('Z-tests, Chi-square, KS-tests, and T-tests for anomaly detection')
