"""Probability Distributions for Fraud Modeling

Implements Poisson, Exponential, and Gaussian distributions for
modeling transaction and fraud occurrence patterns.
"""

import numpy as np
from scipy import stats
from typing import Tuple


class ProbabilityDistributions:
    """Statistical distributions for fraud detection modeling"""
    
    @staticmethod
    def poisson_pmf(data: np.ndarray, lambda_param: float) -> np.ndarray:
        """Poisson distribution for transaction count modeling"""
        return stats.poisson.pmf(data, lambda_param)
    
    @staticmethod
    def exponential_pdf(x: np.ndarray, scale: float) -> np.ndarray:
        """Exponential distribution for time-between-transactions"""
        return stats.expon.pdf(x, scale=scale)
    
    @staticmethod
    def gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Gaussian distribution for feature modeling"""
        return stats.norm.pdf(x, loc=mean, scale=std)
    
    @staticmethod
    def fit_poisson(data: np.ndarray) -> float:
        """Estimate Poisson lambda parameter from data"""
        return float(np.mean(data))
    
    @staticmethod
    def fit_gaussian(data: np.ndarray) -> Tuple[float, float]:
        """Estimate Gaussian parameters from data"""
        return float(np.mean(data)), float(np.std(data))


if __name__ == '__main__':
    print('Probability Distributions for Fraud Detection')
