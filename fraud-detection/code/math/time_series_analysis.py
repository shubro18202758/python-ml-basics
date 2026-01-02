"""Time Series Analysis for Fraud Detection

Temporal pattern analysis for transaction data using trend detection,
autocorrelation, and seasonal decomposition.
"""

import numpy as np
from typing import Tuple


class TimeSeriesAnalysis:
    """Time series methods for fraud detection"""
    
    @staticmethod
    def detect_trend(data: np.ndarray, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Extract trend using moving average"""
        trend = np.convolve(data, np.ones(window)/window, mode='same')
        return trend, data - trend
    
    @staticmethod
    def autocorrelation(data: np.ndarray, max_lag: int = 20) -> np.ndarray:
        """Calculate autocorrelation for pattern detection"""
        data_norm = (data - np.mean(data)) / (np.std(data) + 1e-10)
        acf = np.correlate(data_norm, data_norm, mode='full')
        acf = acf[len(acf)//2:] / acf[len(acf)//2]
        return acf[:max_lag]
    
    @staticmethod
    def detect_changepoint(data: np.ndarray) -> int:
        """Detect significant shifts in transaction behavior"""
        data_norm = (data - np.mean(data)) / (np.std(data) + 1e-10)
        cumsum = np.abs(np.cumsum(data_norm))
        return int(np.argmax(cumsum))


if __name__ == '__main__':
    print('Time Series Analysis for Fraud Detection')
