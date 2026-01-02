"""Fraud Detection Evaluation Metrics

Comprehensive evaluation metrics for fraud detection models including:
- Precision, Recall, F1-Score
- ROC-AUC and PR-AUC curves
- Threshold optimization
- Business cost metrics
- Confusion matrices
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
import matplotlib.pyplot as plt


class FraudDetectionMetrics:
    """Evaluate fraud detection model performance with multiple metrics."""
    
    def __init__(self, cost_fp: float = 1.0, cost_fn: float = 100.0):
        """
        Initialize metrics calculator.
        
        Args:
            cost_fp: Cost of false positive (incorrect fraud alert)
            cost_fn: Cost of false negative (missed fraud)
        """
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """Calculate all evaluation metrics.
        
        Args:
            y_true: Ground truth labels (0/1)
            y_pred: Binary predictions (0/1)
            y_pred_proba: Probability predictions [0, 1]
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = np.mean(y_true == y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        
        # Business cost
        metrics['business_cost'] = (fp * self.cost_fp) + (fn * self.cost_fn)
        
        return metrics
    
    def find_optimal_threshold(self, y_true: np.ndarray, 
                              y_pred_proba: np.ndarray,
                              metric: str = 'f1') -> Tuple[float, float]:
        """Find optimal classification threshold.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Probability predictions
            metric: Metric to optimize ('f1', 'roc_auc', 'cost')
        
        Returns:
            (optimal_threshold, optimal_metric_value)
        """
        thresholds = np.arange(0, 1.01, 0.01)
        best_threshold = 0.5
        best_value = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                value = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'roc_auc':
                value = roc_auc_score(y_true, y_pred_proba)
            elif metric == 'cost':
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                value = -((fp * self.cost_fp) + (fn * self.cost_fn))  # Negative for minimization
            
            if value > best_value:
                best_value = value
                best_threshold = threshold
        
        return best_threshold, best_value if metric != 'cost' else -best_value
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


if __name__ == '__main__':
    # Example usage
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred_proba = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.3, 0.85, 0.88, 0.15, 0.25])
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    evaluator = FraudDetectionMetrics(cost_fp=1.0, cost_fn=100.0)
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
    
    print('Fraud Detection Metrics:')
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f'{metric:20s}: {value:.4f}')
        else:
            print(f'{metric:20s}: {value}')
    
    optimal_threshold, optimal_f1 = evaluator.find_optimal_threshold(
        y_true, y_pred_proba, metric='f1'
    )
    print(f'\nOptimal Threshold: {optimal_threshold:.2f}')
    print(f'F1-Score at Optimal: {optimal_f1:.4f}')
