# Fraud Detection & Anomaly Detection Toolkit - Implementation Guide

## Overview

This is a comprehensive, production-ready fraud detection toolkit designed for students and professionals. It separates **documentation/narrative content** from **code implementations** for clean organization and easy learning.

## Repository Structure

```
fraud-detection/
├── README.md                          # Main overview and concepts
├── IMPLEMENTATION_GUIDE.md            # This file - structure & implementation details
│
├── 01-mathematical-foundations/       # Pure documentation (narrative only)
│   ├── README.md                      # Overview of mathematical concepts
│   ├── statistics-for-fraud-detection.md      # Statistical theory & formulas
│   └── linear-algebra-for-ml.md               # Linear algebra foundations
│
└── code/                              # All executable code modules
    ├── feature_engineering/
    │   ├── temporal_features.py               # ✅ Transaction velocity, time gaps
    │   ├── behavioral_deviation.py            # ✅ Z-score, Isolation Forest, Mahalanobis, LOF
    │   ├── aggregation_features.py            # Rolling statistics, user aggregates
    │   ├── geographic_features.py             # IP location, impossible travel detection
    │   └── graph_features.py                  # Fraud ring detection, network analysis
    │
    ├── models/
    │   ├── xgboost_fraud_detector.py          # ✅ Gradient boosting with imbalance handling
    │   ├── pyod_anomaly_models.py             # 40+ anomaly detection algorithms
    │   ├── lightgbm_fraud_detector.py         # Alternative gradient boosting
    │   ├── deep_learning_detector.py          # Neural network implementations
    │   ├── ensemble_models.py                 # Voting & stacking ensembles
    │   └── evaluation_metrics.py              # PR-AUC, F1, business cost metrics
    │
    ├── math/
    │   ├── bayesian_fraud_detector.py         # Bayesian inference implementations
    │   ├── statistical_tests.py               # Z-tests, chi-square, hypothesis testing
    │   ├── time_series_analysis.py            # ARIMA, anomaly in sequences
    │   ├── distance_metrics.py                # Mahalanobis, Euclidean, Cosine
    │   └── probability_distributions.py       # Poisson, exponential, Gaussian models
    │
    └── pipelines/
        ├── preprocessing_pipeline.py          # Data cleaning, normalization
        ├── feature_engineering_pipeline.py    # Automated feature extraction
        ├── model_training_pipeline.py         # Train/validation/test splits
        └── prediction_pipeline.py             # Real-time fraud scoring
```

## Created Files (Completed)

### ✅ Feature Engineering

1. **temporal_features.py** ✓
   - Extract time-based patterns
   - Transaction velocity per hour/day
   - Time gaps between transactions
   - Rush hour detection (sudden spike)
   - Example: Users normally transact once per day; 10 transactions in 5 minutes = anomaly

2. **behavioral_deviation.py** ✓
   - Z-score anomaly detection: $z = \frac{x - \mu}{\sigma}$
   - Isolation Forest: Random partitioning isolates anomalies
   - Mahalanobis distance: $MD = \sqrt{(x-\mu)^T S^{-1} (x-\mu)}$
   - Local Outlier Factor (LOF): Density-based approach
   - Combines multiple methods for robust detection

### ✅ Model Implementations

3. **xgboost_fraud_detector.py** ✓
   - Production-ready gradient boosting model
   - Handles class imbalance with `scale_pos_weight`
   - Formula: $scale\_pos\_weight = \frac{\text{normal samples}}{\text{fraud samples}}$
   - Feature importance tracking
   - Early stopping to prevent overfitting
   - Industry performance: 85-95% precision

## Files to Create (Template Below)

### Feature Engineering

- **aggregation_features.py**: Rolling statistics, user-level aggregates
- **geographic_features.py**: IP geolocation, distance between transactions
- **graph_features.py**: Network analysis, fraud ring detection

### Model Implementations

- **pyod_anomaly_models.py**: 40+ algorithms from PyOD library
- **lightgbm_fraud_detector.py**: Alternative to XGBoost
- **deep_learning_detector.py**: Neural networks with imbalance handling
- **ensemble_models.py**: Voting ensemble, stacking
- **evaluation_metrics.py**: PR-AUC, F1-score, business cost curves

### Mathematical Implementations

- **bayesian_fraud_detector.py**: Posterior probability updates
- **statistical_tests.py**: Z-tests, chi-square, Kolmogorov-Smirnov
- **time_series_analysis.py**: ARIMA, change point detection
- **distance_metrics.py**: Vectorized implementations
- **probability_distributions.py**: Poisson, exponential modeling

## Using the Code

### Basic Example: Temporal Features

```python
from code.feature_engineering.temporal_features import TemporalFeatureEngineer

# Create engineer
engineer = TemporalFeatureEngineer(time_column='transaction_time')

# Extract features from transaction data
df_with_features = engineer.extract_temporal_features(df)
df_with_gaps = engineer.calculate_time_gaps(df_with_features)
df_final = engineer.detect_rush_hours(df_with_gaps)

print(df_final[['user_id', 'velocity_1h', 'is_rush_hour', 'time_gap_minutes']])
```

### XGBoost Model Example

```python
from code.models.xgboost_fraud_detector import XGBoostFraudDetector
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Initialize with imbalance ratio
detector = XGBoostFraudDetector(scale_pos_weight=99)  # 99:1 normal:fraud
detector.build_model()
detector.train(X_train, y_train)

# Predict with custom threshold
fraud_proba = detector.predict_proba(X_test)
fraud_predictions = detector.predict(X_test, threshold=0.3)  # Lower threshold = more fraud alerts

# Evaluate
metrics = detector.evaluate(X_test, y_test)
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

### Behavioral Deviation Example

```python
from code.feature_engineering.behavioral_deviation import BehavioralDeviationDetector

detector = BehavioralDeviationDetector(z_score_threshold=2.5)

# Apply multiple detection methods
df = detector.z_score_anomalies(df, ['transaction_amount', 'merchant_count'])
df = detector.isolation_forest_anomalies(df, ['transaction_amount', 'merchant_count'])
df = detector.mahalanobis_distance_anomalies(df, ['transaction_amount', 'merchant_count'])
df = detector.local_outlier_factor(df, ['transaction_amount', 'merchant_count'])

# Ensemble: flag if multiple methods detect anomaly
df['is_fraud'] = (df[['is_anomaly_zscore', 'is_anomaly_iforest', 
                       'is_anomaly_mahal', 'is_anomaly_lof']].sum(axis=1) >= 2).astype(int)
```

## Documentation Structure

### Separation of Concerns

**README.md & Markdown files** (01-mathematical-foundations/):
- Conceptual explanations
- Mathematical formulas with proper LaTeX rendering
- Theory and intuition
- Links to implementations
- **NO code blocks** (except small pseudocode)

**Python files** (code/):
- Complete implementations
- Docstrings with parameter details
- Working examples at module level
- Runnable from command line

## Mathematical Concepts Reference

### Statistics

- **Z-Score Anomaly**: $|z| > 3 \Rightarrow$ anomaly (3-sigma rule)
- **Mahalanobis Distance**: Accounts for feature correlations
- **Probability**: Bayes' theorem for fraud classification

### Linear Algebra

- **Distance Metrics**: Euclidean, Manhattan, Cosine similarity
- **Matrix Operations**: Covariance matrices, eigendecomposition
- **SVD**: Dimensionality reduction

### Gradient Boosting

- **Objective Function**: $L = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$
- **Scale Pos Weight**: Loss weight adjustment for imbalanced data

## Industry Applications

- **Banking**: FICO Falcon, Kount, Feedzai
- **Payment Processing**: PayPal, Square fraud detection
- **E-commerce**: Shopify, Amazon fraud prevention
- **Real-time**: Kafka + ML model streaming
- **Latency**: <100ms per transaction
- **Throughput**: 10,000+ TPS

## Next Steps for Students

1. Read README.md for overview
2. Study 01-mathematical-foundations/ for theory
3. Run code examples from code/ directory
4. Modify hyperparameters and observe results
5. Combine multiple detection methods for ensemble
6. Deploy to production with monitoring

## References

- XGBoost: Chen & Guestrin (2016)
- Isolation Forest: Liu et al. (2008)
- Mahalanobis: Mahalanobis (1936)
- PyOD: Zhao et al. (2019)

---

**Created**: January 2026 | **Last Updated**: January 2, 2026
