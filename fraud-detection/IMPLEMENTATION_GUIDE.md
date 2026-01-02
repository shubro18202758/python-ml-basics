# Comprehensive Implementation Guide for Fraud Detection Toolkit

Here's a detailed as well as exhaustive documentation of how to use the repo modules for building sample anomaly detection system:

***

# Fraud Detection \& Anomaly Detection Toolkit - Comprehensive Implementation Guide

## Table of Contents

1. [Overview \& Architecture](#overview--architecture)
2. [Repository Structure Deep Dive](#repository-structure-deep-dive)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Feature Engineering Modules](#feature-engineering-modules)
5. [Model Implementations](#model-implementations)
6. [Pipeline Architecture](#pipeline-architecture)
7. [End-to-End Implementation Workflow](#end-to-end-implementation-workflow)
8. [Advanced Usage Patterns](#advanced-usage-patterns)
9. [Production Deployment Guide](#production-deployment-guide)
10. [Performance Optimization](#performance-optimization)
11. [Troubleshooting \& Best Practices](#troubleshooting--best-practices)

***

## Overview \& Architecture

This fraud detection toolkit implements a **modular, production-ready framework** for detecting fraudulent transactions in financial systems. The architecture separates concerns between mathematical theory, feature engineering, model training, and production deployment.

### Design Philosophy

- **Separation of Concerns**: Documentation lives in `/01-mathematical-foundations/`, code in `/code/`
- **Modularity**: Each component is independently usable and testable
- **Industry Standards**: Implements techniques used by FICO Falcon, PayPal, and leading fraud detection systems
- **Scalability**: Designed for real-time processing (>10,000 TPS)
- **Educational**: Rich documentation with LaTeX formulas for learning


### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                      │
│         (Transaction streams, user data, metadata)           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Feature Engineering Pipeline                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │Temporal  │ │Behavioral│ │Geographic│ │  Graph   │      │
│  │Features  │ │Deviation │ │Features  │ │Features  │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                 Model Ensemble Layer                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐             │
│  │  LightGBM  │ │ Isolation  │ │   PyOD     │             │
│  │  Detector  │ │   Forest   │ │  Ensemble  │             │
│  └────────────┘ └────────────┘ └────────────┘             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│            Prediction & Scoring Layer                        │
│    (Threshold optimization, business cost calculation)       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Decision & Action Layer                         │
│  (Block transaction, manual review, approve with limits)     │
└─────────────────────────────────────────────────────────────┘
```


***

## Repository Structure Deep Dive

### Complete File Hierarchy

```
fraud-detection/
│
├── README.md                              # Main documentation with LaTeX math
├── IMPLEMENTATION_GUIDE.md                # This comprehensive guide
│
├── 01-mathematical-foundations/           # Pure theory & documentation
│   ├── README.md                          # Mathematical concepts overview
│   ├── statistics-for-fraud-detection.md  # Statistical methods & theory
│   └── linear-algebra-for-ml.md           # Linear algebra foundations
│
└── code/                                  # All executable implementations
    │
    ├── feature_engineering/               # Feature extraction modules
    │   ├── temporal_features.py           ✅ Transaction velocity, time patterns
    │   ├── behavioral_deviation.py        ✅ Z-score, Isolation Forest, LOF, Mahalanobis
    │   ├── aggregation_features.py        ✅ Rolling statistics, user aggregates
    │   ├── geographic_features.py         ✅ IP geolocation, impossible travel
    │   └── graph_features.py              ✅ Fraud ring detection, network analysis
    │
    ├── models/                            # ML model implementations
    │   ├── evaluation_metrics.py          ✅ Precision, Recall, F1, PR-AUC, ROC-AUC
    │   ├── pyod_anomaly_models.py         ✅ 40+ anomaly detection algorithms
    │   ├── lightgbm_fraud_detector.py     ✅ Gradient boosting detector
    │   ├── ensemble_models.py             🔄 Voting & stacking ensembles
    │   ├── xgboost_fraud_detector.py      🔄 XGBoost implementation
    │   └── deep_learning_detector.py      🔄 Neural network detector
    │
    ├── math/                              # Mathematical implementations
    │   ├── distance_metrics.py            🔄 Euclidean, Mahalanobis, Cosine
    │   ├── bayesian_fraud_detector.py     ✅ Bayesian inference (partial)
    │   ├── statistical_tests.py           ✅ Hypothesis testing (partial)
    │   ├── time_series_analysis.py        ✅ ARIMA, seasonal decomposition
    │   └── probability_distributions.py   ✅ Poisson, Normal, Exponential
    │
    └── pipelines/                         # End-to-end workflows
        ├── preprocessing_pipeline.py      🔄 Data cleaning, normalization
        ├── feature_engineering_pipeline.py ✅ Automated feature extraction
        ├── model_training_pipeline.py     ✅ Training & cross-validation
        └── prediction_pipeline.py         ✅ Real-time fraud scoring

Legend: ✅ Fully implemented | 🔄 Partial/stub implementation
```


***

## Mathematical Foundations

### Core Mathematical Concepts

#### 1. Probability \& Statistics

**Normal Distribution** - Foundation for anomaly detection:

$f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

- Used in Z-score calculation: $z = \frac{x - \mu}{\sigma}$
- Transactions beyond 3σ are flagged as suspicious
- **Application**: Detect unusual transaction amounts for a specific user

**Poisson Distribution** - Model transaction frequencies:

$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$

- $\lambda$ = average transaction rate per time window
- **Application**: Detect sudden spikes in transaction count (possible account takeover)

**Chi-Square Test** - Test independence between categorical variables:

$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$

- **Application**: Test if transaction location is independent of user's billing country


#### 2. Linear Algebra for Fraud Detection

**Mahalanobis Distance** - Multivariate anomaly detection accounting for correlations:

$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$

Where:

- $x$ = feature vector for transaction
- $\mu$ = mean vector of normal transactions
- $\Sigma$ = covariance matrix
- $\Sigma^{-1}$ = inverse covariance matrix

**Why Mahalanobis over Euclidean?**

- Accounts for feature correlations
- Scale-invariant (works with different units)
- More robust to outliers in multivariate space

**Singular Value Decomposition (SVD)**:

$A = U \Sigma V^T$

- **Application**: Dimensionality reduction while preserving fraud patterns
- Removes noise from transaction data


#### 3. Information Theory

**Entropy** - Measure uncertainty in transaction patterns:

$H(X) = -\sum_{i=1}^{n} P(x_i) \log P(x_i)$

**Kullback-Leibler Divergence** - Quantify distribution shift:

$D_{KL}(P || Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$

- **Application**: Detect concept drift in transaction patterns over time
- Monitor if current transaction distribution differs from baseline

***

## Feature Engineering Modules

### 1. Temporal Features Module

**File**: `code/feature_engineering/temporal_features.py`

#### Core Concepts

Temporal patterns are critical fraud indicators because:

- Account takeovers show sudden changes in transaction timing
- Fraudsters often act quickly after compromise
- Normal users have consistent temporal patterns


#### Implementation Details

```python
from code.feature_engineering.temporal_features import TemporalFeatureEngineer

# Initialize with your timestamp column name
engineer = TemporalFeatureEngineer(time_column='transaction_time')
```


#### Features Extracted

| Feature Name | Description | Fraud Indicator |
| :-- | :-- | :-- |
| `hour_of_day` | Hour (0-23) when transaction occurred | Unusual hours for user |
| `day_of_week` | Day (0=Monday, 6=Sunday) | Transactions on atypical days |
| `is_weekend` | Binary flag for weekend transactions | Deviates from user pattern |
| `is_night` | Transactions between 10 PM - 5 AM | High-risk time window |
| `velocity_1h` | Transaction count in last hour | Sudden spike indicates bot |
| `velocity_1d` | Transaction count in last 24 hours | Account takeover pattern |
| `time_gap_minutes` | Minutes since last transaction | Unusually short gaps |
| `unusual_time_gap` | Flag for abnormal time gaps | Deviation from history |
| `is_rush_hour` | >5 transactions in 10-minute window | Bot-like behavior |

#### Complete Usage Example

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Sample transaction data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=1000, freq='15min')

df = pd.DataFrame({
    'user_id': np.random.choice(['user_001', 'user_002', 'user_003'], 1000),
    'transaction_time': np.random.choice(dates, 1000),
    'amount': np.random.lognormal(mean=5, sigma=1.5, size=1000)
})

# Sort by user and time (critical for temporal features)
df = df.sort_values(['user_id', 'transaction_time'])

# Extract all temporal features
engineer = TemporalFeatureEngineer(time_column='transaction_time')

# Step 1: Basic temporal features
df = engineer.extract_temporal_features(df, user_id_column='user_id')

# Step 2: Time gap analysis
df = engineer.calculate_time_gaps(df, user_id_column='user_id')

# Step 3: Rush hour detection
df = engineer.detect_rush_hours(df, threshold=5, user_id_column='user_id')

# Identify high-risk transactions
high_risk = df[
    (df['is_night'] == 1) & 
    (df['velocity_1h'] > 10) |
    (df['is_rush_hour'] == 1)
]

print(f"High-risk transactions: {len(high_risk)} / {len(df)}")
print(f"Risk rate: {len(high_risk)/len(df)*100:.2f}%")
```


#### Best Practices

1. **Always sort by user_id and timestamp** before feature extraction
2. **Handle missing timestamps** - fill with median or drop rows
3. **Time zone normalization** - convert all timestamps to UTC
4. **Business hours consideration** - define custom hours based on merchant category
5. **Seasonal patterns** - incorporate month/quarter for recurring patterns

***

### 2. Behavioral Deviation Module

**File**: `code/feature_engineering/behavioral_deviation.py`

#### Core Concepts

Behavioral deviation detection identifies transactions that deviate from a user's historical patterns. This is the most powerful signal for fraud detection.

#### Algorithms Implemented

##### A. Z-Score Based Anomaly Detection

**Formula**:
$z = \frac{x - \mu}{\sigma}$

**Interpretation**:

- $|z| < 2$: Normal (95% of data)
- $2 \leq |z| < 3$: Unusual (4.6% of data)
- $|z| \geq 3$: Anomaly (0.3% of data)

**Advantages**:

- Simple, interpretable
- Fast computation
- Works well for univariate analysis

**Limitations**:

- Assumes normal distribution
- Doesn't capture feature correlations
- Sensitive to outliers in training data


##### B. Isolation Forest

**Concept**: Anomalies are easier to isolate (require fewer random splits)

**Algorithm**:

1. Randomly select feature and split value
2. Recursively partition data
3. Anomalies have shorter path lengths
4. Aggregate results across multiple trees

**Advantages**:

- Handles multivariate data
- No distribution assumptions
- Scales to large datasets
- Robust to outliers

**Hyperparameters**:

- `contamination`: Expected fraud rate (default 0.05)
- `n_estimators`: Number of trees (default 100)
- `max_samples`: Samples per tree (default 256)


##### C. Mahalanobis Distance

**Why use it?**

- Accounts for feature correlations
- Example: High transaction amount + new merchant is more suspicious than each alone

**Implementation Details**:

```python
# Covariance matrix captures feature relationships
cov = np.cov(X.T)
inv_cov = np.linalg.inv(cov)

# Distance for each sample
for sample in X:
    diff = sample - mean
    distance = sqrt(diff.T @ inv_cov @ diff)
```

**Advantages**:

- Scale-invariant
- Captures feature interactions
- More accurate than Euclidean distance

**Limitations**:

- Requires invertible covariance matrix
- Computationally expensive for high dimensions
- Needs sufficient training data


##### D. Local Outlier Factor (LOF)

**Concept**: Compare local density of a point with its neighbors

**How it works**:

1. Calculate k-nearest neighbors for each point
2. Compute local reachability density
3. Compare point's density with neighbor densities
4. LOF > 1 indicates outlier

**Advantages**:

- Detects local anomalies (not just global outliers)
- Works with varying density regions
- No assumption about data distribution

**Best for**: Mixed fraud patterns where different users have different baselines

#### Complete Usage Example

```python
from code.feature_engineering.behavioral_deviation import BehavioralDeviationDetector
import pandas as pd
import numpy as np

# Generate realistic transaction data with anomalies
np.random.seed(42)
n_normal = 1800
n_fraud = 200

# Normal transactions
normal_data = pd.DataFrame({
    'user_id': np.repeat(['user_' + str(i) for i in range(1, 11)], n_normal // 10),
    'transaction_amount': np.random.gamma(shape=2, scale=250, size=n_normal),
    'merchant_diversity': np.random.poisson(lam=8, size=n_normal),
    'transactions_per_day': np.random.poisson(lam=3, size=n_normal),
    'is_fraud': 0
})

# Fraudulent transactions (different distribution)
fraud_data = pd.DataFrame({
    'user_id': np.random.choice(['user_' + str(i) for i in range(1, 11)], n_fraud),
    'transaction_amount': np.random.uniform(5000, 50000, n_fraud),
    'merchant_diversity': np.random.poisson(lam=25, size=n_fraud),
    'transactions_per_day': np.random.poisson(lam=20, size=n_fraud),
    'is_fraud': 1
})

df = pd.concat([normal_data, fraud_data], ignore_index=True)

# Initialize detector
detector = BehavioralDeviationDetector(z_score_threshold=2.5)

# Define features for analysis
features = ['transaction_amount', 'merchant_diversity', 'transactions_per_day']

# Apply all detection methods
print("Running Z-Score Detection...")
df = detector.z_score_anomalies(df, features, group_by='user_id')

print("Running Isolation Forest...")
df = detector.isolation_forest_anomalies(df, features, contamination=0.1)

print("Running Mahalanobis Distance...")
df = detector.mahalanobis_distance_anomalies(df, features, threshold_percentile=90)

print("Running Local Outlier Factor...")
df = detector.local_outlier_factor(df, features, n_neighbors=20)

# Ensemble voting - transaction is fraud if 2+ methods agree
df['ensemble_fraud_flag'] = (
    df['is_anomaly_zscore'] + 
    df['is_anomaly_iforest'] + 
    df['is_anomaly_mahal'] + 
    df['is_anomaly_lof']
) >= 2

# Evaluate performance
from sklearn.metrics import classification_report, confusion_matrix

print("\n=== Ensemble Performance ===")
print(classification_report(df['is_fraud'], df['ensemble_fraud_flag']))
print("\nConfusion Matrix:")
print(confusion_matrix(df['is_fraud'], df['ensemble_fraud_flag']))

# Analyze detection rates by method
detection_summary = pd.DataFrame({
    'Method': ['Z-Score', 'Isolation Forest', 'Mahalanobis', 'LOF', 'Ensemble'],
    'Detected': [
        df['is_anomaly_zscore'].sum(),
        df['is_anomaly_iforest'].sum(),
        df['is_anomaly_mahal'].sum(),
        df['is_anomaly_lof'].sum(),
        df['ensemble_fraud_flag'].sum()
    ],
    'True Positives': [
        ((df['is_anomaly_zscore'] == 1) & (df['is_fraud'] == 1)).sum(),
        ((df['is_anomaly_iforest'] == 1) & (df['is_fraud'] == 1)).sum(),
        ((df['is_anomaly_mahal'] == 1) & (df['is_fraud'] == 1)).sum(),
        ((df['is_anomaly_lof'] == 1) & (df['is_fraud'] == 1)).sum(),
        ((df['ensemble_fraud_flag'] == 1) & (df['is_fraud'] == 1)).sum()
    ]
})

print("\n=== Method Comparison ===")
print(detection_summary)
```


***

### 3. Aggregation Features Module

**File**: `code/feature_engineering/aggregation_features.py`

#### Purpose

Aggregate user behavior over time windows to establish baseline patterns and detect deviations.

#### Key Features Extracted

**Rolling Statistics** (1 hour, 1 day, 7 days, 30 days):

- Mean transaction amount
- Standard deviation of amounts
- Maximum transaction
- Transaction count
- Unique merchants contacted

**User-Level Lifetime Statistics**:

- Total transactions to date
- Average transaction amount (all time)
- Days since account creation
- Days since last transaction
- Merchant diversity score


#### Usage Pattern

```python
from code.feature_engineering.aggregation_features import AggregationFeatureEngineer

engineer = AggregationFeatureEngineer()

# Extract rolling statistics
df = engineer.extract_rolling_features(
    df, 
    windows=['1h', '1d', '7d', '30d'],
    features=['amount', 'merchant_id']
)

# Extract user lifetime statistics
df = engineer.extract_user_aggregates(df, user_id_column='user_id')

# Deviation features
df['amount_vs_user_mean'] = df['amount'] / (df['user_mean_amount'] + 1e-5)
df['is_max_amount'] = (df['amount'] == df['user_max_amount']).astype(int)
```


***

### 4. Geographic Features Module

**File**: `code/feature_engineering/geographic_features.py`

#### Impossible Travel Detection

**Haversine Formula** - Calculate distance between two lat/lon points:

$d = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$

Where:

- $r$ = Earth radius (6371 km)
- $\phi$ = latitude
- $\lambda$ = longitude

**Velocity Calculation**:
$v = \frac{d}{\Delta t}$

**Fraud Indicator**: $v > 1000$ km/hr (faster than commercial flight)

#### Features Extracted

- Distance from previous transaction (km)
- Travel velocity (km/hr)
- Country mismatch flag
- IP geolocation risk score
- VPN/proxy detection

***

### 5. Graph Features Module

**File**: `code/feature_engineering/graph_features.py`

#### Fraud Ring Detection

Build transaction network where:

- **Nodes**: Users, merchants, devices, IP addresses
- **Edges**: Transactions connecting entities


#### Graph Metrics

**Degree Centrality**: Number of connections
$C_D(v) = \frac{deg(v)}{n-1}$

**Betweenness Centrality**: Number of shortest paths through node
$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$

**Community Detection**: Identify tightly connected fraud rings

#### Application

```python
import networkx as nx

# Build transaction graph
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row['user_id'], row['merchant_id'], weight=row['amount'])

# Calculate centrality
centrality = nx.degree_centrality(G)
df['user_centrality'] = df['user_id'].map(centrality)

# Detect communities (fraud rings)
communities = nx.community.greedy_modularity_communities(G)
```


***

## Model Implementations

### 1. Evaluation Metrics Module

**File**: `code/models/evaluation_metrics.py`

#### Why Standard Accuracy Fails

For fraud detection with 0.1% fraud rate:

- Model that predicts "NO FRAUD" for everything: **99.9% accuracy** ✗
- But catches **0% of fraud** ✗


#### Correct Metrics

**Precision** - Of flagged transactions, how many are truly fraud?
$\text{Precision} = \frac{TP}{TP + FP}$

**Recall** - Of actual fraud, how much did we catch?
$\text{Recall} = \frac{TP}{TP + FN}$

**F1-Score** - Harmonic mean balancing precision and recall:
$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

**PR-AUC** - Area under Precision-Recall curve (better for imbalanced data)

**Business Cost Metric**:
$\text{Cost} = (FP \times \text{cost}_{\text{FP}}) + (FN \times \text{cost}_{\text{FN}})$

Where:

- $\text{cost}_{\text{FP}}$ = Customer frustration + manual review (\$1-10)
- $\text{cost}_{\text{FN}}$ = Fraud loss + chargeback (\$100-10,000)


#### Complete Usage

```python
from code.models.evaluation_metrics import FraudDetectionMetrics

# Initialize with business costs
evaluator = FraudDetectionMetrics(
    cost_fp=5.0,      # $5 per false alarm
    cost_fn=500.0     # $500 per missed fraud
)

# Calculate all metrics
metrics = evaluator.calculate_metrics(
    y_true=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba
)

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1-Score: {metrics['f1_score']:.3f}")
print(f"PR-AUC: {metrics['pr_auc']:.3f}")
print(f"Business Cost: ${metrics['business_cost']:,.2f}")

# Find optimal threshold
optimal_threshold, optimal_f1 = evaluator.find_optimal_threshold(
    y_test, 
    y_pred_proba, 
    metric='f1'
)

print(f"\nOptimal Decision Threshold: {optimal_threshold:.3f}")
print(f"F1-Score at Optimal: {optimal_f1:.3f}")

# Plot curves
evaluator.plot_roc_curve(y_test, y_pred_proba)
evaluator.plot_precision_recall_curve(y_test, y_pred_proba)
```


***

### 2. LightGBM Fraud Detector

**File**: `code/models/lightgbm_fraud_detector.py`

#### Why LightGBM for Fraud Detection?

- **Leaf-wise growth**: Deeper, more accurate trees
- **Histogram-based**: Faster training on large datasets
- **Native categorical support**: No need to one-hot encode
- **Built-in class imbalance handling**: `is_unbalance=True`


#### Hyperparameters Explained

```python
params = {
    'objective': 'binary',           # Binary classification
    'metric': 'auc',                 # Optimize for ROC-AUC
    'boosting_type': 'gbdt',         # Gradient boosting decision tree
    'num_leaves': 31,                # Max leaves per tree (2^depth - 1)
    'learning_rate': 0.05,           # Step size shrinkage
    'feature_fraction': 0.9,         # Sample 90% features per tree
    'bagging_fraction': 0.8,         # Sample 80% data per iteration
    'bagging_freq': 5,               # Bagging every 5 iterations
    'is_unbalance': True,            # Handle class imbalance
    'max_depth': -1,                 # No limit (use num_leaves instead)
    'min_child_samples': 20,         # Min samples in leaf
    'reg_alpha': 0.1,                # L1 regularization
    'reg_lambda': 0.1,               # L2 regularization
}
```


#### Training Pipeline

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    early_stopping_rounds=50,
    verbose_eval=100
)

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance.head(10))
```


***

### 3. PyOD Anomaly Models

**File**: `code/models/pyod_anomaly_models.py`

#### Multi-Algorithm Ensemble

PyOD provides 40+ anomaly detection algorithms. This module implements ensemble voting:

```python
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

class PyODAnomalyDetector:
    def __init__(self, contamination=0.05):
        self.models = {
            'iforest': IForest(contamination=contamination),
            'lof': LOF(contamination=contamination),
            'ocsvm': OCSVM(contamination=contamination)
        }
    
    def fit(self, X):
        for name, model in self.models.items():
            model.fit(X)
    
    def predict(self, X):
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Majority voting
        ensemble_pred = (sum(predictions.values()) >= 2).astype(int)
        return ensemble_pred
```


***

## Pipeline Architecture

### Feature Engineering Pipeline

**File**: `code/pipelines/feature_engineering_pipeline.py`

#### Complete End-to-End Example

```python
from code.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
import pandas as pd

# Load raw transaction data
df = pd.read_csv('transactions.csv')

# Initialize pipeline
pipeline = FeatureEngineeringPipeline(
    include_temporal=True,
    include_behavioral=True,
    include_geographic=True,
    include_graph=True
)

# Transform data
df_features, feature_names = pipeline.fit_transform(df)

print(f"Original features: {df.shape[^1]}")
print(f"Engineered features: {df_features.shape[^1]}")
print(f"New features added: {len(feature_names)}")
```


***

### Model Training Pipeline

**File**: `code/pipelines/model_training_pipeline.py`

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, y_pred_proba)
    cv_scores.append(auc)
    print(f"Fold {fold+1}: AUC = {auc:.4f}")

print(f"\nMean CV AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
```


***

### Prediction Pipeline

**File**: `code/pipelines/prediction_pipeline.py`

Real-time fraud scoring system:

```python
class FraudPredictionPipeline:
    def __init__(self, feature_pipeline, model, threshold=0.5):
        self.feature_pipeline = feature_pipeline
        self.model = model
        self.threshold = threshold
    
    def predict_transaction(self, transaction_dict):
        """Score single transaction in real-time"""
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction_dict])
        
        # Extract features
        df_features, _ = self.feature_pipeline.fit_transform(df)
        
        # Get fraud probability
        fraud_prob = self.model.predict_proba(df_features)[0, 1]
        
        # Make decision
        decision = {
            'fraud_probability': fraud_prob,
            'is_fraud': fraud_prob >= self.threshold,
            'action': 'BLOCK' if fraud_prob >= 0.9 else 
                     'REVIEW' if fraud_prob >= 0.5 else 'APPROVE',
            'risk_level': 'HIGH' if fraud_prob >= 0.7 else
                         'MEDIUM' if fraud_prob >= 0.3 else 'LOW'
        }
        
        return decision

# Usage
pipeline = FraudPredictionPipeline(feature_pipeline, model, threshold=0.6)

transaction = {
    'user_id': 'user_12345',
    'amount': 5000.00,
    'merchant_id': 'merch_999',
    'timestamp': '2024-01-15 03:45:00',
    'latitude': 40.7128,
    'longitude': -74.0060
}

result = pipeline.predict_transaction(transaction)
print(f"Fraud Probability: {result['fraud_probability']:.2%}")
print(f"Action: {result['action']}")
```


***

## End-to-End Implementation Workflow

### Step-by-Step Production Deployment

#### Step 1: Data Preparation

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('transactions.csv', parse_dates=['timestamp'])

# Basic cleaning
df = df.drop_duplicates()
df = df.dropna(subset=['user_id', 'amount', 'timestamp'])

# Sort by time (critical for temporal features)
df = df.sort_values(['user_id', 'timestamp'])

print(f"Loaded {len(df):,} transactions")
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
```


#### Step 2: Feature Engineering

```python
# Temporal features
from code.feature_engineering.temporal_features import TemporalFeatureEngineer
temp_engineer = TemporalFeatureEngineer(time_column='timestamp')
df = temp_engineer.extract_temporal_features(df)
df = temp_engineer.calculate_time_gaps(df)
df = temp_engineer.detect_rush_hours(df, threshold=5)

# Behavioral features
from code.feature_engineering.behavioral_deviation import BehavioralDeviationDetector
behav_detector = BehavioralDeviationDetector(z_score_threshold=3)
features_to_analyze = ['amount', 'velocity_1h', 'time_gap_hours']
df = behav_detector.z_score_anomalies(df, features_to_analyze)
df = behav_detector.isolation_forest_anomalies(df, features_to_analyze)

print(f"Feature count after engineering: {df.shape[^1]}")
```


#### Step 3: Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Features for modeling (exclude target and IDs)
exclude_cols = ['is_fraud', 'user_id', 'transaction_id', 'timestamp']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['is_fraud']

# Stratified split (maintain fraud rate)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training set: {len(X_train):,} ({y_train.mean():.2%} fraud)")
print(f"Test set: {len(X_test):,} ({y_test.mean():.2%} fraud)")
```


#### Step 4: Model Training

```python
import lightgbm as lgb

# Handle class imbalance with scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'scale_pos_weight': scale_pos_weight,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    early_stopping_rounds=50
)
```


#### Step 5: Model Evaluation

```python
from code.models.evaluation_metrics import FraudDetectionMetrics

# Predictions
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Evaluate
evaluator = FraudDetectionMetrics(cost_fp=10, cost_fn=500)
metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)

print("\n=== Model Performance ===")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1-Score: {metrics['f1_score']:.3f}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"PR-AUC: {metrics['pr_auc']:.3f}")
print(f"\nConfusion Matrix:")
print(f"TN: {metrics['tn']}, FP: {metrics['fp']}")
print(f"FN: {metrics['fn']}, TP: {metrics['tp']}")
print(f"\nBusiness Cost: ${metrics['business_cost']:,.2f}")

# Find optimal threshold
optimal_threshold, optimal_f1 = evaluator.find_optimal_threshold(
    y_test, y_pred_proba, metric='f1'
)
print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
```


#### Step 6: Model Persistence

```python
import joblib
import json

# Save model
model.save_model('fraud_model_v1.txt')
joblib.dump(feature_cols, 'feature_columns.pkl')

# Save metadata
metadata = {
    'model_version': '1.0',
    'training_date': '2024-01-15',
    'training_samples': len(X_train),
    'test_auc': float(metrics['roc_auc']),
    'optimal_threshold': float(optimal_threshold),
    'feature_count': len(feature_cols)
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```


***

## Advanced Usage Patterns

### 1. Ensemble Model Stacking

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# Base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)

# Get out-of-fold predictions for stacking
rf_pred = cross_val_predict(rf, X_train, y_train, cv=5, method='predict_proba')[:, 1]
gb_pred = cross_val_predict(gb, X_train, y_train, cv=5, method='predict_proba')[:, 1]
lgb_pred = cross_val_predict(lgb_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]

# Create meta-features
X_meta = np.column_stack([rf_pred, gb_pred, lgb_pred])

# Train meta-learner
meta_learner = LogisticRegression()
meta_learner.fit(X_meta, y_train)

# Predictions on test set
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

X_test_meta = np.column_stack([
    rf.predict_proba(X_test)[:, 1],
    gb.predict_proba(X_test)[:, 1],
    lgb_model.predict_proba(X_test)[:, 1]
])

y_pred_stacked = meta_learner.predict_proba(X_test_meta)[:, 1]
```


### 2. Real-Time Feature Store

```python
import redis
import json

class FeatureStore:
    """Redis-based feature store for real-time fraud detection"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
    
    def update_user_features(self, user_id, features):
        """Update user's aggregated features"""
        key = f"user_features:{user_id}"
        self.redis_client.set(key, json.dumps(features), ex=86400)  # 24h TTL
    
    def get_user_features(self, user_id):
        """Retrieve user features for real-time scoring"""
        key = f"user_features:{user_id}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else None
    
    def increment_velocity(self, user_id, window='1h'):
        """Increment transaction velocity counter"""
        key = f"velocity:{user_id}:{window}"
        self.redis_client.incr(key)
        self.redis_client.expire(key, 3600 if window == '1h' else 86400)

# Usage
store = FeatureStore()

# Update after each transaction
store.update_user_features('user_123', {
    'mean_amount': 150.50,
    'std_amount': 75.25,
    'total_transactions': 245
})

store.increment_velocity('user_123', window='1h')

# Retrieve for real-time scoring
features = store.get_user_features('user_123')
```


### 3. Concept Drift Monitoring

```python
from scipy.stats import ks_2samp

def detect_concept_drift(X_train, X_new, threshold=0.05):
    """Detect distribution shift using Kolmogorov-Smirnov test"""
    
    drift_detected = {}
    
    for col in X_train.columns:
        statistic, p_value = ks_2samp(X_train[col], X_new[col])
        drift_detected[col] = {
            'drift': p_value < threshold,
            'p_value': p_value,
            'statistic': statistic
        }
    
    return drift_detected

# Monthly drift check
drift_results = detect_concept_drift(X_train, X_recent_month)
drifted_features = [k for k, v in drift_results.items() if v['drift']]

if drifted_features:
    print(f"⚠️ Drift detected in {len(drifted_features)} features:")
    print(drifted_features)
    print("→ Consider retraining model")
```


***

## Production Deployment Guide

### Architecture Components

```
┌─────────────────────────────────────────────────────────┐
│                  API Gateway (NGINX)                     │
│                  Rate Limiting: 10K TPS                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│             Fraud Scoring Service (FastAPI)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Feature Eng  │  │ Model Scorer │  │ Rule Engine  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Feature Store (Redis Cluster)               │
│         User aggregates, velocity counters               │
└─────────────────────────────────────────────────────────┘
```


### FastAPI Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import lightgbm as lgb
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection API")

# Load model at startup
model = lgb.Booster(model_file='fraud_model_v1.txt')
feature_cols = joblib.load('feature_columns.pkl')

class Transaction(BaseModel):
    user_id: str
    amount: float
    merchant_id: str
    timestamp: str
    latitude: float
    longitude: float

class FraudScore(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    action: str

@app.post("/score", response_model=FraudScore)
async def score_transaction(transaction: Transaction):
    try:
        # Feature engineering (simplified)
        features = extract_features(transaction)
        
        # Score with model
        fraud_prob = model.predict([features])[^0]
        
        # Determine action
        if fraud_prob >= 0.9:
            action = "BLOCK"
            risk_level = "HIGH"
        elif fraud_prob >= 0.5:
            action = "REVIEW"
            risk_level = "MEDIUM"
        else:
            action = "APPROVE"
            risk_level = "LOW"
        
        return FraudScore(
            transaction_id=generate_id(),
            fraud_probability=float(fraud_prob),
            risk_level=risk_level,
            action=action
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```


### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and model
COPY code/ ./code/
COPY fraud_model_v1.txt .
COPY feature_columns.pkl .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```


### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detection
  template:
    metadata:
      labels:
        app: fraud-detection
    spec:
      containers:
      - name: fraud-api
        image: fraud-detection:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```


***

## Performance Optimization

### 1. Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# Statistical feature selection
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X_train, y_train)

# Tree-based feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Select top 50 features
top_features = feature_importance.head(50)['feature'].tolist()
X_train_optimized = X_train[top_features]
```


### 2. Model Compression

```python
# Quantization for faster inference
model_compressed = lgb.Booster(model_file='fraud_model_v1.txt')
model_compressed.save_model('fraud_model_compressed.txt', 
                           num_iteration=100,  # Use first 100 trees
                           importance_type='split')

# ONNX conversion for cross-platform deployment
import onnxmltools
from onnxmltools.convert import convert_lightgbm

onnx_model = convert_lightgbm(model, 
                              initial_types=[('input', FloatTensorType([None, n_features]))])
onnxmltools.utils.save_model(onnx_model, 'fraud_model.onnx')
```


### 3. Caching Strategy

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def get_user_features_cached(user_id):
    """Cache user features for repeated lookups"""
    return feature_store.get_user_features(user_id)

def cache_key(transaction_dict):
    """Generate cache key from transaction"""
    key_string = f"{transaction_dict['user_id']}_{transaction_dict['amount']}"
    return hashlib.md5(key_string.encode()).hexdigest()
```


***

## Troubleshooting \& Best Practices

### Common Issues

#### Issue 1: Class Imbalance

**Problem**: Model predicts all transactions as non-fraud

**Solutions**:

```python
# 1. Use scale_pos_weight
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
params['scale_pos_weight'] = scale_pos_weight

# 2. SMOTE oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 3. Adjust class weights
from sklearn.utils import class_weight
weights = class_weight.compute_class_weight('balanced', 
                                            classes=np.unique(y_train),
                                            y=y_train)
```


#### Issue 2: Feature Leakage

**Problem**: Unrealistically high performance on test set

**Check for**:

- Using future information in features
- Including target-derived features
- Data leakage from train to test

**Solution**:

```python
# Temporal validation split (not random)
split_date = '2024-01-01'
train = df[df['timestamp'] < split_date]
test = df[df['timestamp'] >= split_date]
```


#### Issue 3: Overfitting

**Symptoms**:

- Training AUC >> Test AUC
- Model performs poorly on new data

**Solutions**:

```python
# Increase regularization
params['reg_alpha'] = 1.0  # L1
params['reg_lambda'] = 1.0  # L2
params['min_child_samples'] = 50
params['max_depth'] = 5

# Use early stopping
early_stopping_rounds = 50

# Reduce model complexity
params['num_leaves'] = 15
```


### Production Monitoring

```python
import prometheus_client as prom

# Define metrics
fraud_rate = prom.Gauge('fraud_detection_rate', 'Current fraud detection rate')
latency = prom.Histogram('fraud_scoring_latency', 'Scoring latency in seconds')
model_version = prom.Info('fraud_model_version', 'Current model version')

# Update metrics
@latency.time()
def score_transaction(transaction):
    # Scoring logic
    pass

fraud_rate.set(current_fraud_rate)
model_version.info({'version': '1.0', 'trained': '2024-01-15'})
```


***

## Conclusion

This comprehensive implementation guide covers all aspects of the fraud detection toolkit from mathematical foundations to production deployment. The modular architecture allows you to:

1. **Learn**: Study mathematical concepts with LaTeX formulas
2. **Experiment**: Run individual modules independently
3. **Integrate**: Combine modules into complete pipelines
4. **Deploy**: Scale to production with provided architecture

### Next Steps

1. **Implement and integrate your own modules**: Complete XGBoost, deep learning, and ensemble models based on choice of dataset
2. **Adding data sources**: Integrate with your transaction databases
3. **Tune hyperparameters**: Optimize for your specific fraud patterns
4. **Deploy to production**: Use provided FastAPI and Docker configurations
5. **Monitor performance**: Track metrics and retrain regularly

### Resources

- **Documentation**: `/01-mathematical-foundations/` for theory
- **Code Examples**: Each module has runnable `if __name__ == '__main__'` blocks
- **Papers**: References in README.md
- **Industry Standards**: FICO Falcon, PayPal techniques

***

**Version**: 2.0
**Last Updated**: January 2, 2026
**Status**: Production-Ready Core + Extensible Framework
**GitHub**: [shubro18202758/python-ml-basics/fraud-detection](https://github.com/shubro18202758/python-ml-basics/tree/main/fraud-detection)

***

This comprehensive guide provides everything needed to understand, implement, and deploy the fraud detection system.
