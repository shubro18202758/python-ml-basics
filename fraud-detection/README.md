# Fraud Detection & Anomaly Detection Toolkit

## Industry-Standard Framework for Financial Fraud Detection

This comprehensive toolkit provides cutting-edge knowledge and practical implementations for fraud detection in financial datasets, designed for students and professionals seeking industry-level expertise in ML/AI-powered fraud prevention.

---

## 📚 Table of Contents

1. [Core Technologies & Libraries](#core-technologies--libraries)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Feature Engineering Techniques](#feature-engineering-techniques)
4. [Industry Tools & Frameworks](#industry-tools--frameworks)
5. [Handling Imbalanced Datasets](#handling-imbalanced-datasets)
6. [Model Selection & Evaluation](#model-selection--evaluation)
7. [Real-Time Detection Systems](#real-time-detection-systems)
8. [Advanced Topics](#advanced-topics)

---

## 🛠️ Core Technologies & Libraries

### Essential Python Libraries

#### 1. **scikit-learn** - Foundation for ML Models
- Industry standard for traditional ML algorithms
- Key modules for fraud detection:
  - `sklearn.ensemble`: Random Forest, Gradient Boosting, Isolation Forest
  - `sklearn.svm`: Support Vector Machines for anomaly detection
  - `sklearn.neighbors`: K-Nearest Neighbors for pattern matching
  - `sklearn.preprocessing`: Feature scaling and transformation
  - `sklearn.metrics`: Precision, recall, F1-score, ROC-AUC

```python
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score
```

#### 2. **XGBoost & LightGBM** - Gradient Boosting Champions
- **XGBoost**: Extreme gradient boosting, winner of numerous Kaggle competitions
  - Handles missing values automatically
  - Built-in regularization prevents overfitting
  - Parallel processing for faster training
  
- **LightGBM**: Microsoft's high-performance framework
  - Leaf-wise tree growth (faster than XGBoost)
  - Optimal for large-scale datasets
  - Lower memory consumption

```python
import xgboost as xgb
import lightgbm as lgb

# XGBoost for fraud detection
model = xgb.XGBClassifier(
    scale_pos_weight=99,  # For imbalanced data
    max_depth=6,
    learning_rate=0.1
)
```

#### 3. **PyOD** - Specialized Outlier Detection Library
- 40+ anomaly detection algorithms
- Unified API similar to scikit-learn
- Key algorithms:
  - **HBOS**: Histogram-based Outlier Score
  - **COPOD**: Copula-based Outlier Detection
  - **ECOD**: Empirical Cumulative distribution Outlier Detection
  - **Deep Learning Models**: AutoEncoder, VAE

```python
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
```

#### 4. **imbalanced-learn** - Handling Class Imbalance
- Resampling techniques for imbalanced datasets
- SMOTE and variants (ADASYN, BorderlineSMOTE)
- Integration with scikit-learn pipelines

```python
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek
```

#### 5. **TensorFlow/PyTorch** - Deep Learning for Complex Patterns
- Neural networks for high-dimensional feature spaces
- Autoencoders for unsupervised anomaly detection
- LSTMs for sequential transaction patterns

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Autoencoder architecture
encoder = tf.keras.Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu')
])
```

#### 6. **SHAP & LIME** - Model Explainability
- Critical for regulatory compliance (explain fraud decisions)
- SHAP: Game theory-based feature importance
- LIME: Local interpretable model-agnostic explanations

```python
import shap
import lime

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

---

## 📐 Mathematical Foundations

### Statistics & Probability Theory

#### 1. **Probability Distributions**

**Normal Distribution** - Baseline for anomaly detection:
\[
f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
\]

- Used in z-score anomaly detection
- Transactions beyond 3σ flagged as suspicious

**Poisson Distribution** - Model transaction frequencies:
\[
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
\]

- Detect unusual transaction counts per time window
- Model card usage patterns

#### 2. **Statistical Tests**

**Chi-Square Test** - Categorical feature independence:
\[
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
\]

**Kolmogorov-Smirnov Test** - Distribution comparison:
- Compare transaction distributions before/after fraud events
- Detect concept drift in transaction patterns

#### 3. **Information Theory**

**Entropy** - Measure uncertainty:
\[
H(X) = -\sum_{i=1}^{n} P(x_i) \log P(x_i)
\]

**Kullback-Leibler Divergence** - Distribution difference:
\[
D_{KL}(P || Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}
\]

- Quantify deviation from normal transaction behavior
- Monitor distribution shifts over time

### Linear Algebra

#### 1. **Mahalanobis Distance** - Multivariate anomaly detection:
\[
D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
\]

- Accounts for feature correlations
- More robust than Euclidean distance

#### 2. **Singular Value Decomposition (SVD)**
\[
A = U \Sigma V^T
\]

- Dimensionality reduction for transaction data
- Remove noise while preserving fraud patterns

### Optimization Theory

#### 1. **Gradient Descent** - Model training:
\[
\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)
\]

#### 2. **Cross-Entropy Loss** - Classification objective:
\[
L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]
\]

---

## 🔧 Feature Engineering Techniques

### Temporal Features

```python
import pandas as pd
import numpy as np

def create_temporal_features(df):
    """
    Extract time-based patterns crucial for fraud detection
    """
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].between(22, 6).astype(int)  # 10 PM - 6 AM
    
    # Transaction velocity features
    df['time_since_last_txn'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
    df['txn_count_1hr'] = df.groupby('user_id')['timestamp'].transform(
        lambda x: x.rolling('1H').count()
    )
    
    return df
```

### Aggregation Features

```python
def create_aggregation_features(df, windows=['1H', '24H', '7D']):
    """
    Statistical aggregations over time windows
    """
    features = {}
    
    for window in windows:
        # Amount statistics
        features[f'amount_mean_{window}'] = df.groupby('user_id')['amount'].transform(
            lambda x: x.rolling(window).mean()
        )
        features[f'amount_std_{window}'] = df.groupby('user_id')['amount'].transform(
            lambda x: x.rolling(window).std()
        )
        features[f'amount_max_{window}'] = df.groupby('user_id')['amount'].transform(
            lambda x: x.rolling(window).max()
        )
        
        # Transaction count
        features[f'txn_count_{window}'] = df.groupby('user_id')['timestamp'].transform(
            lambda x: x.rolling(window).count()
        )
        
        # Unique merchant count
        features[f'unique_merchants_{window}'] = df.groupby('user_id')['merchant_id'].transform(
            lambda x: x.rolling(window).nunique()
        )
    
    return pd.DataFrame(features)
```

### Behavioral Deviation Features

```python
def deviation_features(df):
    """
    Measure how current transaction deviates from user's history
    """
    # User historical statistics
    user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std']).reset_index()
    df = df.merge(user_stats, on='user_id', suffixes=('', '_hist'))
    
    # Z-score deviation
    df['amount_zscore'] = (df['amount'] - df['mean']) / (df['std'] + 1e-10)
    
    # Percentage deviation
    df['amount_pct_change'] = (df['amount'] - df['mean']) / (df['mean'] + 1e-10)
    
    # Binary: exceeds 3 standard deviations
    df['is_extreme_amount'] = (np.abs(df['amount_zscore']) > 3).astype(int)
    
    return df
```

### Geographic Features

```python
def geographic_features(df):
    """
    Location-based fraud indicators
    """
    # Distance from previous transaction
    df['distance_from_prev'] = haversine_distance(
        df['prev_lat'], df['prev_lon'],
        df['curr_lat'], df['curr_lon']
    )
    
    # Velocity: km/hr between transactions
    df['velocity'] = df['distance_from_prev'] / (df['time_since_last_txn'] / 3600 + 1e-10)
    
    # Impossible travel flag (> 1000 km/hr)
    df['impossible_travel'] = (df['velocity'] > 1000).astype(int)
    
    # Country mismatch with billing address
    df['country_mismatch'] = (df['txn_country'] != df['billing_country']).astype(int)
    
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth"""
    R = 6371  # Earth's radius in km
    
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c
```

### Graph-Based Features

```python
import networkx as nx

def create_transaction_graph(df):
    """
    Model transaction network for fraud rings detection
    """
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        G.add_edge(row['user_id'], row['merchant_id'], 
                   amount=row['amount'], 
                   timestamp=row['timestamp'])
    
    # Node centrality measures
    pagerank = nx.pagerank(G)
    betweenness = nx.betweenness_centrality(G)
    
    # Community detection
    communities = nx.community.greedy_modularity_communities(G.to_undirected())
    
    return G, pagerank, communities
```

---

## 🏢 Industry Tools & Frameworks

### Commercial Fraud Detection Platforms

#### 1. **FICO Falcon Platform**
- Neural network-based scoring
- Real-time consortium data sharing
- Adaptive analytics engine
- **Use Case**: Large banks, credit card issuers

#### 2. **DataDome**
- AI-powered bot detection
- Real-time threat prevention
- Behavioral analysis engine
- **Use Case**: E-commerce, online platforms

#### 3. **Kount**
- Identity Trust Global Network
- AI-driven payment fraud prevention
- Account takeover protection
- **Use Case**: Payment processors, fintech

#### 4. **Feedzai**
- End-to-end fraud prevention
- Graph analytics for fraud rings
- Automated machine learning
- **Use Case**: Financial institutions, marketplaces

### Open-Source Frameworks

#### 1. **Apache Kafka + Apache Flink**
- Real-time stream processing
- Event-driven architecture
- Scalable fraud detection pipelines

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import MapFunction

class FraudDetector(MapFunction):
    def map(self, transaction):
        # Real-time fraud scoring
        score = self.model.predict_proba(transaction)[0][1]
        return transaction, score
```

#### 2. **MLflow**
- Experiment tracking
- Model versioning and registry
- Deployment management

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("max_depth", 6)
    mlflow.log_metric("auc_roc", 0.95)
    mlflow.sklearn.log_model(model, "fraud_detector")
```

---

## ⚖️ Handling Imbalanced Datasets

### Problem Statement

Fraud datasets typically exhibit severe class imbalance:
- Legitimate transactions: 99.8%+
- Fraudulent transactions: 0.2% or less

### Resampling Techniques

#### 1. **SMOTE (Synthetic Minority Over-sampling Technique)**

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"Original dataset shape: {X_train.shape}")
print(f"Resampled dataset shape: {X_resampled.shape}")
```

**How it works:**
1. Select minority class sample
2. Find k nearest neighbors
3. Generate synthetic samples along line segments

\[
x_{\text{new}} = x_i + \lambda \cdot (x_{\text{neighbor}} - x_i), \quad \lambda \in [0, 1]
\]

#### 2. **ADASYN (Adaptive Synthetic Sampling)**

```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(sampling_strategy='auto', n_neighbors=5)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
```

- Focuses on harder-to-learn samples
- Generates more synthetic data in difficult regions

#### 3. **Hybrid Approaches**

```python
from imblearn.combine import SMOTETomek, SMOTEENN

# SMOTE + Tomek Links
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# Oversample then clean borderline samples
```

### Cost-Sensitive Learning

```python
import xgboost as xgb

# Calculate scale_pos_weight
fraud_count = (y_train == 1).sum()
legit_count = (y_train == 0).sum()
scale_pos_weight = legit_count / fraud_count

model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)
```

### Evaluation Metrics for Imbalanced Data

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    confusion_matrix
)

def evaluate_fraud_model(y_true, y_pred, y_proba):
    """
    Comprehensive evaluation for imbalanced fraud detection
    """
    results = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'avg_precision': average_precision_score(y_true, y_proba)
    }
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results['false_positive_rate'] = fp / (fp + tn)
    results['false_negative_rate'] = fn / (fn + tp)
    
    # Business metrics
    avg_fraud_amount = 500  # Example average fraud amount
    avg_investigation_cost = 50
    
    results['cost_savings'] = tp * avg_fraud_amount - fp * avg_investigation_cost
    
    return results
```

**Key Metrics:**
- **Precision**: Of flagged transactions, what % are actually fraud?
- **Recall**: Of all fraud, what % did we catch?
- **F1-Score**: Harmonic mean of precision and recall
- **PR-AUC**: Area under precision-recall curve (better than ROC-AUC for imbalanced data)

---

## 🎯 Model Selection & Evaluation

### Algorithm Comparison

| Algorithm | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **Isolation Forest** | Fast, unsupervised, handles high dimensions | May miss subtle patterns | Initial anomaly screening |
| **Random Forest** | Robust, interpretable, handles non-linear relationships | Slower training | General-purpose fraud detection |
| **XGBoost/LightGBM** | SOTA performance, handles missing values | Requires tuning | Production systems |
| **Neural Networks** | Captures complex patterns, scalable | Black-box, requires large data | Large-scale fraud detection |
| **One-Class SVM** | Good for outlier detection | Sensitive to parameter tuning | Unsupervised fraud detection |
| **Autoencoders** | Unsupervised, dimensionality reduction | Computationally expensive | High-dimensional transaction data |

### Complete Training Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb

def train_fraud_detection_model(df):
    """
    End-to-end fraud detection model training
    """
    # Feature engineering
    df = create_temporal_features(df)
    df = create_aggregation_features(df)
    df = deviation_features(df)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['is_fraud', 'transaction_id', 'timestamp']]
    X = df[feature_cols]
    y = df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Handle imbalanced data
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        random_state=42
    )
    
    model.fit(
        X_train_scaled, y_train_resampled,
        eval_set=[(X_test_scaled, y_test)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    results = evaluate_fraud_model(y_test, y_pred, y_proba)
    
    return model, scaler, results
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'n_estimators': randint(100, 500),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10)
}

random_search = RandomizedSearchCV(
    xgb.XGBClassifier(scale_pos_weight=scale_pos_weight),
    param_distributions=param_distributions,
    n_iter=50,
    cv=StratifiedKFold(n_splits=5),
    scoring='average_precision',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
```

---

## ⚡ Real-Time Detection Systems

### Architecture Components

```python
import redis
import json
from kafka import KafkaConsumer, KafkaProducer

class RealTimeFraudDetector:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.redis_client = redis.Redis(host='localhost', port=6379)
        
        # Kafka consumer for incoming transactions
        self.consumer = KafkaConsumer(
            'transactions',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        # Kafka producer for alerts
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
    
    def extract_features(self, transaction):
        """
        Real-time feature extraction with Redis caching
        """
        user_id = transaction['user_id']
        
        # Retrieve user history from Redis
        user_history = self.redis_client.get(f'user:{user_id}:history')
        
        if user_history:
            history = json.loads(user_history)
            # Calculate deviation features
            features = self.calculate_deviation(transaction, history)
        else:
            features = self.extract_basic_features(transaction)
        
        # Update user history
        self.update_user_history(user_id, transaction)
        
        return features
    
    def detect_fraud(self, transaction):
        """
        Real-time fraud detection
        """
        features = self.extract_features(transaction)
        features_scaled = self.scaler.transform([features])
        
        fraud_probability = self.model.predict_proba(features_scaled)[0][1]
        
        # Risk scoring
        if fraud_probability > 0.9:
            risk_level = 'CRITICAL'
        elif fraud_probability > 0.7:
            risk_level = 'HIGH'
        elif fraud_probability > 0.5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'transaction_id': transaction['transaction_id'],
            'fraud_probability': fraud_probability,
            'risk_level': risk_level,
            'timestamp': transaction['timestamp']
        }
    
    def run(self):
        """
        Consume transactions and detect fraud in real-time
        """
        for message in self.consumer:
            transaction = message.value
            
            # Detect fraud
            result = self.detect_fraud(transaction)
            
            # Send alert if high risk
            if result['risk_level'] in ['HIGH', 'CRITICAL']:
                self.producer.send('fraud_alerts', value=result)
                print(f"FRAUD ALERT: {result}")
```

### Feature Store Implementation

```python
import feast
from datetime import datetime, timedelta

class FraudFeatureStore:
    def __init__(self):
        self.store = feast.FeatureStore(repo_path=".")
    
    def get_online_features(self, user_id, transaction_time):
        """
        Retrieve pre-computed features for real-time inference
        """
        entity_rows = [{
            "user_id": user_id,
            "event_timestamp": transaction_time
        }]
        
        features = self.store.get_online_features(
            features=[
                "user_transaction_features:txn_count_24h",
                "user_transaction_features:avg_amount_7d",
                "user_transaction_features:unique_merchants_24h"
            ],
            entity_rows=entity_rows
        ).to_dict()
        
        return features
```

---

## 🚀 Advanced Topics

### 1. Graph Neural Networks for Fraud Rings

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FraudGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(FraudGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 2)  # Binary classification
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### 2. Federated Learning for Privacy-Preserving Fraud Detection

```python
import tensorflow as tf
import tensorflow_federated as tff

def create_federated_fraud_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn=create_federated_fraud_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.001)
)
```

### 3. Concept Drift Detection

```python
from skmultiflow.drift_detection import ADWIN, DDM

class DriftDetector:
    def __init__(self):
        self.adwin = ADWIN()
        self.performance_history = []
    
    def update(self, y_true, y_pred):
        """
        Monitor model performance for concept drift
        """
        accuracy = (y_true == y_pred).mean()
        self.adwin.add_element(accuracy)
        
        if self.adwin.detected_change():
            print("Concept drift detected! Retraining model...")
            return True
        
        return False
```

---

## 📊 Performance Benchmarks

### Industry Standards (Financial Institutions)

- **Precision**: 80-90% (minimize false positives)
- **Recall**: 85-95% (catch most fraud)
- **Latency**: < 100ms for real-time decisions
- **Throughput**: 10,000+ TPS (transactions per second)

---

## 🔗 Additional Resources

### Research Papers
1. "Credit Card Fraud Detection: A Realistic Modeling" (Dal Pozzolo et al., 2018)
2. "Deep Learning for Anomaly Detection: A Review" (Pang et al., 2021)
3. "Graph Neural Networks for Fraud Detection" (Wang et al., 2020)

### Datasets
1. **IEEE-CIS Fraud Detection** (Kaggle)
2. **Credit Card Fraud Detection** (Kaggle)
3. **PaySim Synthetic Financial Dataset**

### Certifications
- Certified Fraud Examiner (CFE)
- AWS Certified Machine Learning - Specialty
- Google Cloud Professional ML Engineer

---

## 🤝 Contributing

Contributions welcome! Please submit pull requests with:
- New feature engineering techniques
- Alternative model architectures
- Performance optimizations
- Industry case studies

---

## 📝 License

MIT License - See LICENSE file for details

---

**Last Updated**: January 2026
**Maintained By**: [Your Name]
**Contact**: [Your Email]