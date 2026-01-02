# Statistical Methods for Fraud Detection

## Probability Theory Fundamentals

### 1. Bayesian Approach to Fraud Detection

**Bayes' Theorem** is fundamental for updating fraud probabilities:

\[
P(\text{Fraud} | \text{Transaction}) = \frac{P(\text{Transaction} | \text{Fraud}) \cdot P(\text{Fraud})}{P(\text{Transaction})}
\]

**Components:**
- \(P(\text{Fraud})\): Prior probability (base fraud rate, typically 0.1-0.5%)
- \(P(\text{Transaction} | \text{Fraud})\): Likelihood (transaction pattern given fraud)
- \(P(\text{Fraud} | \text{Transaction})\): Posterior probability (updated fraud probability)

#### Practical Implementation

```python
import numpy as np
from scipy.stats import norm

class BayesianFraudDetector:
    def __init__(self, prior_fraud_rate=0.002):
        """
        Initialize Bayesian fraud detector
        
        Args:
            prior_fraud_rate: Historical fraud rate (e.g., 0.2%)
        """
        self.prior = prior_fraud_rate
        self.fraud_distribution = None
        self.legitimate_distribution = None
    
    def fit(self, X_fraud, X_legit):
        """
        Estimate distributions from training data
        
        Args:
            X_fraud: Feature values for fraudulent transactions
            X_legit: Feature values for legitimate transactions
        """
        # Estimate Gaussian distributions
        self.fraud_mean = np.mean(X_fraud, axis=0)
        self.fraud_std = np.std(X_fraud, axis=0)
        
        self.legit_mean = np.mean(X_legit, axis=0)
        self.legit_std = np.std(X_legit, axis=0)
    
    def calculate_likelihood(self, x, mean, std):
        """
        Calculate likelihood P(x | class) using Gaussian distribution
        """
        return np.prod(norm.pdf(x, loc=mean, scale=std + 1e-10))
    
    def predict_proba(self, X):
        """
        Calculate posterior probability of fraud
        
        P(Fraud | X) = P(X | Fraud) * P(Fraud) / P(X)
        """
        probabilities = []
        
        for x in X:
            # Calculate likelihoods
            likelihood_fraud = self.calculate_likelihood(x, self.fraud_mean, self.fraud_std)
            likelihood_legit = self.calculate_likelihood(x, self.legit_mean, self.legit_std)
            
            # Calculate marginal probability P(X)
            marginal = (likelihood_fraud * self.prior + 
                       likelihood_legit * (1 - self.prior))
            
            # Calculate posterior
            posterior_fraud = (likelihood_fraud * self.prior) / (marginal + 1e-10)
            
            probabilities.append(posterior_fraud)
        
        return np.array(probabilities)

# Example usage
detector = BayesianFraudDetector(prior_fraud_rate=0.002)
detector.fit(X_fraud_train, X_legit_train)
fraud_probabilities = detector.predict_proba(X_test)
```

---

### 2. Statistical Hypothesis Testing

#### Z-Test for Anomalous Amounts

**Null Hypothesis (H₀)**: Transaction amount is consistent with user's normal behavior

**Test Statistic:**
\[
z = \frac{x - \mu}{\sigma}
\]

Where:
- \(x\): Current transaction amount
- \(\mu\): User's historical average
- \(\sigma\): User's historical standard deviation

```python
import numpy as np
from scipy import stats

def zscore_anomaly_detection(transaction_amount, user_history, significance=0.001):
    """
    Detect anomalous transactions using z-score
    
    Args:
        transaction_amount: Current transaction amount
        user_history: Array of user's historical transactions
        significance: Significance level (default: 0.001 for 3-sigma rule)
    
    Returns:
        is_anomaly: Boolean indicating if transaction is anomalous
        z_score: Calculated z-score
        p_value: Probability of observing this value
    """
    mu = np.mean(user_history)
    sigma = np.std(user_history)
    
    # Calculate z-score
    z_score = (transaction_amount - mu) / (sigma + 1e-10)
    
    # Two-tailed test
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    is_anomaly = p_value < significance
    
    return is_anomaly, z_score, p_value

# Example
user_transactions = np.array([25, 30, 28, 32, 27, 29, 31])
new_transaction = 150

is_fraud, z, p = zscore_anomaly_detection(new_transaction, user_transactions)
print(f"Z-score: {z:.2f}, P-value: {p:.6f}, Fraud: {is_fraud}")
```

#### Chi-Square Test for Categorical Independence

**Use Case**: Test if transaction category is independent of fraud status

\[
\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
\]

```python
from scipy.stats import chi2_contingency
import pandas as pd

def test_category_fraud_independence(df):
    """
    Test if transaction category affects fraud probability
    """
    contingency_table = pd.crosstab(df['category'], df['is_fraud'])
    
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Degrees of freedom: {dof}")
    
    if p_value < 0.05:
        print("REJECT null hypothesis: Category and fraud are dependent")
    else:
        print("ACCEPT null hypothesis: Category and fraud are independent")
    
    return chi2, p_value
```

---

### 3. Time Series Analysis for Transaction Patterns

#### ARIMA for Transaction Frequency Prediction

**Autoregressive Integrated Moving Average** model:
\[
ARIMA(p, d, q): \quad y_t = c + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q} + \epsilon_t
\]

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def predict_transaction_frequency(transaction_counts, forecast_periods=24):
    """
    Predict expected transaction frequency to detect deviations
    
    Args:
        transaction_counts: Time series of transaction counts per hour
        forecast_periods: Number of hours to forecast
    """
    # Fit ARIMA model
    model = ARIMA(transaction_counts, order=(2, 1, 2))
    fitted_model = model.fit()
    
    # Forecast
    forecast = fitted_model.forecast(steps=forecast_periods)
    confidence_intervals = fitted_model.get_forecast(steps=forecast_periods).conf_int()
    
    return forecast, confidence_intervals

def detect_frequency_anomalies(actual_counts, expected_counts, confidence_intervals):
    """
    Flag periods where transaction frequency is outside confidence interval
    """
    anomalies = []
    
    for i, (actual, expected, (lower, upper)) in enumerate(
        zip(actual_counts, expected_counts, confidence_intervals)):
        
        if actual < lower or actual > upper:
            anomalies.append({
                'time_period': i,
                'actual': actual,
                'expected': expected,
                'deviation': abs(actual - expected)
            })
    
    return anomalies
```

#### Seasonal Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def analyze_transaction_seasonality(df, freq='D'):
    """
    Decompose transaction patterns into trend, seasonal, and residual components
    """
    # Aggregate transactions by time period
    ts = df.groupby(pd.Grouper(key='timestamp', freq=freq))['amount'].sum()
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(ts, model='additive', period=7)  # Weekly seasonality
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # Flag large residuals as potential fraud periods
    residual_zscore = (residual - residual.mean()) / residual.std()
    anomalous_periods = residual_zscore[abs(residual_zscore) > 3]
    
    return trend, seasonal, residual, anomalous_periods
```

---

### 4. Probability Distributions for Fraud Modeling

#### Poisson Distribution for Event Counting

**Model transaction counts in fixed time windows:**

\[
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
\]

```python
from scipy.stats import poisson

def poisson_anomaly_score(observed_count, historical_mean):
    """
    Calculate anomaly score based on Poisson distribution
    
    Args:
        observed_count: Number of transactions in current window
        historical_mean: Average transactions per window (λ)
    """
    # Probability of observing this many or more transactions
    p_value = 1 - poisson.cdf(observed_count - 1, mu=historical_mean)
    
    # Convert to anomaly score (higher = more anomalous)
    anomaly_score = -np.log(p_value + 1e-10)
    
    return anomaly_score, p_value

# Example: User typically makes 3 transactions per day
historical_lambda = 3
observed_today = 15

score, p = poisson_anomaly_score(observed_today, historical_lambda)
print(f"Anomaly score: {score:.2f}, P-value: {p:.6f}")
```

#### Exponential Distribution for Inter-Transaction Times

**Model time between consecutive transactions:**

\[
f(x | \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0
\]

```python
from scipy.stats import expon

def analyze_transaction_velocity(transaction_times):
    """
    Detect unusually fast transaction sequences
    """
    # Calculate inter-arrival times
    inter_arrival_times = np.diff(transaction_times)
    
    # Fit exponential distribution
    rate = 1 / np.mean(inter_arrival_times)
    
    # Calculate probability for each inter-arrival time
    probabilities = expon.pdf(inter_arrival_times, scale=1/rate)
    
    # Flag unusually short intervals (low probability)
    suspicious_intervals = inter_arrival_times[probabilities < 0.01]
    
    return suspicious_intervals, rate
```

---

### 5. Multivariate Statistical Methods

#### Mahalanobis Distance

**Measure distance accounting for correlations:**

\[
D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
\]

Where:
- \(\mu\): Mean vector
- \(\Sigma\): Covariance matrix

```python
from scipy.spatial.distance import mahalanobis
import numpy as np

class MahalanobisAnomalyDetector:
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.mean = None
        self.cov_inv = None
    
    def fit(self, X):
        """
        Learn normal transaction distribution
        """
        self.mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        
        # Add regularization to handle singular matrices
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
        self.cov_inv = np.linalg.inv(cov_reg)
    
    def predict(self, X):
        """
        Detect anomalies based on Mahalanobis distance
        """
        distances = []
        
        for x in X:
            dist = mahalanobis(x, self.mean, self.cov_inv)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Convert to chi-square probability
        from scipy.stats import chi2
        p_values = 1 - chi2.cdf(distances**2, df=len(self.mean))
        
        anomalies = distances > self.threshold
        
        return anomalies, distances, p_values

# Usage
detector = MahalanobisAnomalyDetector(threshold=3.0)
detector.fit(X_train_legit)
anomalies, distances, p_values = detector.predict(X_test)
```

#### Principal Component Analysis (PCA)

**Dimensionality reduction for fraud detection:**

```python
from sklearn.decomposition import PCA
import numpy as np

class PCAAnomalyDetector:
    def __init__(self, n_components=0.95):
        """
        Args:
            n_components: Number of components or variance ratio to retain
        """
        self.pca = PCA(n_components=n_components)
        self.threshold = None
    
    def fit(self, X):
        """
        Fit PCA on normal transactions
        """
        # Transform to principal components
        X_transformed = self.pca.fit_transform(X)
        
        # Calculate reconstruction error
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        reconstruction_errors = np.sum((X - X_reconstructed)**2, axis=1)
        
        # Set threshold at 99th percentile
        self.threshold = np.percentile(reconstruction_errors, 99)
        
        return self
    
    def predict(self, X):
        """
        Detect anomalies based on reconstruction error
        """
        X_transformed = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        
        reconstruction_errors = np.sum((X - X_reconstructed)**2, axis=1)
        
        anomalies = reconstruction_errors > self.threshold
        
        return anomalies, reconstruction_errors

# Usage
pca_detector = PCAAnomalyDetector(n_components=0.95)
pca_detector.fit(X_train_legit)
anomalies, errors = pca_detector.predict(X_test)
```

---

### 6. Information Theory Metrics

#### Entropy-Based Anomaly Detection

**Shannon Entropy measures uncertainty:**

\[
H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)
\]

```python
import numpy as np
from scipy.stats import entropy

def calculate_transaction_entropy(transactions, feature='merchant_id'):
    """
    Low entropy = predictable behavior
    High entropy = diverse/random behavior (potential fraud)
    """
    value_counts = transactions[feature].value_counts()
    probabilities = value_counts / len(transactions)
    
    # Calculate Shannon entropy
    H = entropy(probabilities, base=2)
    
    return H

def entropy_based_fraud_score(user_transactions, window='24H'):
    """
    Compare entropy in recent window vs historical baseline
    """
    # Historical entropy (baseline)
    historical_entropy = calculate_transaction_entropy(user_transactions[:-24])
    
    # Recent entropy
    recent_entropy = calculate_transaction_entropy(user_transactions[-24:])
    
    # Relative entropy increase (fraud indicator)
    entropy_ratio = recent_entropy / (historical_entropy + 1e-10)
    
    return entropy_ratio
```

#### Kullback-Leibler Divergence

**Measure distribution shift:**

\[
D_{KL}(P || Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}
\]

```python
from scipy.stats import entropy as kl_divergence

def detect_distribution_shift(historical_dist, current_dist):
    """
    Detect if current transaction distribution deviates from historical
    
    Args:
        historical_dist: Distribution of feature values (historical)
        current_dist: Distribution of feature values (current period)
    """
    # Ensure same support
    all_values = set(historical_dist.index) | set(current_dist.index)
    
    P = np.array([historical_dist.get(v, 1e-10) for v in all_values])
    Q = np.array([current_dist.get(v, 1e-10) for v in all_values])
    
    # Normalize
    P = P / P.sum()
    Q = Q / Q.sum()
    
    # Calculate KL divergence
    kl_div = kl_divergence(P, Q)
    
    return kl_div

# Example: Detect shift in transaction amount distribution
historical_amounts = user_data[user_data['timestamp'] < '2025-01-01']['amount']
current_amounts = user_data[user_data['timestamp'] >= '2025-01-01']['amount']

hist_dist, _ = np.histogram(historical_amounts, bins=20, density=True)
curr_dist, _ = np.histogram(current_amounts, bins=20, density=True)

shift_score = detect_distribution_shift(hist_dist, curr_dist)
print(f"Distribution shift score: {shift_score:.4f}")
```

---

### 7. Correlation Analysis

#### Pearson Correlation for Feature Selection

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_fraud_correlations(df, target='is_fraud'):
    """
    Identify features most correlated with fraud
    """
    # Calculate correlations
    correlations = df.corr()[target].sort_values(ascending=False)
    
    # Remove self-correlation
    correlations = correlations.drop(target)
    
    # Visualize
    plt.figure(figsize=(10, 8))
    correlations.plot(kind='barh')
    plt.title('Feature Correlations with Fraud')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.show()
    
    return correlations

# Identify highly correlated features for removal
def remove_multicollinearity(df, threshold=0.9):
    """
    Remove redundant features with high correlation
    """
    corr_matrix = df.corr().abs()
    
    # Upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation > threshold
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]
    
    return df.drop(columns=to_drop)
```

---

### 8. Confidence Intervals for Fraud Thresholds

```python
from scipy import stats

def calculate_confidence_interval(fraud_scores, confidence=0.95):
    """
    Calculate confidence interval for fraud threshold
    """
    mean = np.mean(fraud_scores)
    sem = stats.sem(fraud_scores)  # Standard error of mean
    
    # Calculate confidence interval
    ci = stats.t.interval(confidence, len(fraud_scores)-1, 
                          loc=mean, scale=sem)
    
    return ci

def adaptive_threshold(legitimate_scores, false_positive_rate=0.01):
    """
    Set fraud threshold to achieve target false positive rate
    """
    threshold = np.percentile(legitimate_scores, (1 - false_positive_rate) * 100)
    
    return threshold
```

---

## Summary of Key Statistical Methods

| Method | Use Case | Strengths | Limitations |
|--------|----------|-----------|-------------|
| **Bayes' Theorem** | Probability updating | Incorporates prior knowledge | Assumes independence |
| **Z-Score** | Simple anomaly detection | Fast, interpretable | Assumes normal distribution |
| **Chi-Square** | Categorical analysis | Tests independence | Requires sufficient samples |
| **ARIMA** | Time series prediction | Captures temporal patterns | Stationary data required |
| **Mahalanobis** | Multivariate anomalies | Accounts for correlations | Sensitive to outliers |
| **Entropy** | Behavior diversity | Distribution-free | Hard to interpret |

---

## Practice Exercises

1. **Exercise 1**: Implement Bayesian updating for fraud probability as new evidence arrives
2. **Exercise 2**: Use chi-square test to identify which transaction features are most predictive of fraud
3. **Exercise 3**: Build an ARIMA model to predict hourly transaction volumes and flag anomalies
4. **Exercise 4**: Compare Euclidean vs Mahalanobis distance for multi-feature fraud detection

---

## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
2. James, G., et al. (2013). *An Introduction to Statistical Learning*. Springer.
3. Aggarwal, C. C. (2017). *Outlier Analysis*. Springer.
