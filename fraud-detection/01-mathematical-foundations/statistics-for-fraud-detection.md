# Statistical Methods for Fraud Detection

## Probability Theory Fundamentals

### 1. Bayesian Approach to Fraud Detection

**Introduction to Bayesian Methods**

Bayesian inference is a cornerstone of modern fraud detection systems. Unlike frequentist approaches that rely solely on observed data, Bayesian methods incorporate prior knowledge about fraud rates and update beliefs as new evidence arrives. This probabilistic framework is particularly valuable in fraud detection because it naturally handles uncertainty and can incorporate domain expertise about known fraud patterns.

The fundamental principle underlying Bayesian fraud detection is **Bayes' Theorem**, which provides a mathematical framework for calculating the probability that a transaction is fraudulent given observed characteristics. In fraud detection contexts, we use Bayes' theorem to compute the posterior probability—the updated probability of fraud after observing transaction features.

**Bayes' Theorem Formula:**

The mathematical expression of Bayes' theorem for fraud detection is:

$$P(\text{Fraud} | \text{Transaction}) = \frac{P(\text{Transaction} | \text{Fraud}) \cdot P(\text{Fraud})}{P(\text{Transaction})}$$

**Component Interpretation:**

1. **Prior Probability** $P(\text{Fraud})$: The baseline probability that a transaction is fraudulent, typically ranging from 0.1% to 0.5% in real-world scenarios. This reflects our initial belief about fraud rates before observing any transaction details. For example, if historical data shows that 2 out of 1,000 transactions are fraudulent, then $P(\text{Fraud}) = 0.002$.

2. **Likelihood** $P(\text{Transaction} | \text{Fraud})$: The probability of observing these specific transaction characteristics given that the transaction is fraudulent. This likelihood is estimated from historical fraudulent transactions and represents patterns typical of fraud (e.g., unusual transaction amounts, geographic anomalies, velocity patterns).

3. **Evidence** $P(\text{Transaction})$: The total probability of observing this transaction across all possibilities (both fraudulent and legitimate). This acts as a normalizing constant:

$$P(\text{Transaction}) = P(\text{Transaction} | \text{Fraud}) \cdot P(\text{Fraud}) + P(\text{Transaction} | \text{Legitimate}) \cdot P(\text{Legitimate})$$

4. **Posterior Probability** $P(\text{Fraud} | \text{Transaction})$: The updated probability that this specific transaction is fraudulent after considering all evidence. This is our final fraud risk score that drives decision-making (e.g., blocking, reviewing, or approving transactions).

**Why Bayesian Methods Matter for Fraud Detection:**

- **Incorporates Domain Knowledge**: Prior probabilities can be set based on institutional knowledge and historical fraud patterns
- **Handles Uncertainty**: Provides probabilistic scores rather than binary classifications
- **Adaptive Learning**: Can be updated as new fraud patterns emerge
- **Interpretable Results**: Fraud probability is intuitive and actionable for business stakeholders

### 2. Probability Distributions in Fraud Modeling

**Normal Distribution (Gaussian Distribution)**

The normal distribution is fundamental in statistical fraud detection because many transaction features approximate normality. Its probability density function is:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

Where:
- $\mu$ is the mean (expected value)
- $\sigma$ is the standard deviation (spread)

**Application in Fraud Detection**: Transaction amounts typically follow a normal distribution for legitimate users. Deviations beyond 2-3 standard deviations often indicate anomalies. For example, if a customer's transaction amounts have a mean of \$100 with std dev of \$20, a \$200 transaction would be 5 standard deviations away, triggering fraud alerts.

**Poisson Distribution**

The Poisson distribution models the number of events occurring within a fixed time interval. Its probability mass function is:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Where:
- $\lambda$ is the expected number of events (transactions) in the interval
- $k$ is the observed number of events

**Application**: Transaction frequency per hour/day follows Poisson patterns. Sudden spikes (e.g., 20 transactions in 5 minutes when typical rate is 2 per hour) signal potential card compromise or bot attacks.

**Exponential Distribution**

The exponential distribution models the time between successive events:

$$f(x) = \lambda e^{-\lambda x}$$

Where $x \geq 0$ and $\lambda > 0$.

**Application**: Legitimate transactions typically have exponential inter-arrival times—time between successive transactions varies naturally. Sudden clustering (transactions arriving every 10 seconds) is anomalous and fraud-indicative.

### 3. Statistical Hypothesis Testing for Fraud Detection

**Z-Test for Anomalies**

The Z-test measures how many standard deviations a value is from the mean:

$$Z = \frac{x - \mu}{\sigma}$$

Where:
- $x$ is the observed value (transaction amount)
- $\mu$ is the mean of legitimate transactions
- $\sigma$ is the standard deviation

Values with $|Z| > 3$ are extremely unusual (99.7% of normal data falls within 3$\sigma$).

**Chi-Square Test for Independence**

The chi-square statistic tests whether observed categorical frequencies differ from expected:

$$\chi^2 = \sum_{i=1}^{n} \frac{(O_i - E_i)^2}{E_i}$$

Where:
- $O_i$ = observed frequency for category $i$
- $E_i$ = expected frequency for category $i$

**Application**: Test if merchant category distribution differs between fraudulent and legitimate transactions. High $\chi^2$ values indicate fraud-specific patterns.

**Kolmogorov-Smirnov Test**

The KS-test measures the maximum distance between two cumulative distribution functions:

$$D = \max_x |F_1(x) - F_2(x)|$$

Where $F_1$ and $F_2$ are cumulative distributions.

**Application**: Compare transaction amount distributions before and after a suspected compromise. Significant KS-statistic indicates the distributions differ substantially.

### 4. Entropy and Information Theory

**Information Entropy**

Entropy measures uncertainty or randomness in a probability distribution:

$$H(X) = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

Where $p_i$ is the probability of event $i$.

**Application**: High entropy in merchant choices suggests normal behavior (varied spending patterns). Low entropy (always same merchant) may indicate account compromise or restricted trading.

**Kullback-Leibler Divergence (KL Divergence)**

KL divergence measures how one probability distribution diverges from a reference distribution:

$$D_{KL}(P || Q) = \sum_{x} P(x) \log\left(\frac{P(x)}{Q(x)}\right)$$

Where:
- $P$ is the observed distribution (current user behavior)
- $Q$ is the reference distribution (historical legitimate behavior)

**Application**: Quantify behavioral shift. Large KL divergence indicates the user's current behavior significantly deviates from historical patterns, suggesting compromise or account takeover.

## Practical Implementation Considerations

**Handling Imbalanced Data**: Fraud datasets are naturally imbalanced (fraud rate ~0.1-0.5%). Statistical methods must account for this through:
- Adjusted priors reflecting true fraud rates
- Class weights in likelihood calculations
- Stratified validation procedures

**Real-Time Requirements**: Fraud detection systems must score transactions in milliseconds. This requires:
- Precomputed sufficient statistics (means, standard deviations, distributions)
- Efficient likelihood calculations
- Caching of distribution parameters

**Concept Drift**: User behavior and fraud patterns evolve over time. Adaptive systems must:
- Periodically retrain on recent data
- Track distribution parameter changes
- Adjust priors as fraud rates shift

## References

- Phua, C., Lee, V., Smith, K., & Gayler, R. (2010). A comprehensive survey of data mining-based fraud detection research. arXiv preprint arXiv:1009.6119.
- Bolton, R. J., & Hand, D. J. (2002). Statistical fraud detection: a review. Statistical science, 17(3), 235-249.
- Goldstein, M., & Uchida, S. (2016). A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data. PLoS one, 11(4), e0152173.
