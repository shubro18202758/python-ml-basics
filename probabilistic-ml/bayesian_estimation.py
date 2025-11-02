"""Bayesian Estimation - Parameter Estimation with Bayesian Methods

This module demonstrates Bayesian parameter estimation techniques including:
- Prior, likelihood, and posterior distributions
- Maximum A Posteriori (MAP) estimation
- Bayesian updating with new evidence
- Conjugate priors (Beta-Binomial, Normal-Normal)
- Credible intervals
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

print("=== Bayesian Estimation Examples ===")

# Example 1: Bayesian Coin Flip (Beta-Binomial Conjugate Prior)
print("\n=== Example 1: Bayesian Coin Flip Estimation ===")
print("""
Scenario: Estimating the probability of heads for a coin.

Beta-Binomial Model:
- Prior: Beta(alpha, beta) - represents our initial belief
- Likelihood: Binomial(n, p) - observed data
- Posterior: Beta(alpha + heads, beta + tails)
""")

# Prior parameters (uniform prior: Beta(1,1))
alpha_prior = 1
beta_prior = 1

print(f"\nPrior: Beta({alpha_prior}, {beta_prior}) - uniform distribution")

# Simulate coin flips
n_flips = 10
true_p = 0.7  # True probability of heads (unknown in practice)
flips = np.random.binomial(1, true_p, n_flips)
heads = np.sum(flips)
tails = n_flips - heads

print(f"\nObserved data: {heads} heads, {tails} tails in {n_flips} flips")

# Posterior parameters
alpha_post = alpha_prior + heads
beta_post = beta_prior + tails

print(f"\nPosterior: Beta({alpha_post}, {beta_post})")

# Calculate posterior mean (Bayesian estimate)
posterior_mean = alpha_post / (alpha_post + beta_post)
print(f"Posterior mean (Bayesian estimate): {posterior_mean:.4f}")
print(f"True probability: {true_p:.4f}")
print(f"Frequentist estimate (MLE): {heads/n_flips:.4f}")

# Calculate 95% credible interval
credible_interval = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)
print(f"95% Credible Interval: [{credible_interval[0]:.4f}, {credible_interval[1]:.4f}]")

# Plot prior, likelihood, and posterior
x = np.linspace(0, 1, 1000)
prior = stats.beta.pdf(x, alpha_prior, beta_prior)
likelihood = stats.binom.pmf(heads, n_flips, x)
posterior = stats.beta.pdf(x, alpha_post, beta_post)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, prior, 'b-', linewidth=2)
plt.title('Prior Distribution')
plt.xlabel('Probability of Heads')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(x, likelihood, 'g-', linewidth=2)
plt.title('Likelihood')
plt.xlabel('Probability of Heads')
plt.ylabel('Likelihood')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(x, posterior, 'r-', linewidth=2)
plt.axvline(posterior_mean, color='black', linestyle='--', label=f'Mean: {posterior_mean:.3f}')
plt.axvline(credible_interval[0], color='gray', linestyle=':', label='95% CI')
plt.axvline(credible_interval[1], color='gray', linestyle=':')
plt.title('Posterior Distribution')
plt.xlabel('Probability of Heads')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bayesian_coin_flip.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'bayesian_coin_flip.png'")
plt.close()

# Example 2: Sequential Bayesian Updating
print("\n=== Example 2: Sequential Bayesian Updating ===")
print("Updating beliefs as new evidence arrives...\n")

# Start with uniform prior
alpha = 1
beta = 1

print(f"Initial prior: Beta({alpha}, {beta})")

# Simulate sequential observations
observations = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]  # 1=heads, 0=tails

for i, obs in enumerate(observations, 1):
    # Update parameters
    if obs == 1:
        alpha += 1
    else:
        beta += 1
    
    mean = alpha / (alpha + beta)
    print(f"After observation {i} ({'H' if obs else 'T'}): "
          f"Beta({alpha}, {beta}), Mean = {mean:.4f}")

print(f"\nFinal estimate: {mean:.4f}")

# Example 3: Normal-Normal Conjugate Prior (Estimating Mean)
print("\n=== Example 3: Bayesian Estimation of Normal Mean ===")
print("""
Scenario: Estimating the mean of a normal distribution with known variance.

Normal-Normal Conjugate:
- Prior: N(μ₀, σ₀²) - prior belief about mean
- Likelihood: N(μ, σ²) - observed data with known variance
- Posterior: N(μ_n, σ_n²)
""")

# Prior parameters
mu_prior = 0.0
sigma_prior = 10.0

print(f"\nPrior: N(μ={mu_prior}, σ={sigma_prior})")

# Generate data from true distribution
true_mu = 5.0
sigma_known = 2.0
n_samples = 20

data = np.random.normal(true_mu, sigma_known, n_samples)
data_mean = np.mean(data)

print(f"\nObserved: {n_samples} samples with mean = {data_mean:.4f}")
print(f"True mean: {true_mu:.4f}")

# Calculate posterior parameters
precision_prior = 1 / (sigma_prior ** 2)
precision_data = n_samples / (sigma_known ** 2)

precision_post = precision_prior + precision_data
mu_post = (precision_prior * mu_prior + precision_data * data_mean) / precision_post
sigma_post = np.sqrt(1 / precision_post)

print(f"\nPosterior: N(μ={mu_post:.4f}, σ={sigma_post:.4f})")
print(f"MLE estimate: {data_mean:.4f}")
print(f"Bayesian estimate (MAP): {mu_post:.4f}")

# 95% credible interval
ci_lower = mu_post - 1.96 * sigma_post
ci_upper = mu_post + 1.96 * sigma_post
print(f"95% Credible Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Example 4: MAP Estimation vs MLE
print("\n=== Example 4: MAP vs MLE Comparison ===")
print("""
Maximum A Posteriori (MAP) vs Maximum Likelihood Estimation (MLE):

MAP: arg max_θ P(θ|data) = arg max_θ P(data|θ) * P(θ)
     (incorporates prior information)

MLE: arg max_θ P(data|θ)
     (only uses observed data)
""")

# Simulate with different sample sizes
sample_sizes = [5, 10, 50, 100]
results = []

for n in sample_sizes:
    data = np.random.normal(true_mu, sigma_known, n)
    data_mean = np.mean(data)
    
    # MLE
    mle = data_mean
    
    # MAP (using same prior as before)
    precision_prior = 1 / (sigma_prior ** 2)
    precision_data = n / (sigma_known ** 2)
    precision_post = precision_prior + precision_data
    map_estimate = (precision_prior * mu_prior + precision_data * data_mean) / precision_post
    
    results.append({
        'n': n,
        'MLE': mle,
        'MAP': map_estimate,
        'True': true_mu
    })
    
    print(f"\nn={n:3d}: MLE={mle:.4f}, MAP={map_estimate:.4f}, True={true_mu:.4f}")

print("""
\nObservation:
- With few samples, MAP is closer to prior
- With many samples, MAP converges to MLE
- Prior has less influence as data increases
""")

# Example 5: Bayesian Inference for Proportion
print("\n=== Example 5: Bayesian A/B Testing ===")
print("""
Scenario: Comparing conversion rates of two website versions.
""")

# Version A data
n_A = 100
conversions_A = 12

# Version B data
n_B = 100
conversions_B = 18

print(f"\nVersion A: {conversions_A}/{n_A} conversions ({conversions_A/n_A:.2%})")
print(f"Version B: {conversions_B}/{n_B} conversions ({conversions_B/n_B:.2%})")

# Uniform priors for both
alpha_A = 1 + conversions_A
beta_A = 1 + (n_A - conversions_A)

alpha_B = 1 + conversions_B
beta_B = 1 + (n_B - conversions_B)

print(f"\nPosterior A: Beta({alpha_A}, {beta_A})")
print(f"Posterior B: Beta({alpha_B}, {beta_B})")

# Sample from posteriors to estimate P(B > A)
n_simulations = 100000
samples_A = np.random.beta(alpha_A, beta_A, n_simulations)
samples_B = np.random.beta(alpha_B, beta_B, n_simulations)

prob_B_better = np.mean(samples_B > samples_A)
print(f"\nP(Version B > Version A) = {prob_B_better:.4f}")

if prob_B_better > 0.95:
    print("Strong evidence that B is better than A")
elif prob_B_better > 0.90:
    print("Moderate evidence that B is better than A")
else:
    print("Insufficient evidence to conclude B is better")

print("\n=== Key Concepts in Bayesian Estimation ===")
print("""
1. Prior Distribution: Encodes prior knowledge/beliefs
2. Likelihood: Probability of data given parameters
3. Posterior Distribution: Updated beliefs after observing data
4. Conjugate Priors: Prior and posterior have same family
5. Credible Intervals: Bayesian alternative to confidence intervals
6. MAP Estimation: Mode of posterior distribution
7. Sequential Updates: Posterior becomes prior for new data

Advantages:
- Incorporates prior knowledge
- Provides full probability distribution
- Natural handling of uncertainty
- Intuitive interpretation

Challenges:
- Choice of prior can be subjective
- Computational complexity for complex models
- May require specialized algorithms (MCMC, etc.)
""")

print("\n=== All examples completed successfully! ===")
