"""Gaussian Naive Bayes - Probabilistic Classification

This module demonstrates Gaussian Naive Bayes algorithm, which is based on
Bayes' theorem with the assumption of conditional independence between features.

Bayes' Theorem: P(y|X) = P(X|y) * P(y) / P(X)

For Gaussian Naive Bayes:
- Features are assumed to follow a Gaussian (normal) distribution
- Class conditional probability: P(x_i|y) ~ N(μ_y, σ_y²)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

print("=== Gaussian Naive Bayes Examples ===")

# Example 1: Understanding Bayes' Theorem
print("\n=== Example 1: Bayes' Theorem Basics ===")
print("""\nBayes' Theorem Formula:\nP(Class|Features) = P(Features|Class) * P(Class) / P(Features)\n\nWhere:\n- P(Class|Features): Posterior probability (what we want to find)
- P(Features|Class): Likelihood (probability of features given class)
- P(Class): Prior probability (probability of class)
- P(Features): Evidence (probability of features)\n""")

# Example 2: Simple Gaussian Naive Bayes on Iris Dataset
print("\n=== Example 2: Iris Dataset Classification ===")

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"Dataset shape: {X.shape}")
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Create and train Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")

# Example 3: Probability Predictions
print("\n=== Example 3: Probability Predictions ===")

# Get probability predictions
proba_predictions = gnb.predict_proba(X_test[:5])

print("\nFirst 5 test samples - Probability predictions:")
for i, (true_label, pred_label, probas) in enumerate(zip(
    y_test[:5], y_pred[:5], proba_predictions
)):
    print(f"\nSample {i+1}:")
    print(f"  True class: {iris.target_names[true_label]}")
    print(f"  Predicted class: {iris.target_names[pred_label]}")
    print(f"  Probabilities:")
    for j, class_name in enumerate(iris.target_names):
        print(f"    {class_name}: {probas[j]:.4f}")

# Example 4: Understanding Model Parameters
print("\n=== Example 4: Model Parameters ===")

print("\nClass prior probabilities:")
for i, class_name in enumerate(iris.target_names):
    print(f"  P({class_name}) = {np.exp(gnb.class_log_prior_[i]):.4f}")

print("\nMean (μ) for each feature per class:")
for i, class_name in enumerate(iris.target_names):
    print(f"\n  {class_name}:")
    for j, feature_name in enumerate(iris.feature_names):
        print(f"    {feature_name}: μ = {gnb.theta_[i, j]:.4f}")

print("\nVariance (σ²) for each feature per class:")
for i, class_name in enumerate(iris.target_names):
    print(f"\n  {class_name}:")
    for j, feature_name in enumerate(iris.feature_names):
        print(f"    {feature_name}: σ² = {gnb.var_[i, j]:.4f}")

# Example 5: Custom Binary Classification Problem
print("\n=== Example 5: Binary Classification ===")

# Generate synthetic binary classification data
X_binary, y_binary = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=42
)

# Split data
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

# Train model
gnb_binary = GaussianNB()
gnb_binary.fit(X_train_bin, y_train_bin)

# Evaluate
y_pred_bin = gnb_binary.predict(X_test_bin)
accuracy_bin = accuracy_score(y_test_bin, y_pred_bin)

print(f"Binary Classification Accuracy: {accuracy_bin:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test_bin, y_pred_bin)}")

# Example 6: Handling New Data
print("\n=== Example 6: Predicting New Samples ===")

# Create new samples similar to iris data
new_samples = np.array([
    [5.0, 3.5, 1.5, 0.2],  # Similar to Setosa
    [6.5, 3.0, 5.5, 1.8],  # Similar to Virginica
    [5.5, 2.5, 4.0, 1.3]   # Similar to Versicolor
])

print("\nNew samples to classify:")
for i, sample in enumerate(new_samples):
    prediction = gnb.predict([sample])[0]
    probabilities = gnb.predict_proba([sample])[0]
    
    print(f"\nSample {i+1}: {sample}")
    print(f"  Predicted class: {iris.target_names[prediction]}")
    print(f"  Confidence: {probabilities[prediction]:.4f}")
    print(f"  All probabilities: {dict(zip(iris.target_names, probabilities))}")

# Example 7: Comparison with Different Prior
print("\n=== Example 7: Custom Class Priors ===")

# Default GNB (learns priors from data)
gnb_default = GaussianNB()
gnb_default.fit(X_train, y_train)
acc_default = accuracy_score(y_test, gnb_default.predict(X_test))

# GNB with custom uniform priors
gnb_uniform = GaussianNB(priors=[1/3, 1/3, 1/3])
gnb_uniform.fit(X_train, y_train)
acc_uniform = accuracy_score(y_test, gnb_uniform.predict(X_test))

print(f"\nDefault priors accuracy: {acc_default:.4f}")
print(f"Uniform priors accuracy: {acc_uniform:.4f}")

print("\n=== Key Advantages of Gaussian Naive Bayes ===")
print("""
1. Fast training and prediction
2. Works well with small datasets
3. Handles multi-class problems naturally
4. Provides probabilistic predictions
5. Simple and interpretable

Limitations:
1. Assumes feature independence ("naive" assumption)
2. Assumes Gaussian distribution of features
3. Can be outperformed by more complex models on large datasets
""")

print("\n=== All examples completed successfully! ===")
