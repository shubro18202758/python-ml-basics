"""Scikit-Learn Machine Learning Pipeline Examples

This module demonstrates fundamental scikit-learn concepts including:
- Train-test split and data preparation
- Model training and evaluation
- Cross-validation
- Pipeline creation
- Feature scaling and preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification, load_iris

# Set random seed for reproducibility
np.random.seed(42)

print("=== Scikit-Learn ML Pipeline Examples ===")

# Example 1: Basic Train-Test Split
print("\n=== Example 1: Train-Test Split ===")

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set: X_test={X_test.shape}, y_test={y_test.shape}")

# Example 2: Simple Logistic Regression Model
print("\n=== Example 2: Logistic Regression ===")

# Train logistic regression model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Example 3: Decision Tree Classifier
print("\n=== Example 3: Decision Tree Classifier ===")

# Train decision tree model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")

# Example 4: Feature Scaling with StandardScaler
print("\n=== Example 4: Feature Scaling ===")

# Create scaler
scaler = StandardScaler()

# Fit and transform training data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Original data mean: {X_train.mean():.4f}")
print(f"Scaled data mean: {X_train_scaled.mean():.4f}")
print(f"Original data std: {X_train.std():.4f}")
print(f"Scaled data std: {X_train_scaled.std():.4f}")

# Train model on scaled data
lr_scaled = LogisticRegression(random_state=42, max_iter=1000)
lr_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = lr_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
print(f"\nLogistic Regression Accuracy (with scaling): {accuracy_scaled:.4f}")

# Example 5: Creating a Pipeline
print("\n=== Example 5: ML Pipeline ===")

# Create pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred_pipeline = pipeline.predict(X_test)

# Evaluate pipeline
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)
print(f"Pipeline Accuracy: {accuracy_pipeline:.4f}")
print("Pipeline steps:", [step[0] for step in pipeline.steps])

# Example 6: Cross-Validation
print("\n=== Example 6: Cross-Validation ===")

# Perform 5-fold cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")
print(f"Std CV score: {cv_scores.std():.4f}")

# Example 7: Comparing Multiple Models
print("\n=== Example 7: Model Comparison ===")

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    
    # Test set performance
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_accuracy
    }
    
    print(f"\n{name}:")
    print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Test Accuracy: {test_accuracy:.4f}")

# Example 8: Working with Real Dataset (Iris)
print("\n=== Example 8: Iris Dataset Example ===")

# Load iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"Iris dataset shape: {X_iris.shape}")
print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")

# Split data
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# Create and train pipeline
iris_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

iris_pipeline.fit(X_train_iris, y_train_iris)

# Evaluate
y_pred_iris = iris_pipeline.predict(X_test_iris)
accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)

print(f"\nIris Classification Accuracy: {accuracy_iris:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test_iris, y_pred_iris)}")
print(f"\nClassification Report:\n{classification_report(y_test_iris, y_pred_iris, target_names=iris.target_names)}")

print("\n=== All examples completed successfully! ===")
