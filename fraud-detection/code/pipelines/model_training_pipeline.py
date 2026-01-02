"""Model Training Pipeline

End-to-end model training with cross-validation and hyperparameter tuning.
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

class ModelTrainingPipeline:
    """End-to-end model training pipeline."""
    
    def __init__(self, model, test_size=0.2, random_state=42):
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self, X, y):
        """Split data into train and test sets."""
        return train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
        return scores
    
    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        score = self.model.score(X_test, y_test)
        return score
