"""Gaussian Naive Bayes implementation with sklearn comparison."""

import pickle
import numpy as np
from typing import Dict, List, Tuple, Any
import random
import math
from collections import defaultdict

# Gaussian Naive Bayes Implementation
class GaussianNB:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.class_means = {}
        self.class_vars = {}
    
    def fit(self, X, y):
        """Train the Gaussian Naive Bayes classifier."""
        self.classes = list(set(y))
        n_samples = len(X)
        
        for cls in self.classes:
            # Get samples belonging to this class
            class_samples = [X[i] for i in range(n_samples) if y[i] == cls]
            n_class_samples = len(class_samples)
            
            # Calculate prior probability
            self.class_priors[cls] = n_class_samples / n_samples
            
            # Calculate mean and variance for each feature
            n_features = len(X[0])
            self.class_means[cls] = []
            self.class_vars[cls] = []
            
            for feat_idx in range(n_features):
                feature_values = [sample[feat_idx] for sample in class_samples]
                mean = sum(feature_values) / len(feature_values)
                variance = sum((x - mean) ** 2 for x in feature_values) / len(feature_values)
                
                # Add small epsilon to variance to avoid division by zero
                variance = max(variance, 1e-6)
                
                self.class_means[cls].append(mean)
                self.class_vars[cls].append(variance)
    
    def _gaussian_likelihood(self, x, mean, var):
        """Calculate Gaussian likelihood for a single feature."""
        eps = 1e-6
        coeff = 1.0 / math.sqrt(2 * math.pi * var + eps)
        exponent = math.exp(-(x - mean) ** 2 / (2 * var + eps))
        return coeff * exponent
    
    def _log_posterior(self, x):
        """Calculate log-posterior for each class."""
        log_posteriors = {}
        
        for cls in self.classes:
            # Start with log prior
            log_posterior = math.log(self.class_priors[cls] + 1e-10)
            
            # Add log likelihoods for each feature
            for feat_idx in range(len(x)):
                mean = self.class_means[cls][feat_idx]
                var = self.class_vars[cls][feat_idx]
                likelihood = self._gaussian_likelihood(x[feat_idx], mean, var)
                log_posterior += math.log(likelihood + 1e-10)
            
            log_posteriors[cls] = log_posterior
        
        return log_posteriors
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        predictions = []
        
        for x in X:
            log_posteriors = self._log_posterior(x)
            # Choose class with maximum log-posterior
            predicted_class = max(log_posteriors, key=log_posteriors.get)
            predictions.append(predicted_class)
        
        return predictions

    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        probabilities = []
        
        for x in X:
            log_posteriors = self._log_posterior(x)
            
            # Convert log-posteriors to probabilities using softmax
            max_log_posterior = max(log_posteriors.values())
            exp_log_posteriors = {cls: math.exp(lp - max_log_posterior) 
                                 for cls, lp in log_posteriors.items()}
            total = sum(exp_log_posteriors.values())
            
            probs = {cls: exp_log_posteriors[cls] / total for cls in self.classes}
            probabilities.append(probs)
        
        return probabilities

# Data generation
def generate_gaussian_data(n_samples=1000, n_features=4, n_classes=3):
    """Generate synthetic Gaussian data."""
    random.seed(42)
    np.random.seed(42)
    
    X = []
    y = []
    
    # Generate class centers with some separation
    class_centers = [[i * 3 + random.uniform(-0.5, 0.5) for _ in range(n_features)] 
                    for i in range(n_classes)]
    
    samples_per_class = n_samples // n_classes
    
    for cls in range(n_classes):
        center = class_centers[cls]
        # Generate samples around the class center
        for _ in range(samples_per_class):
            sample = [center[j] + random.gauss(0, 1.0) for j in range(n_features)]
            X.append(sample)
            y.append(cls)
    
    return X, y

def split_data(X, y, train_ratio=0.8):
    """Split data into training and validation sets."""
    n_samples = len(X)
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    split_idx = int(n_samples * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_val = [X[i] for i in val_indices]
    y_val = [y[i] for i in val_indices]
    
    return X_train, y_train, X_val, y_val

def evaluate(model, X, y):
    """Evaluate the model and return metrics."""
    predictions = model.predict(X)
    
    # Calculate accuracy
    correct = sum(1 for i in range(len(y)) if predictions[i] == y[i])
    accuracy = correct / len(y)
    
    # Calculate MSE (for continuous approximation)
    mse = sum((predictions[i] - y[i]) ** 2 for i in range(len(y))) / len(y)
    
    # Calculate R2 score
    mean_y = sum(y) / len(y)
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((y[i] - predictions[i]) ** 2 for i in range(len(y)))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'mse': mse,
        'r2': r2
    }

def save_artifacts(model, filepath):
    """Save model artifacts."""
    with open(filepath, 'wb') as f:
        pickle.dump({
            'classes': model.classes,
            'class_priors': model.class_priors,
            'class_means': model.class_means,
            'class_vars': model.class_vars,
        }, f)

