#!/usr/bin/env python3
"""
Multinomial Naive Bayes Text Classification
Implements NB with Laplace smoothing for text classification on 20newsgroups dataset.
"""

import os
import sys
import numpy as np
from collections import defaultdict

# Set seed for reproducibility
def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)

# Get device (CPU/CUDA)
def get_device():
    """Get computation device (prefer CUDA if available)."""
    return 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', '') else 'cpu'

try:
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    DEVICE = 'cpu'

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error, r2_score


class MultinomialNB:
    """Multinomial Naive Bayes classifier with Laplace smoothing."""
    
    def __init__(self, alpha=1.0):
        """
        Initialize classifier.
        
        Args:
            alpha: Laplace smoothing parameter (default: 1.0)
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_priors_ = None
        self.feature_probs_ = None
        self.n_features_ = None
    
    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Args:
            X: Sparse matrix of shape (n_samples, n_features)
            y: Array of shape (n_samples,) with class labels
        """
        self.classes_ = np.unique(y)
        n_samples = len(y)
        n_features = X.shape[1]
        self.n_features_ = n_features
        
        # Calculate class priors with Laplace smoothing
        self.class_priors_ = {}
        for c in self.classes_:
            class_count = np.sum(y == c)
            self.class_priors_[c] = (class_count + self.alpha) / (n_samples + self.alpha * len(self.classes_))
        
        # Calculate feature probabilities for each class
        self.feature_probs_ = {}
        for c in self.classes_:
            # Get indices of samples belonging to class c
            class_mask = (y == c)
            X_class = X[class_mask]
            
            # Sum feature counts for class c
            feature_sums = np.array(X_class.sum(axis=0)).flatten()
            
            # Apply Laplace smoothing and normalize
            total_count = feature_sums.sum() + self.alpha * n_features
            self.feature_probs_[c] = (feature_sums + self.alpha) / total_count
        
        return self
    
    def _log_likelihood(self, x, c):
        """
        Compute log likelihood for a single sample and class.
        
        Args:
            x: Feature vector (sparse or dense)
            c: Class label
            
        Returns:
            Log likelihood value
        """
        if hasattr(x, 'toarray'):
            features = x.toarray().flatten()
        else:
            features = np.array(x).flatten()
        
        probs = self.feature_probs_[c]
        
        # Compute log probability in log space to avoid underflow
        log_prob = 0.0
        for i, feat_count in enumerate(features):
            if feat_count > 0:
                log_prob += feat_count * np.log(probs[i])
        
        return log_prob
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Args:
            X: Sparse matrix of shape (n_samples, n_features)
            
        Returns:
            Array of predicted class labels
        """
        predictions = []
        
        for sample in X:
            class_scores = {}
            
            for c in self.classes_:
                # Log prior
                log_prior = np.log(self.class_priors_[c])
                # Log likelihood
                log_likelihood = self._log_likelihood(sample, c)
                # Log posterior (unnormalized)
                class_scores[c] = log_prior + log_likelihood
            
            # Choose class with highest score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def score(self, X, y):
        """
        Compute accuracy score.
        
        Args:
            X: Sparse matrix of shape (n_samples, n_features)
            y: Array of true class labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def compute_metrics(y_true, y_pred, target_names):
    """
    Compute evaluation metrics including MSE, R2, and F1.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        target_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    # Convert string labels to indices for MSE/R2 calculation
    label_to_idx = {name: idx for idx, name in enumerate(target_names)}
    y_true_idx = np.array([label_to_idx[label] for label in y_true])
    y_pred_idx = np.array([label_to_idx[label] for label in y_pred])
    
    # Compute metrics
    mse = mean_squared_error(y_true_idx, y_pred_idx)
    r2 = r2_score(y_true_idx, y_pred_idx)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    return {
        'mse': mse,
        'r2': r2,
        'f1_macro': f1_macro
    }


def sanity_check(model, target_names):
    """
    Perform sanity checks on the model.
    
    Args:
        model: Trained MultinomialNB model
        target_names: List of class names
        
    Returns:
        Dictionary of sanity check results
    """
    results = {}
    
    # Check 1: Priors sum to approximately 1
    prior_sum = sum(model.class_priors_.values())
    results['priors_sum_to_one'] = abs(prior_sum - 1.0) < 0.01
    
    # Check 2: All priors are positive
    results['all_priors_positive'] = all(p > 0 for p in model.class_priors_.values())
    
    # Check 3: Feature probabilities are between 0 and 1
    all_valid_probs = True
    for c in model.classes_:
        probs = model.feature_probs_[c]
        if not (np.all(probs >= 0) and np.all(probs <= 1)):
            all_valid_probs = False
    results['all_feature_probs_valid'] = all_valid_probs
    
    # Check 4: Model has been trained (has classes_)
    results['model_trained'] = model.classes_ is not None
    
    return results


def main():
    """Main function to run the Multinomial Naive Bayes text classification."""
    print("=" * 60)
    print("Multinomial Naive Bayes Text Classification")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Load dataset
    print("\nLoading 20 newsgroups dataset (subset)...")
    categories = ['comp.graphics', 'rec.sport.baseball', 'sci.med', 'talk.politics.mideast']
    
    try:
        dataset = fetch_20newsgroups(
            subset='all',
            categories=categories,
            shuffle=True,
            random_state=42,
            remove=('headers', 'footers', 'quotes')
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using small toy corpus instead...")
        # Fallback to small toy corpus
        dataset = type('obj', (object,), {
            'data': [
                'computer graphics rendering 3d',
                'baseball game score run',
                'medical doctor hospital treatment',
                'politics war peace middle east',
                'computer programming code software',
                'baseball pitcher hit home run',
                'medical disease doctor prescription',
                'politics government election vote',
                'computer algorithm data network',
                'baseball bat ball field',
                'medical surgery patient recovery',
                'politics law constitution rights'
            ] * 50,
            'target': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] * 50,
            'target_names': categories
        })()
    
    print(f"Total samples: {len(dataset.data)}")
    print(f"Categories: {dataset.target_names}")
    
    # Split data
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target
    )
    
    print(f"Training set: {len(X_train_text)} samples")
    print(f"Validation set: {len(X_val_text)} samples")
    
    # Convert text to feature vectors using bag-of-words
    print("\nConverting text to bag-of-words features...")
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    
    target_names = dataset.target_names
    
    # Train model
    print("\nTraining Multinomial Naive Bayes classifier...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, [target_names[y] for y in y_train])
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_pred = model.predict(X_train)
    train_metrics = compute_metrics([target_names[y] for y in y_train], train_pred, target_names)
    print(f"Training MSE: {train_metrics['mse']:.4f}")
    print(f"Training R²: {train_metrics['r2']:.4f}")
    print(f"Training F1 Macro: {train_metrics['f1_macro']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_pred = model.predict(X_val)
    val_metrics = compute_metrics([target_names[y] for y in y_val], val_pred, target_names)
    print(f"Validation MSE: {val_metrics['mse']:.4f}")
    print(f"Validation R²: {val_metrics['r2']:.4f}")
    print(f"Validation F1 Macro: {val_metrics['f1_macro']:.4f}")
    
    # Sanity checks
    print("\nRunning sanity checks...")
    sanity_results = sanity_check(model, target_names)
    for check, passed in sanity_results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}: {passed}")
    
    # Quality thresholds
    print("\n" + "-" * 40)
    try:
        assert val_metrics['mse'] < 1.0, f"Validation MSE {val_metrics['mse']:.4f} > 1.0"
        assert val_metrics['r2'] > 0.0, f"Validation R² {val_metrics['r2']:.4f} <= 0"
        assert val_metrics['f1_macro'] > 0.5, f"Validation F1 {val_metrics['f1_macro']:.4f} <= 0.5"
        assert all(sanity_results.values()), "Sanity checks failed"
        
        print("✓ All quality thresholds passed!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"✗ Quality check failed: {e}")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
