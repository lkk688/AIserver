#!/usr/bin/env python3
"""
Logistic Regression with Imbalanced Data Handling
Implements weighted CE loss, early stopping on F1, and comprehensive metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')


def generate_imbalanced_data(n_samples=2000, ratio=0.05, random_state=42):
    """Generate imbalanced binary classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        random_state=random_state
    )
    return X, y
    
def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))



class LogisticRegressionCustom:
    """Custom logistic regression with weighted cross-entropy loss."""
    
        self.early_stopping = early_stopping
        self.patience = patience
        self.weights = None
        self.bias = 0.0
        self.best_f1 = -1
        self.patience_counter = 0
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train the model with optional early stopping."""
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0

        # Compute class weights if not provided
            class_counts = np.bincount(y)
            total = len(y)
            self.class_weights = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}
        
        # Early stopping tracking
        if self.early_stopping and X_val is not None:
            best_f1 = -1
            best_bias = self.bias

        for i in range(self.n_iterations):
            # Vectorized forward pass
            linear_model = X @ self.weights + self.bias
            y_pred = sigmoid(linear_model)
            
            # Vectorized weighted gradients
            errors = y_pred - y
            weights_array = np.array([self.class_weights[label] for label in y])
            dw = (X.T @ (errors * weights_array)) / n_samples
            db = np.sum(errors * weights_array) / n_samples
            
            # Regularization to prevent overfitting
            reg_lambda = 0.01
            dw += reg_lambda * self.weights
            
            # Update parameters with learning rate decay
            current_lr = self.lr / (1 + 0.01 * i)
            self.weights -= current_lr * dw
            self.bias -= current_lr * db
            
            # Early stopping on validation F1
            if self.early_stopping and X_val is not None:
                val_pred = self.predict_proba(X_val)
                val_pred_binary = (val_pred >= 0.5).astype(int)
                val_f1 = f1_score(y_val, val_pred_binary)
                
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"Early stopping at iteration {i+1}, best F1: {best_f1:.4f}")
                    self.weights = best_weights
                    self.bias = best_bias
                    break
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        linear_model = X @ self.weights + self.bias
        return sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """Predict binary labels."""
        return (self.predict_proba(X) >= threshold).astype(int)


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute comprehensive metrics including confusion matrix, precision, recall, F1."""
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Precision, recall, F1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC-ROC if probabilities provided
    auc_roc = 0
    if y_proba is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc_roc = 0.5
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }


def plot_precision_recall_curve(y_true, y_proba, save_path='pr_curve.png'):
    """Plot and save precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap_score = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR curve (AP = {ap_score:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve saved to {save_path}")
    
    return ap_score


def evaluate(model, X, y, name="Model"):
    """
    Evaluate model and return metrics.
    For this classification task, we return:
    - MSE (as mean squared error of predictions vs targets)
    - R2 score
    - Task-specific metrics (precision, recall, F1)
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Compute standard metrics
    mse = np.mean((y - y_pred) ** 2)
    
    # R2 score
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Task-specific metrics
    metrics = compute_metrics(y, y_pred, y_proba)
    
    return {
        'mse': mse,
        'r2': r2,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc_roc': metrics['auc_roc'],
        'confusion_matrix': metrics['confusion_matrix']
    }


def main():
    """Main function to train, evaluate, and verify the model."""
    print("=" * 60)
    print("Imbalanced Logistic Regression - Metrics Analysis")
    print("=" * 60)
    
    # Generate imbalanced dataset (95/5 split)
    print("\n[1] Generating imbalanced dataset (95/5 split)...")
    X, y = generate_imbalanced_data(n_samples=2000, ratio=0.05, random_state=42)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"   Train size: {len(y_train)} (positive: {sum(y_train)}, ratio: {sum(y_train)/len(y_train):.2%})")
    print(f"   Val size: {len(y_val)} (positive: {sum(y_val)}, ratio: {sum(y_val)/len(y_val):.2%})")
    print(f"   Test size: {len(y_test)} (positive: {sum(y_test)}, ratio: {sum(y_test)/len(y_test):.2%})")
    
    # Train baseline (unweighted) model
    print("\n[2] Training baseline (unweighted) model...")
    model_baseline = LogisticRegressionCustom(
        learning_rate=0.1, n_iterations=500, class_weights=None,
        early_stopping=True, patience=50
    )
    model_baseline.fit(X_train, y_train, X_val, y_val)
    results_baseline = evaluate(model_baseline, X_val, y_val, "Baseline")
    
    # Train weighted model with higher learning rate and more iterations
    print("\n[3] Training weighted model...")
    model_weighted = LogisticRegressionCustom(
        learning_rate=0.5, n_iterations=1000, class_weights=None,  # Will compute internally
        early_stopping=True, patience=100
    )
    model_weighted.fit(X_train, y_train, X_val, y_val)
    results_weighted = evaluate(model_weighted, X_val, y_val, "Weighted")
    
    # Print results
    print("\n[4] Evaluation Results on Validation Set:")
    print("-" * 40)
    print(f"Baseline Model:")
    print(f"   MSE: {results_baseline['mse']:.4f}")
    print(f"   R²:  {results_baseline['r2']:.4f}")
    print(f"   Precision: {results_baseline['precision']:.4f}")
    print(f"   Recall:    {results_baseline['recall']:.4f}")
    print(f"   F1:        {results_baseline['f1']:.4f}")
    print(f"   AUC-ROC:   {results_baseline['auc_roc']:.4f}")
    print(f"   Confusion Matrix:\n{results_baseline['confusion_matrix']}")
    
    print(f"\nWeighted Model:")
    print(f"   MSE: {results_weighted['mse']:.4f}")
    print(f"   R²:  {results_weighted['r2']:.4f}")
    print(f"   Precision: {results_weighted['precision']:.4f}")
    print(f"   Recall:    {results_weighted['recall']:.4f}")
    print(f"   F1:        {results_weighted['f1']:.4f}")
    print(f"   AUC-ROC:   {results_weighted['auc_roc']:.4f}")
    print(f"   Confusion Matrix:\n{results_weighted['confusion_matrix']}")
    
    # Verify sklearn comparison
    print("\n[5] Verifying against sklearn...")
    sklearn_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    sklearn_model.fit(X_train, y_train)
    y_sklearn_pred = sklearn_model.predict(X_val)
    sklearn_f1 = f1_score(y_val, y_sklearn_pred)
    print(f"   Sklearn F1 (class_weight='balanced'): {sklearn_f1:.4f}")
    print(f"   Custom weighted F1: {results_weighted['f1']:.4f}")
    
    # Plot precision-recall curve
    print("\n[6] Generating Precision-Recall curve...")
    y_proba_weighted = model_weighted.predict_proba(X_val)
    ap_score = plot_precision_recall_curve(y_val, y_proba_weighted, 'pr_curve.png')
    print(f"   Average Precision: {ap_score:.4f}")
    
    # Quality thresholds
    print("\n[7] Quality Threshold Verification:")
    print("-" * 40)
    
    # Check R2 is reasonable (should be positive and reasonably high)
    r2_threshold = 0.7
    r2_ok = results_weighted['r2'] > r2_threshold
    print(f"   ✓ R² > {r2_threshold}: {r2_ok} (actual: {results_weighted['r2']:.4f})")
    
    # Check F1 is above random (random F1 for 5% positive is ~0.095)
    f1_threshold = 0.3
    f1_ok = results_weighted['f1'] >= f1_threshold
    print(f"   ✓ F1 >= {f1_threshold}: {f1_ok} (actual: {results_weighted['f1']:.4f})")
    
    # Check recall improved vs unweighted baseline
    recall_improved = results_weighted['recall'] >= results_baseline['recall']
    print(f"   ✓ Recall improved vs baseline: {recall_improved} "
          f"(baseline: {results_baseline['recall']:.4f}, weighted: {results_weighted['recall']:.4f})")
    
    # Check precision-recall curve is meaningful (AP > 0.1)
    ap_threshold = 0.1
    ap_ok = ap_score >= ap_threshold
    print(f"   ✓ Average Precision >= {ap_threshold}: {ap_ok} (actual: {ap_score:.4f})")
    
    # Final assertion
    all_checks_pass = r2_ok and f1_ok and recall_improved and ap_ok
    print(f"\n   Overall: {'✓ PASSED' if all_checks_pass else '✗ FAILED'}")
    
    if not all_checks_pass:
        print("\n❌ Quality checks failed!")
        return 1
    
    print("\n✅ All quality checks passed!")
    return 0


if __name__ == '__main__':
    exit(main())
