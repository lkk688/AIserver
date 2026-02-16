#!/usr/bin/env python3
"""
Anomaly Detection using Z-score and IQR - Level 1
Task: Implement evaluate() returning MSE, R2, and metrics
Implementation using Z-score and IQR methods for anomaly detection
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def generate_data(num_samples=500, num_features=3, anomaly_ratio=0.1):
    """Generate synthetic data with anomalies."""
    # Generate normal data
    X = np.random.randn(num_samples, num_features)
    
    # Generate anomaly labels
    n_anomalies = int(num_samples * anomaly_ratio)
    y = np.zeros(num_samples, dtype=int)
    y[n_normal:] = 1  # Last n_anomalies are anomalies
    
    # Make anomalies stand out by adding large values
    anomaly_indices = np.arange(n_normal, num_samples)
    X[anomaly_indices] += np.random.choice([-5, 5], size=(n_anomalies, num_features)) * np.random.rand(n_anomalies, num_features)
    
    return X, y


def zscore_anomaly_detection(X, threshold=2.5):
    """Detect anomalies using Z-score method."""
    # Calculate mean and std for each feature
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    # Calculate Z-scores
    zscores = np.abs((X - mean) / std)
    
    # Calculate anomaly scores (average Z-score across features)
    scores = np.mean(zscores, axis=1)
    
    # Predict anomalies (1 if score > threshold, 0 otherwise)
    predictions = (scores > threshold).astype(int)
    
    return scores, predictions


def iqr_anomaly_detection(X, multiplier=1.5):
    """Detect anomalies using IQR method."""
    # Calculate Q1, Q3, and IQR for each feature
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    
    # Avoid division by zero
    IQR = np.where(IQR == 0, 1, IQR)
    
    # Calculate anomaly scores based on IQR
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Calculate how far each point is from the bounds
    lower_dist = np.abs(X - lower_bound)
    upper_dist = np.abs(upper_bound - X)
    
    # Score is the minimum distance to bounds (normalized by IQR)
    scores = np.minimum(lower_dist, upper_dist) / IQR
    scores = np.mean(scores, axis=1)
    
    # Predict anomalies (1 if score > threshold, 0 otherwise)
    threshold = np.percentile(scores, 90)  # Top 10% are anomalies
    predictions = (scores > threshold).astype(int)
    
    return scores, predictions


def ensemble_anomaly_detection(zscore_scores, iqr_scores, zscore_weight=0.5, iqr_weight=0.5):
    """Combine Z-score and IQR scores using weighted ensemble."""
    # Normalize scores to [0, 1] range
    zscore_norm = (zscore_scores - np.min(zscore_scores)) / (np.max(zscore_scores) - np.min(zscore_scores) + 1e-10)
    iqr_norm = (iqr_scores - np.min(iqr_scores)) / (np.max(iqr_scores) - np.min(iqr_scores) + 1e-10)
    
    # Weighted combination
    ensemble_scores = zscore_weight * zscore_norm + iqr_weight * iqr_norm
    
    return ensemble_scores


def evaluate(model, X, y):
    """
    Evaluate the anomaly detection model.
    
    Returns:
        dict: Dictionary containing MSE, R2, and task-specific metrics
    """
    # Get predictions from both methods
    zscore_scores, zscore_preds = zscore_anomaly_detection(X)
    iqr_scores, iqr_preds = iqr_anomaly_detection(X)
    
    # Ensemble scores
    ensemble_scores = ensemble_anomaly_detection(zscore_scores, iqr_scores)
    
    # For ensemble predictions, use a threshold
    threshold = np.percentile(ensemble_scores, 90)
    ensemble_preds = (ensemble_scores > threshold).astype(int)
    
    # Calculate standard metrics using ensemble predictions
    mse = mean_squared_error(y, ensemble_scores)
    r2 = r2_score(y, ensemble_scores)
    
    # Calculate task-specific metrics for anomaly detection
    precision = precision_score(y, ensemble_preds, zero_division=0)
    recall = recall_score(y, ensemble_preds, zero_division=0)
    f1 = f1_score(y, ensemble_preds, zero_division=0)
    
    # Calculate AUC-ROC
    try:
        auc_roc = roc_auc_score(y, ensemble_scores)
    except ValueError:
        auc_roc = 0.5  # Default if only one class present
    
    # Calculate confusion matrix components
    tp = np.sum((ensemble_preds == 1) & (y == 1))
    tn = np.sum((ensemble_preds == 0) & (y == 0))
    fp = np.sum((ensemble_preds == 1) & (y == 0))
    fn = np.sum((ensemble_preds == 0) & (y == 1))
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'predicted_anomalies': int(ensemble_preds.sum()),
        'actual_anomalies': int(y.sum()),
        'zscore_mse': float(mean_squared_error(y, zscore_scores)),
        'iqr_mse': float(mean_squared_error(y, iqr_scores)),
    }


def visualize_results(y_val, zscore_scores, iqr_scores, ensemble_scores, save_dir='.'):
    """Generate visualization for anomaly detection results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Z-score results
    axes[0].scatter(range(len(y_val)), y_val, c='blue', label='Actual', alpha=0.7)
    axes[0].scatter(range(len(y_val)), zscore_scores, c='red', label='Z-score Scores', alpha=0.5)
    axes[0].set_title('Z-score Anomaly Detection')
    axes[0].set_xlabel('Sample Index')
    axes[0].legend()
    
    # IQR results
    axes[1].scatter(range(len(y_val)), y_val, c='blue', label='Actual', alpha=0.7)
    axes[1].scatter(range(len(y_val)), iqr_scores, c='green', label='IQR Scores', alpha=0.5)
    axes[1].set_title('IQR Anomaly Detection')
    axes[1].set_xlabel('Sample Index')
    axes[1].legend()
    
    # Ensemble results
    axes[2].scatter(range(len(y_val)), y_val, c='blue', label='Actual', alpha=0.7)
    axes[2].scatter(range(len(y_val)), ensemble_scores, c='purple', label='Ensemble Scores', alpha=0.5)
    axes[2].set_title('Ensemble Anomaly Detection')
    axes[2].set_xlabel('Sample Index')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/anomaly_detection_results.png', dpi=150)
    plt.close()


def main():  # noqa: C901
    """Main function to run the anomaly detection task."""
    print("=" * 60)
    print("Anomaly Detection using Z-score and IQR - Level 1")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Generating synthetic data...")
    X, y = generate_data(num_samples=500, num_features=3, anomaly_ratio=0.1)
    print(f"   Data shape: {X.shape}")
    print(f"   Anomaly ratio: {y.sum() / len(y):.2%} ({y.sum()} anomalies)")
    
    # 2. Split data (use 80% for training, 20% for validation)
    print("\n2. Splitting data...")
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # 3. Train model (fit anomaly detection parameters on training data)
    print("\n3. Training anomaly detection models...")
    zscore_scores_train, zscore_preds_train = zscore_anomaly_detection(X_train)
    iqr_scores_train, iqr_preds_train = iqr_anomaly_detection(X_train)
    print("   Z-score and IQR models trained on training data")
    
    # 4. Evaluate on training data
    print("\n4. Evaluating on training data...")
    train_metrics = evaluate(None, X_train, y_train)
    print(f"   Training MSE: {train_metrics['mse']:.6f}")
    print(f"   Training R2: {train_metrics['r2']:.6f}")
    print(f"   Training Precision: {train_metrics['precision']:.4f}")
    print(f"   Training Recall: {train_metrics['recall']:.4f}")
    print(f"   Training F1: {train_metrics['f1_score']:.4f}")
    print(f"   Training AUC-ROC: {train_metrics['auc_roc']:.4f}")
    
    # 5. Evaluate on validation data
    print("\n5. Evaluating on validation data...")
    val_metrics = evaluate(None, X_val, y_val)
    print(f"   Validation MSE: {val_metrics['mse']:.6f}")
    print(f"   Validation R2: {val_metrics['r2']:.6f}")
    print(f"   Validation Precision: {val_metrics['precision']:.4f}")
    print(f"   Validation Recall: {val_metrics['recall']:.4f}")
    print(f"   Validation F1: {val_metrics['f1_score']:.4f}")
    print(f"   Validation AUC-ROC: {val_metrics['auc_roc']:.4f}")
    
    # 6. Generate visualizations
    print("\n6. Generating visualizations...")
    zscore_scores_val, _ = zscore_anomaly_detection(X_val)
    iqr_scores_val, _ = iqr_anomaly_detection(X_val)
    ensemble_scores_val = ensemble_anomaly_detection(zscore_scores_val, iqr_scores_val)
    visualize_results(y_val, zscore_scores_val, iqr_scores_val, ensemble_scores_val, save_dir='.')
    print("   Saved: anomaly_detection_results.png")
    
    # 7. Quality checks
    print("\n7. Quality checks...")
    
    # Check R² score (should be positive, indicating model explains variance)
    assert val_metrics['r2'] > 0.0, f"R² should be positive: {val_metrics['r2']:.4f}"
    print(f"✓ R² score is positive: {val_metrics['r2']:.4f}")
    
    # Check MSE is reasonable (should be less than some threshold)
    assert val_metrics['mse'] < 1.0, f"MSE should be < 1.0: {val_metrics['mse']:.4f}"
    print(f"✓ MSE is reasonable: {val_metrics['mse']:.4f}")
    
    # Check precision > 0.8 (task requirement)
    assert val_metrics['precision'] > 0.8, f"Precision should be > 0.8: {val_metrics['precision']:.4f}"
    print(f"✓ Precision > 0.8: {val_metrics['precision']:.4f}")
    
    # Check F1 score is reasonable
    assert val_metrics['f1_score'] > 0.5, f"F1 should be > 0.5: {val_metrics['f1_score']:.4f}"
    print(f"✓ F1 score > 0.5: {val_metrics['f1_score']:.4f}")
    
    # Check AUC-ROC is reasonable (better than random)
    assert val_metrics['auc_roc'] > 0.6, f"AUC-ROC should be > 0.6: {val_metrics['auc_roc']:.4f}"
    print(f"✓ AUC-ROC > 0.6: {val_metrics['auc_roc']:.4f}")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
