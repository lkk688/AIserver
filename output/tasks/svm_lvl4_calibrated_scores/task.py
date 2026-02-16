#!/usr/bin/env python3
"""
SVM with Calibrated Scores - Level 4
Task: Train SVM with probability calibration and evaluate using ROC/PR curves
Task: Save visualizations and metrics
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, mean_squared_error, r2_score
)
from sklearn.datasets import make_classification

np.random.seed(42)


def generate_data(num_samples=1000, num_features=10, noise=0.1):
    """
    Generate synthetic binary classification data.
    
    Args:
        num_samples: Number of samples to generate
        num_features: Number of features
        noise: Noise level
    
    Returns:
        tuple: (X, y)
    """
    # Generate features
    X = np.random.randn(num_samples, num_features)
    
    # Create meaningful relationship with some features
    y = (0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.4 * X[:, 2] + 
         0.2 * X[:, 3] * X[:, 4] + np.random.randn(num_samples) * noise > 0).astype(int)
    
    return X, y


def make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create simple data loaders (using numpy arrays as PyTorch is not required).
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size
    
    Returns:
        tuple: (train_loader, val_loader) - just returns the data for this task
    """
    # For this task, we'll just return the data directly
    # In a real PyTorch implementation, we'd create DataLoader objects
    return (X_train, y_train), (X_val, y_val)


def predict(model, X):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        X: Features
    
    Returns:
        tuple: (predictions, probabilities)
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba


def train_model(X_train, y_train):
    """Train SVM with probability calibration."""
    # Train a basic SVM with linear kernel
    base_svm = SVC(kernel='rbf', probability=False, random_state=42)
    
    # Calibrate the SVM using isotonic regression
    calibrated_svm = CalibratedClassifierCV(base_svm, method='isotonic', cv=3)
    calibrated_svm.fit(X_train, y_train)
    
    return calibrated_svm


def evaluate(model, X, y, name="Validation"):
    """
    Evaluate the model and return metrics.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        name: Dataset name for logging (default: "Validation")
    
    Returns:
        dict: Metrics including MSE, R2, and task-specific metrics
    """
    # Get predictions
    y_pred, y_proba = predict(model, X)
    
    # Calculate standard metrics
    mse = mean_squared_error(y, y_proba)
    r2 = r2_score(y, y_proba)
    
    # Calculate task-specific metrics
    f1 = f1_score(y, y_pred, zero_division=0)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    ap = average_precision_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'ap': ap,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    print(f"\n{name} Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R2 Score: {r2:.6f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    return metrics


def save_artifacts(model, metrics, save_dir='.'):
    """
    Save model artifacts and metrics.
    
    Args:
        model: Trained model
        metrics: Dictionary of metrics
        save_dir: Directory to save artifacts
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model to: {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(save_dir, 'metrics.json')
    # Convert numpy types to Python types for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.int64, np.int32)):
            serializable_metrics[key] = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            serializable_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        else:
            serializable_metrics[key] = value
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")
    
    return model_path, metrics_path


def visualize_results(y_val, y_proba_val, y_train, y_proba_train, save_dir='.'):
    """Generate and save ROC and PR curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr_val, tpr_val, _ = roc_curve(y_val, y_proba_val)
    auc_val = roc_auc_score(y_val, y_proba_val)
    ax1.plot(fpr_val, tpr_val, color='blue', lw=2, label=f'Validation (AUC = {auc_val:.4f})', alpha=0.8)
    
    fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train)
    auc_train = roc_auc_score(y_train, y_proba_train)
    ax1.plot(fpr_train, tpr_train, color='red', lw=2, label=f'Train (AUC = {auc_train:.4f})', alpha=0.8, linestyle='--')
    ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_proba_val)
    ap_val = average_precision_score(y_val, y_proba_val)
    ax2.plot(recall_val, precision_val, color='blue', lw=2, label=f'Validation (AP = {ap_val:.4f})', alpha=0.8)
    
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_proba_train)
    ap_train = average_precision_score(y_train, y_proba_train)
    ax2.plot(recall_train, precision_train, color='red', lw=2, label=f'Train (AP = {ap_train:.4f})', alpha=0.8, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved ROC and PR curves to: {os.path.join(save_dir, 'results.png')}")
    return os.path.join(save_dir, 'results.png')


def main():  # noqa: C901
    print("=" * 60)
    print("SVM with Calibrated Scores - Level 4")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Generating synthetic data...")
    X, y = generate_data(num_samples=1000, num_features=10, noise=0.1)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # 2. Split data
    print("\n2. Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 3. Create data loaders
    print("\n3. Creating data loaders...")
    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val)
    print(f"Train loader: {len(train_loader[0])} samples, Val loader: {len(val_loader[0])} samples")
    
    # 4. Train model
    print("\n4. Training SVM with probability calibration...")
    model = train_model(X_train, y_train)
    print("Model trained successfully!")
    
    # 5. Evaluate on training data
    print("\n5. Evaluating on training data...")
    train_metrics = evaluate(model, X_train, y_train, name="Training")
    
    # 6. Evaluate on validation data
    print("\n6. Evaluating on validation data...")
    val_metrics = evaluate(model, X_val, y_val, name="Validation")
    
    # 6.5. Save artifacts
    print("\n6.5. Saving artifacts...")
    model_path, metrics_path = save_artifacts(model, val_metrics, save_dir='.')
    
    # 7. Generate visualizations
    print("\n7. Generating visualizations...")
    viz_path = visualize_results(y_val, val_metrics['y_proba'], y_train, train_metrics['y_proba'], save_dir='.')
    
    # 8. Quality checks
    print("\n8. Quality checks...")
    
    # Check AUC score (should be good for this synthetic data)
    assert val_metrics['auc'] > 0.85, f"AUC should be > 0.85, got {val_metrics['auc']:.4f}"
    print(f"✓ AUC score: {val_metrics['auc']:.4f} > 0.85")
    
    # Check accuracy
    assert val_metrics['accuracy'] > 0.75, f"Accuracy should be > 0.75, got {val_metrics['accuracy']:.4f}"
    print(f"✓ Accuracy: {val_metrics['accuracy']:.4f} > 0.75")
    
    # Check that validation metrics are reasonable
    assert val_metrics['mse'] < 0.30, f"MSE should be < 0.30, got {val_metrics['mse']:.6f}"
    print(f"✓ MSE: {val_metrics['mse']:.6f} < 0.30")
    
    # Check R2 score (should be positive and reasonably high)
    assert val_metrics['r2'] > 0.4, f"R2 should be > 0.4, got {val_metrics['r2']:.6f}"
    print(f"✓ R2 Score: {val_metrics['r2']:.6f} > 0.4")
    
    # Check F1 score
    assert val_metrics['f1'] > 0.70, f"F1 should be > 0.70, got {val_metrics['f1']:.4f}"
    print(f"✓ F1 Score: {val_metrics['f1']:.4f} > 0.70")
    
    # Check that train and validation metrics are similar (no major overfitting)
    auc_diff = abs(train_metrics['auc'] - val_metrics['auc'])
    assert auc_diff < 0.15, f"Train-validation AUC difference too large: {auc_diff:.4f}"
    print(f"✓ Train-validation AUC difference: {auc_diff:.4f} < 0.15 (no major overfitting)")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0  # Success


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
