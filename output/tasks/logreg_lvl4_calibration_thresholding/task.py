"""Logistic Regression Calibration and Thresholding Task."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    f1_score,
    mean_squared_error,
    r2_score,
    accuracy_score
)
import numpy as np
import os

# Ensure output directory exists for plots
os.makedirs("output/tasks/logreg_lvl4_calibration_thresholding", exist_ok=True)

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error manually."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_confidence_i = np.mean(y_prob[in_bin])
            avg_accuracy_i = np.mean(y_true[in_bin])
            ece += np.abs(avg_accuracy_i - avg_confidence_i) * prop_in_bin
    
    return ece


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray = None) -> tuple:
    """Find optimal threshold for F1 score."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, title: str, filename: str, n_bins: int = 10) -> None:
    """Plot reliability diagram."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    accuracies = []
    confidences = []
    counts = []
    
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            accuracies.append(np.mean(y_true[in_bin]))
            confidences.append(np.mean(y_prob[in_bin]))
            counts.append(np.sum(in_bin))
        else:
            accuracies.append(0)
            confidences.append(bin_centers[i])
            counts.append(0)
    
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, accuracies, width=0.08, alpha=0.7, label='Accuracy', edgecolor='black')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, filename: str) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def evaluate(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, model, threshold: float = 0.5) -> tuple:
    """Evaluate model and return metrics."""
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_val = model.predict_proba(X_val)[:, 1]
    
    # Compute predictions with threshold
    y_pred_train = (y_prob_train >= threshold).astype(int)
    y_pred_val = (y_prob_val >= threshold).astype(int)
    
    # Compute metrics for training set
    train_mse = mean_squared_error(y_train, y_prob_train)
    train_r2 = r2_score(y_train, y_prob_train)
    train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
    train_ece = compute_ece(y_train, y_prob_train)
    
    # Compute metrics for validation set
    val_mse = mean_squared_error(y_val, y_prob_val)
    val_r2 = r2_score(y_val, y_prob_val)
    val_f1 = f1_score(y_val, y_pred_val, zero_division=0)
    val_ece = compute_ece(y_val, y_prob_val)
    
    train_metrics = {
        'MSE': train_mse,
        'R2': train_r2,
        'F1': train_f1,
        'ECE': train_ece
    }
    
    val_metrics = {
        'MSE': val_mse,
        'R2': val_r2,
        'F1': val_f1,
        'ECE': val_ece
    }
    
    return train_metrics, val_metrics


def main() -> int:
    print("=" * 60)
    print("Logistic Regression Calibration and Thresholding Task")
    print("=" * 60)
    
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split data: 60% train, 20% val, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Split train_val into train and validation (80% train, 20% val of original)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training samples
