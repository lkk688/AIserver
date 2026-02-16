#!/usr/bin/env python3
"""
Anomaly Detection with Isolation Forest - Level 2
Task: Implement save_artifacts
Implementation using Isolation Forest on synthetic data
"""

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from pathlib import Path
import json
import numpy as np


def generate_data(num_samples=1000, num_features=5, contamination=0.1, random_state=42):
    """Generate synthetic data with anomalies."""
    np.random.seed(random_state)
    
    # Generate normal data
    n_normal = int(num_samples * (1 - contamination))
    normal_data = np.random.randn(n_normal, num_features)
    
    # Generate anomalies (outliers)
    n_anomaly = num_samples - n_normal
    anomaly_data = np.random.randn(n_anomaly, num_features) * 3 + 5  # Far from normal data
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    y = np.array([0] * n_normal + [1] * n_anomaly)  # 0 = normal, 1 = anomaly
    
    return X, y


def train_model(X_train, contamination=0.1, random_state=42):
    """Train Isolation Forest model."""
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train)
    return model


def evaluate(model, X, y):
    """
    Evaluate the model and return metrics.
    
    Returns:
        dict: Contains MSE, R2 score, and anomaly detection specific metrics
    """
    # Get predictions (-1 for anomalies, 1 for normal)
    predictions = model.predict(X)
    
    # Convert to binary (1 for anomaly, 0 for normal) for metric calculations
    pred_binary = (predictions == -1).astype(int)
    
    # Calculate standard metrics
    mse = mean_squared_error(y, pred_binary)
    r2 = r2_score(y, pred_binary)
    
    # Calculate anomaly detection specific metrics
    precision = precision_score(y, pred_binary, zero_division=0)
    recall = recall_score(y, pred_binary, zero_division=0)
    f1 = f1_score(y, pred_binary, zero_division=0)
    
    # Calculate AUC if we have probability scores
    try:
        decision_scores = model.decision_function(X)
        auc = roc_auc_score(y, -decision_scores)  # Negate because lower scores = anomalies
    except Exception:
        auc = 0.0
    
    return {
        'mse': mse,
        'r2': r2,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predicted_anomalies': int(pred_binary.sum()),
        'true_anomalies': int(y.sum())
    }


def save_artifacts(model, output_path):
    """Save model artifacts including the trained model and metadata."""
    # Save the model
    model_file = output_path / 'model.pkl'
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")
    
    # Save model metadata
    metadata = {
        'model_type': 'IsolationForest',
        'n_estimators': model.n_estimators,
        'contamination': model.contamination,
        'random_state': model.random_state
    }
    metadata_file = output_path / 'model_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to {metadata_file}")


def main():
    """Main function to run the anomaly detection task."""
    print("=" * 60)
    print("Anomaly Detection with Isolation Forest - Level 2")
    print("=" * 60)
    
    # Generate data
    print("\n1. Generating synthetic data...")
    X, y = generate_data(num_samples=1000, num_features=5, contamination=0.1, random_state=42)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"True anomalies: {y.sum()} ({y.mean()*100:.1f}%)")
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Train model
    print("\n3. Training Isolation Forest model...")
    model = train_model(X_train, contamination=0.1, random_state=42)
    print(f"Model trained with {model.n_estimators} estimators")
    
    # Evaluate on training data
    print("\n4. Evaluating on training data...")
    train_metrics = evaluate(model, X_train, y_train)
    print(f"Training MSE: {train_metrics['mse']:.6f}")
    print(f"Training R2: {train_metrics['r2']:.6f}")
    print(f"Training Precision: {train_metrics['precision']:.4f}")
    print(f"Training Recall: {train_metrics['recall']:.4f}")
    print(f"Training F1: {train_metrics['f1']:.4f}")
    print(f"Training AUC: {train_metrics['auc']:.4f}")
    
    # Evaluate on validation data
    print("\n5. Evaluating on validation data...")
    val_metrics = evaluate(model, X_val, y_val)
    print(f"Validation MSE: {val_metrics['mse']:.6f}")
    print(f"Validation R2: {val_metrics['r2']:.6f}")
    print(f"Validation Precision: {val_metrics['precision']:.4f}")
    print(f"Validation Recall: {val_metrics['recall']:.4f}")
    print(f"Validation F1: {val_metrics['f1']:.4f}")
    print(f"Validation AUC: {val_metrics['auc']:.4f}")
    
    # Save artifacts
    print("\n6. Saving artifacts...")
    output_path = Path('output/tasks/anom_lvl2_isolation_forest_like')
    output_path.mkdir(parents=True, exist_ok=True)
    save_artifacts(model, output_path)
    
    # Quality checks
    print("\n7. Quality checks...")
    
    # Check that model detects more anomalies than random (AUC > 0.5)
    assert val_metrics['auc'] > 0.5, f"AUC should be > 0.5, got {val_metrics['auc']:.4f}"
    print(f"✓ AUC > 0.5: {val_metrics['auc']:.4f}")
    
    # Check that F1 score is reasonable (at least 0.3 for anomaly detection)
    assert val_metrics['f1'] > 0.3, f"F1 score should be > 0.3, got {val_metrics['f1']:.4f}"
    print(f"✓ F1 score > 0.3: {val_metrics['f1']:.4f}")
    
    # Check that recall is reasonable (at least 0.2)
    assert val_metrics['recall'] > 0.2, f"Recall should be > 0.2, got {val_metrics['recall']:.4f}"
    print(f"✓ Recall > 0.2: {val_metrics['recall']:.4f}")
    
    print("All quality checks passed!")
    print("=" * 60)
    
    return 0  # Success


if __name__ == '__main__':
    result = main()
    exit(result)
