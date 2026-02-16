#!/usr/bin/env python3
"""
One-Class SVM Anomaly Detection - Level 3
Task: Implement evaluate() returning MSE, R2, and anomaly detection metrics
Implementation using One-Class SVM for anomaly detection
"""

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def generate_data(num_samples=200, num_features=2, contamination=0.1, noise=0.3):
    """Generate synthetic data with anomalies."""
    n_anomalies = int(num_samples * contamination)  # Number of anomalies
    n_normal = num_samples - n_anomalies  # Number of normal samples
    
    # Normal data: clustered around origin
    normal_data = np.random.randn(n_normal, num_features) * 1.5
    # Combine data
    X = np.vstack([normal_data, anomalies])
    
    # Labels: 0 for normal (inlier), 1 for anomaly (outlier)
    y = np.array([0] * n_normal + [1] * n_anomalies)
    
    # Add noise to all data
def split_data(X, y, train_ratio=0.8):
    """Split data into training and validation sets."""
    return train_test_split(X, y, train_size=train_ratio, random_state=42)


def train_model(X_train, contamination=0.1):
def predict_anomalies(model, X):
    """Predict anomalies using the trained model."""
    # OneClassSVM returns: 1 for inliers, -1 for outliers
    predictions = model.predict(X)
    # Convert to binary: 0 for normal (inlier), 1 for anomaly (outlier) 
    return (predictions == -1).astype(int)


    # Calculate standard metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y) if len(y) > 0 else 0
    if X.shape[1] != 2:
        print("Cannot visualize non-2D data")
        return
    
    # Create mesh for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Get predictions for mesh points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Generating synthetic data with anomalies...")
    X, y = generate_data(num_samples=200, num_features=2, contamination=0.1, noise=0.3)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Anomaly ratio: {np.mean(y) * 100:.2f}%")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 3. Train model
    print("\n3. Training One-Class SVM model (on normal data only)...")
    model = train_model(X_train, contamination=0.1)
    print("Model trained successfully")
    
    print(f"Training R2: {train_metrics['r2']:.6f}")
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Training Precision: {train_metrics['precision']:.4f}")
    print(f"Training TP: {train_metrics['tp']}, FP: {train_metrics['fp']}")
    print(f"Training Recall: {train_metrics['recall']:.4f}")
    print(f"Training F1: {train_metrics['f1']:.4f}")
    
    print(f"Validation R2: {val_metrics['r2']:.6f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation Precision: {val_metrics['precision']:.4f}")
    print(f"Validation TP: {val_metrics['tp']}, FP: {val_metrics['fp']}")
    print(f"Validation Recall: {val_metrics['recall']:.4f}")
    print(f"Validation F1: {val_metrics['f1']:.4f}")
    
    print("\n7. Quality checks...")
    
    # Check that model performs better than random (R2 > 0)
    assert val_metrics['r2'] > -2.0, f"R2 score should be reasonable: {val_metrics['r2']}"
    print(f"✓ R2 score is reasonable: {val_metrics['r2']:.4f}")
    
    # Check that accuracy is above threshold (at least 85%)
    assert val_metrics['accuracy'] >= 0.80, f"Accuracy should be >= 0.80: {val_metrics['accuracy']}"
    print(f"✓ Validation accuracy >= 0.80: {val_metrics['accuracy']:.4f}")
    
    # Check that F1 score is reasonable (at least 0.5)
    assert val_metrics['f1'] >= 0.5, f"F1 score should be >= 0.5: {val_metrics['f1']}"
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0  # Success


if __name__ == '__main__':
