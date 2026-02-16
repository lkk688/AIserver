"""Naive Bayes Production Inference Task - Single file implementation."""

import os
import sys
import json
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch


def generate_synthetic_data(n_samples=2000, n_features=15, noise=0.01):
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        n_redundant=0,
        n_classes=2,
        random_state=42,
        flip_y=noise,
        class_sep=2.5,
        n_clusters_per_class=1,
        hypercube=True,
        shift=0.5,
        scale=1.0
    )
    return X, y


def train_model(X_train, y_train):
    """Train Gaussian Naive Bayes model."""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    
    # Additional metrics for probability predictions
    mse_proba = mean_squared_error(y, y_proba)
    r2_proba = r2_score(y, y_proba)
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'accuracy': float(accuracy),
        'mse_proba': float(mse_proba),
        'r2_proba': float(r2_proba)
    }


def measure_throughput(model, X, n_iterations=10):
    """Measure inference throughput in samples/sec."""
    start_time = time.time()
    for _ in range(n_iterations):
        model.predict(X)
    end_time = time.time()
    
    total_samples = X.shape[0] * n_iterations
    throughput = total_samples / (end_time - start_time)
    return throughput


def save_model(model, filepath='nb_production_model.json'):
    """Save model parameters to JSON."""
    # Extract model parameters
    params = {
        'class_prior': model.class_prior_.tolist(),
        'theta': model.theta_.tolist(),
        'sigma': model.sigma_.tolist(),
        'classes': model.classes_.tolist()
    }
    
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)
    
    return filepath


def load_model(filepath='nb_production_model.json'):
    """Load model parameters from JSON and create model."""
    with open(filepath, 'r') as f:
        params = json.load(f)
    
    model = GaussianNB()
    model.class_prior_ = np.array(params['class_prior'])
    model.theta_ = np.array(params['theta'])
    model.sigma_ = np.array(params['sigma'])
    model.classes_ = np.array(params['classes'])
    model.fitted_ = True
    
    return model


def main():
    """Main function to run the complete task."""
    print("=" * 60)
    print("Naive Bayes Production Inference Task")
    print("=" * 60)
    
    try:
        # Generate data
        print("\n[1] Generating synthetic data...")
        X, y = generate_synthetic_data(n_samples=2000, n_features=15, noise=0.01)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"    Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")
        
        # Train model
        print("\n[2] Training Gaussian Naive Bayes model...")
        model = train_model(X_train, y_train)
        print("    Model trained successfully")
        
        # Evaluate on training set
        print("\n[3] Evaluating on training set...")
        train_metrics = evaluate(model, X_train, y_train)
        print(f"    train_mse: {train_metrics['mse']:.4f}")
        print(f"    train_r2: {train_metrics['r2']:.4f}")
        print(f"    train_accuracy: {train_metrics['accuracy']:.4f}")
        print(f"    train_mse_proba: {train_metrics['mse_proba']:.4f}")
        print(f"    train_r2_proba: {train_metrics['r2_proba']:.4f}")
        
        # Evaluate on validation set
        print("\n[4] Evaluating on validation set...")
        val_metrics = evaluate(model, X_val, y_val)
        print(f"    val_mse: {val_metrics['mse']:.4f}")
        print(f"    val_r2: {val_metrics['r2']:.4f}")
        print(f"    val_accuracy: {val_metrics['accuracy']:.4f}")
        print(f"    val_mse_proba: {val_metrics['mse_proba']:.4f}")
        print(f"    val_r2_proba: {val_metrics['r2_proba']:.4f}")
        
        # Measure throughput
        print("\n[5] Measuring inference throughput...")
        throughput = measure_throughput(model, X_val)
        print(f"    Throughput: {throughput:.2f} samples/sec")
        
        # Save model
        print("\n[6] Saving model to JSON...")
        model_path = save_model(model)
        print(f"    Model saved to {model_path}")
        
        # Load and verify
        print("\n[7] Loading model from JSON and verifying...")
        loaded_model = load_model(model_path)
        val_pred_original = model.predict(X_val)
        val_pred_loaded = loaded_model.predict(X_val)
        predictions_match = np.allclose(val_pred_original, val_pred_loaded)
        print(f"    Predictions match after load: {predictions_match}")
        
        # Check quality thresholds
        print("\n[8] Checking quality thresholds...")
        thresholds = {
            'val_accuracy': (val_metrics['accuracy'], 0.85),
            'val_r2': (val_metrics['r2'], 0.7),
            'train_accuracy': (train_metrics['accuracy'], 0.8),
            'throughput': (throughput, 1000.0)
        }
        
        all_passed = True
        for metric_name, (value, threshold) in thresholds.items():
            passed = value >= threshold
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"    {metric_name}: {value:.4f} >= {threshold} [{status}]")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n" + "=" * 60)
            print("SUCCESS: All quality thresholds met!")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("FAILURE: Some quality thresholds not met!")
            print("=" * 60)
            return 1
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
