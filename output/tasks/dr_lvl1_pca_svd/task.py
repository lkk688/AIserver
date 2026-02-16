#!/usr/bin/env python3
"""
PCA using SVD - Level 1
Task: Implement evaluate() returning MSE, R2, and metrics
Implementation using sklearn PCA with SVD decomposition
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Set random seeds for reproducibility
np.random.seed(42)


def generate_data(num_samples=200, num_features=10, noise=0.1):
    """Generate synthetic high-dimensional data with structure."""
    # Create data with underlying low-dimensional structure
    n_components = 3
    # Generate latent factors
    Z = np.random.randn(num_samples, n_components)
    # Generate mixing matrix
    W = np.random.randn(n_components, num_features)
    # Generate data
    X = Z @ W + noise * np.random.randn(num_samples, num_features)
    return X


def split_data(X, train_ratio=0.8):
    """Split data into train and validation sets."""
    return train_test_split(X, train_size=train_ratio, random_state=42)


def build_model(n_components):
    """Build PCA model with specified number of components."""
    return PCA(n_components=n_components, svd_solver='full')


def train(model, X_train):
    """Train PCA model."""
    model.fit(X_train)
    return model


def evaluate(model, X):
    """
    Evaluate PCA model and return metrics.
    
    Returns:
        dict with reconstruction_mse, r2_score, explained_variance_ratio, cumulative_variance_ratio
    """
    # Transform and reconstruct
    X_transformed = model.transform(X)
    X_reconstructed = model.inverse_transform(X_transformed)
    
    # Calculate reconstruction error
    mse = mean_squared_error(X, X_reconstructed)
    r2 = r2_score(X.flatten(), X_reconstructed.flatten())
    
    return {
        'reconstruction_mse': float(mse),
        'r2_score': float(r2),
        'explained_variance_ratio': model.explained_variance_ratio_.tolist(),
        'cumulative_variance_ratio': np.cumsum(model.explained_variance_ratio_).tolist()
    }


def save_artifacts(model, metrics_dict, save_dir='.'):
    """Save model and metrics to files."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'pca_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics_path = os.path.join(save_dir, 'metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    return {'model_path': model_path, 'metrics_path': metrics_path}


def main():
    """Main function to run the PCA task."""
    print("=" * 60)
    print("PCA using SVD - Level 1")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Generating synthetic high-dimensional data with structure...")
    X = generate_data(num_samples=200, num_features=10, noise=0.1)
    print(f"Data shape: {X.shape}")
    
    X_train, X_val = split_data(X, train_ratio=0.8)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 2. Test different numbers of components (for validation)
    print("\n2. Testing different numbers of components...")
    train_metrics_list = []
    val_metrics_list = []
    train_errors = []
    val_errors = []
    
    max_components = min(X.shape[1], len(X_train))
    for n_comp in range(1, max_components + 1):
        model = build_model(n_components=n_comp)
        model = train(model, X_train)
        
        train_metrics = evaluate(model, X_train)
        val_metrics = evaluate(model, X_val)
        
        train_metrics_list.append(train_metrics)
        val_metrics_list.append(val_metrics)
        train_errors.append(train_metrics['reconstruction_mse'])
        val_errors.append(val_metrics['reconstruction_mse'])
        
        if n_comp % 2 == 0 or n_comp == 1:
            print(f"  n_comp={n_comp:2d}: Train MSE={train_metrics['reconstruction_mse']:.4f}, "
                  f"Val MSE={val_metrics['reconstruction_mse']:.4f}, "
                  f"R2={train_metrics['r2_score']:.4f}")
    
    # 3. Select optimal number of components (95% variance threshold)
    print("\n3. Selecting optimal number of components based on 95% variance threshold...")
    cumulative_ratios = [m['cumulative_variance_ratio'][-1] for m in train_metrics_list]
    
    # Find minimum components for 95% variance
    n_components = 1
    for i, ratio in enumerate(cumulative_ratios):
        if ratio >= 0.95:
            n_components = i + 1
            break
    
    # Ensure at least 3 components for better reconstruction
    n_components = max(n_components, 3)
    n_components = min(n_components, max_components)
    
    print(f"Selected {n_components} components (captures {cumulative_ratios[n_components-1]:.2%} variance)")
    
    # 4. Train final model
    print(f"\n4. Training final model with {n_components} components...")
    final_model = build_model(n_components=n_components)
    final_model = train(final_model, X_train)
    
    # 5. Evaluate on both splits
    train_metrics = evaluate(final_model, X_train)
    val_metrics = evaluate(final_model, X_val)
    
    print(f"\n5. Final Model Results:")
    print(f"   Train - MSE: {train_metrics['reconstruction_mse']:.6f}, R2: {train_metrics['r2_score']:.6f}")
    print(f"   Val   - MSE: {val_metrics['reconstruction_mse']:.6f}, R2: {val_metrics['r2_score']:.6f}")
    print(f"   Explained variance ratio: {train_metrics['explained_variance_ratio']}")
    print(f"   Cumulative variance: {train_metrics['cumulative_variance_ratio'][-1]:.4f}")
    
    # 6. Quality checks (validation: reconstruction error decreases with more components)
    print("\n6. Quality checks...")
    
    # Check reconstruction error decreases with more components
    assert train_errors[0] > train_errors[-1], \
        f"Training reconstruction error should decrease: {train_errors[0]:.4f} -> {train_errors[-1]:.4f}"
    print(f"✓ Training reconstruction error decreases: {train_errors[0]:.4f} -> {train_errors[-1]:.4f}")
    
    # Check validation reconstruction error also decreases (generally true for PCA)
    assert val_errors[0] > val_errors[-1], \
        f"Validation reconstruction error should decrease: {val_errors[0]:.4f} -> {val_errors[-1]:.4f}"
    print(f"✓ Validation reconstruction error decreases: {val_errors[0]:.4f} -> {val_errors[-1]:.4f}")
    
    # Check R2 score is good (require > 0.9 for validation)
    assert val_metrics['r2_score'] > 0.9, \
        f"R2 score should be > 0.9, got {val_metrics['r2_score']:.4f}"
    print(f"✓ R2 score is good: {val_metrics['r2_score']:.4f}")
    
    # Check explained variance is high (require > 0.95)
    assert train_metrics['cumulative_variance_ratio'][-1] > 0.95, \
        f"Should capture > 95% variance, got {train_metrics['cumulative_variance_ratio'][-1]:.4f}"
    print(f"✓ Explained variance is high: {train_metrics['cumulative_variance_ratio'][-1]:.4f}")
    
    # Check validation MSE is reasonable
    assert val_metrics['reconstruction_mse'] < 0.5, \
        f"Validation MSE should be < 0.5, got {val_metrics['reconstruction_mse']:.4f}"
    print(f"✓ Validation MSE is reasonable: {val_metrics['reconstruction_mse']:.4f}")
    
    # 7. Save artifacts
    print("\n7. Saving artifacts...")
    artifacts = save_artifacts(final_model, {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'n_components': n_components,
        'train_errors': train_errors,
        'val_errors': val_errors
    })
    print(f"Saved model to: {artifacts['model_path']}")
    print(f"Saved metrics to: {artifacts['metrics_path']}")
    
    print("=" * 60)
    print("All quality checks passed!")
    print("=" * 60)
    
    return 0  # Success


if __name__ == '__main__':
    exit(main())
