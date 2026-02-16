#!/usr/bin/env python3
"""
Decision Tree Feature Importance Analysis - Level 4
Validates that top-k important features overlap with sklearn permutation importance.
Task functions: get_task_metadata, set_seed, get_device, make_dataloaders, evaluate.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import torch
import warnings
warnings.filterwarnings('ignore')

def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        'task_name': 'dtree_lvl4_feature_importance',
        'task_type': 'regression',
        'description': 'Decision Tree Feature Importance Analysis',
        'validation_method': 'permutation_importance_overlap',
        'target_metric': 'validation_R2'
    }


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Get the computing device (CPU/GPU)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess the California housing dataset."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    return X, y, feature_names


def make_dataloaders(X, y, batch_size: int = 32, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Create data splits for training (returns numpy arrays for tree models)."""
    set_seed(seed)
    
    # Use train_test_split for clean splits - returns numpy arrays directly
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=1 - train_ratio, random_state=seed
    )
    val_size = val_ratio / (1 - train_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=seed
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_model(X_train, y_train, **kwargs) -> DecisionTreeRegressor:
    """Train a decision tree regressor."""
    model = DecisionTreeRegressor(
        max_depth=kwargs.get('max_depth', 10),
        min_samples_split=kwargs.get('min_samples_split', 2),
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def get_feature_importance(model, feature_names):
    """Get feature importance from the model."""
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    return importance_dict


def get_top_k_features(importance_dict, k=5):
    """Get top-k features by importance."""
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return [feat for feat, _ in sorted_features[:k]]


def predict(model, X):
    """Make predictions using the trained model."""
    return model.predict(X)


def save_artifacts(model, feature_names, metrics, output_dir: str = "output/tasks/dtree_lvl4_feature_importance"):
    """Save model artifacts and evaluation results."""
    import os
    import json
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names
    features_path = os.path.join(output_dir, "feature_names.json")
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save top-k features
    tree_importance = get_feature_importance(model, feature_names)
    top_k = get_top_k_features(tree_importance, k=5)
    top_k_path = os.path.join(output_dir, "top_k_features.json")
    with open(top_k_path, 'w') as f:
        json.dump({
            "top_k_features": top_k,
            "importance_values": {feat: tree_importance[feat] for feat in top_k}
        }, f, indent=2)
    
    return {
        "model_path": model_path,
        "features_path": features_path,
        "metrics_path": metrics_path,
        "top_k_path": top_k_path
    }


def evaluate(model, X, y, feature_names, dataset_name: str = "dataset") -> Dict[str, Any]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Task-specific metrics - permutation importance for validation
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    perm_sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    perm_top_k = [feature_names[i] for i in perm_sorted_idx[:5]]
    
    # Get tree-based feature importance
    tree_importance_dict = get_feature_importance(model, feature_names)
    tree_top_k = get_top_k_features(tree_importance_dict, k=5)
    
    # Calculate overlap between tree-based and permutation importance
    overlap = len(set(tree_top_k) & set(perm_top_k))
    overlap_ratio = overlap / 5.0  # 5 is k
    
    task_metrics = {
        'feature_importance_ranking': list(range(len(model.feature_importances_))),
        'permutation_top_k_features': perm_top_k,
        'tree_top_k_features': tree_top_k,
        'feature_importance_overlap': overlap_ratio
    }
    
    metrics = {
        f'{dataset_name}_MSE': mse,
        f'{dataset_name}_R2': r2,
        f'{dataset_name}_RMSE': np.sqrt(mse),
        **task_metrics
    }
    return metrics


def main():
    """Main function to run the decision tree feature importance task."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Load California housing dataset
    X, y, feature_names = load_data()
    
    # Split data: 70% train, 15% validation, 15% test using make_dataloaders
    train_data, val_data, test_data = make_dataloaders(X, y, train_ratio=0.7, val_ratio=0.15, seed=42)
    X_train, y_train = train_data
    X_val, y_val = val_data  
    X_test, y_test = test_data
    
    # Train model (decision trees work directly with numpy arrays)
    model = train_model(X_train, y_train)
    
    # Evaluate on train, validation, and test sets
    train_metrics = evaluate(model, X_train, y_train, feature_names, dataset_name="train")
    val_metrics = evaluate(model, X_val, y_val, feature_names, dataset_name="validation")
    test_metrics = evaluate(model, X_test, y_test, feature_names, dataset_name="test")
    
    # Print metrics
    print("\n=== Training Metrics ===")
    print(f"Train MSE: {train_metrics['train_MSE']:.6f}")
    print(f"Train R2: {train_metrics['train_R2']:.6f}")
    print(f"Train RMSE: {train_metrics['train_RMSE']:.6f}")
    print(f"Train Feature Importance Overlap: {train_metrics['feature_importance_overlap']:.4f}")
    
    print("\n=== Test Metrics ===")
    print(f"Test MSE: {test_metrics['test_MSE']:.6f}")
    print(f"Test R2: {test_metrics['test_R2']:.6f}")
    print(f"Test RMSE: {test_metrics['test_RMSE']:.6f}")
    print(f"Test Feature Importance Overlap: {test_metrics['feature_importance_overlap']:.4f}")
    
    print("\n=== Validation Metrics ===")
    print(f"Validation MSE: {val_metrics['validation_MSE']:.6f}")
    print(f"Validation R2: {val_metrics['validation_R2']:.6f}")
    print(f"Validation RMSE: {val_metrics['validation_RMSE']:.6f}")
    print(f"Validation Feature Importance Overlap: {val_metrics['feature_importance_overlap']:.4f}")
    
    # Use validation metrics for quality assertions
    metrics = val_metrics
    
    # Save artifacts
    artifacts = save_artifacts(model, feature_names, {
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics
    })
    print(f"\nArtifacts saved to: {artifacts}")
    
    # Print top-k features
    print("\n=== Top 5 Features (Tree-based) ===")
    tree_importance = get_feature_importance(model, feature_names)
    top_k_tree = get_top_k_features(tree_importance, k=5)
    for i, feat in enumerate(top_k_tree, 1):
        print(f"{i}. {feat}: {tree_importance[feat]:.6f}")
    
    print("\n=== Top 5 Features (Permutation) ===")
    perm_importance = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)
    perm_sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    for i, idx in enumerate(perm_sorted_idx[:5], 1):
        print(f"{i}. {feature_names[idx]}: {perm_importance.importances_mean[idx]:.6f}")
    
    # Assert quality thresholds
    assert metrics['validation_R2'] > 0.5, f"Validation R2 {metrics['validation_R2']:.4f} < 0.5"
    assert metrics['validation_MSE'] < 1.0, f"Validation MSE {metrics['validation_MSE']:.4f} >= 1.0"
    assert metrics['feature_importance_overlap'] >= 0.6, f"Feature importance overlap {metrics['feature_importance_overlap']:.4f} < 0.6"
    
    print("\n=== All quality thresholds passed! ===")
    return 0


if __name__ == '__main__':
    exit(main())
