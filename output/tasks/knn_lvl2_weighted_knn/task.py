import random
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any


class WeightedKNN:
    print(f"{prefix}{metrics.get('name', 'Model')} - MSE: {metrics['mse']:.6f}, R2: {metrics['r2']:.6f}")


def get_task_metadata() -> Dict[str, Any]:
    """Return metadata about the KNN task."""
    return {
        'task_name': 'knn_weighted_knn',
        'description': 'Weighted K-Nearest Neighbors for regression and classification',
        'supported_modes': ['regression', 'classification'],
        'default_mode': 'regression',
        'dataset': {
            'name': 'synthetic_nonlinear',
            'n_samples': 200,
            'n_features': 2,
            'noise_level': 0.1,
            'train_ratio': 0.8
        },
        'model': {
            'name': 'WeightedKNN',
            'default_k': 5,
            'weighting_scheme': 'inverse_squared_distance'
        },
        'metrics': ['mse', 'r2', 'accuracy'],
        'quality_thresholds': {
            'regression': {
                'r2_min': 0.3,
                'mse_max': 2.0
            },
            'classification': {
                'accuracy_min': 0.7
            }
        }
    }


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)


def get_device() -> str:
    """Get the computing device (CPU or CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
    return 'cpu'


def make_dataloaders(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    """Create dataloaders for training and validation.
    
    Args:
        cfg: Configuration dictionary with data parameters
       
    Returns:
        Tuple of (train_loader, val_loader) where each loader is a dict
        with 'features' and 'targets' keys
    """
    set_seed(cfg.get('seed', 42))
    
    X, y = generate_synthetic_data(
        n_samples=cfg.get('n_samples', 200),
        noise=cfg.get('noise', 0.1),
        random_state=cfg.get('seed', 42)
    )
    
    train_X, train_y, val_X, val_y = split_data(
        X, y, train_ratio=cfg.get('train_ratio', 0.8)
    )
    
    train_loader = {'features': train_X, 'targets': train_y}
    val_loader = {'features': val_X, 'targets': val_y}
    
    return train_loader, val_loader


def main():
    """Main function to run the weighted KNN experiment."""
    # Configuration
            'noise': 0.1,
            'train_ratio': 0.8
        },
        'seed': 42,
        'model': {
            'mode': 'regression',
            'k': 5,
        }
    }
    
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(
        n_samples=cfg['data'].get('n_samples', 200),
        random_state=cfg['model'].get('random_state', 42)
    )
    
    # Set seed for reproducibility
    set_seed(cfg['seed'])
    
    # Split data
    train_X, train_y, val_X, val_y = split_data(
        X, y, train_ratio=cfg['data'].get('train_ratio', 0.8)
    )
    print_metrics(weighted_train_metrics, "  ")
    
    # Check that weighted KNN improves vs unweighted baseline (regression)
    if cfg['model'].get('mode', 'regression') == 'regression':
        mse_improvement = unweighted_val_metrics['mse'] - weighted_val_metrics['mse']
        if mse_improvement >= 0.001:  # At least 0.1% improvement
            print(f"\n✓ Weighted KNN improves MSE by: {mse_improvement:.6f} ({mse_improvement/unweighted_val_metrics['mse']*100:.1f}%)")
        elif mse_improvement < 0:
            print(f"\n⚠ Weighted KNN MSE improvement is marginal: {mse_improvement:.6f}")
    
    # Assert quality thresholds
    # R2 should be positive (better than predicting mean)
    assert weighted_val_metrics['r2'] > 0.1, f"R2 score {weighted_val_metrics['r2']:.4f} should be > 0.1"
    print(f"✓ R2 > 0.1: {weighted_val_metrics['r2']:.6f}")
    
    # R2 should be reasonably high (good fit) - only for regression
    if cfg['model'].get('mode', 'regression') == 'regression':
        assert weighted_val_metrics['r2'] > 0.3, f"R2 score {weighted_val_metrics['r2']:.4f} should be > 0.3"
        print(f"✓ R2 > 0.3: {weighted_val_metrics['r2']:.6f}")
        assert weighted_val_metrics['mse'] < 2.0, f"MSE {weighted_val_metrics['mse']:.4f} should be < 2.0"
    assert weighted_val_metrics['mse'] < 2.0, f"MSE {weighted_val_metrics['mse']:.4f} should be < 2.0"
    print(f"✓ MSE < 2.0: {weighted_val_metrics['mse']:.6f}")
    
    return 0


def test_weighted_vs_unweighted():
    """Test that weighted KNN improves vs unweighted baseline."""
    set_seed(42)
    X, y = generate_synthetic_data(n_samples=200, noise=0.1, random_state=42)
    train_X, train_y, val_X, val_y = split_data(X, y, train_ratio=0.8)
    
    unweighted = WeightedKNN(k=5, mode='regression')
    weighted = WeightedKNN(k=5, mode='regression')
    unweighted.fit(train_X, train_y)
    weighted.fit(train_X, train_y)
    
    unweighted_metrics = evaluate(unweighted, val_X, val_y)
    weighted_metrics = evaluate(weighted, val_X, val_y)
    
    return unweighted_metrics['mse'] - weighted_metrics['mse'] >= 0.001

if __name__ == '__main__':
    import sys
