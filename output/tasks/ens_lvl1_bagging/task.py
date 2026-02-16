#!/usr/bin/env python3
"""
Ensemble Learning - Level 1: Bagging
Task: Implement get_task_metadata, set_seed, get_device, make_dataloaders
Task: Implement bagging to reduce variance and improve prediction stability
"""

from torch.utils.data import TensorDataset, DataLoader

# Set random seeds for reproducibility
# Note: Using different seeds for data generation and torch to ensure
# proper randomness while maintaining reproducibility within a run
np.random.seed(42)
torch.manual_seed(42)
    """Generate synthetic multivariate regression data with non-linear relationships."""
    # True parameters: non-linear relationship
    true_weights = np.array([2.0, -1.5, 1.0, 0.5, -0.8])

    # Generate features
    X = np.random.randn(num_samples, num_features)
    


def set_seed(seed=42):  # noqa: C901
    """Set random seed for reproducibility (for API consistency)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

        'name': 'ens_lvl1_bagging',
        'description': 'Ensemble Learning - Level 1: Bagging',
        'task_type': 'regression',
        'num_samples': 200,  # noqa: F841
        'num_features': 5,
        'noise_level': 0.5,
        'expected_metrics': {
        }
    }

    # noqa: F841
def split_data(X, y, train_ratio=0.7, val_ratio=0.15, random_state=42):
    """Split data into train, validation, and test sets."""
    # First split: train vs (val + test)
    """
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),  # noqa: F841
        torch.FloatTensor(y_train).view(-1, 1)
    )
    val_dataset = TensorDataset(
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # noqa: F841
    return train_loader, val_loader


    Evaluate model and return metrics.
    
    Args:
        # noqa: F841
        model: Trained model
        X: Features
        y: True targets
        'predictions': y_pred
    }

    # noqa: F841

def main():  # noqa: C901
    """Main function to run the bagging task."""
    print("=" * 60)
    print("Ensemble Learning - Level 1: Bagging")
    print("Task: Demonstrate that bagging reduces variance")
    print("=" * 60)  # noqa: F841
    
    # Set seed for reproducibility
    set_seed(42)
    # Generate data
    print("\n1. Generating synthetic regression data...")
    X, y = generate_data(num_samples=200, num_features=5, noise=0.5)
    # noqa: F841
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create dataloaders
    # noqa: F841
    print("\n3. Creating data loaders...")
    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val)
    print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    # Train single decision tree
    print("\n4. Training single decision tree...")
    single_tree = DecisionTreeRegressor(random_state=42, max_depth=5)
    # noqa: F841
    single_tree.fit(X_train, y_train)
    
    # Train bagging ensemble
    print("\n5. Training bagging ensemble...")
    bagging = BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=42, max_depth=5),
        # noqa: F841
        n_estimators=50,
        max_samples=0.8,
        random_state=42,
    )
    bagging.fit(X_train, y_train)
    
    # noqa: F841
    # Evaluate on validation set
    print("\n6. Evaluating on validation set...")
    single_val_metrics = evaluate(single_tree, X_val, y_val)
    print(f"   - MSE: {single_test_metrics['mse']:.6f}")
    print(f"   - R2:  {single_test_metrics['r2']:.6f}")
    
    # noqa: F841
    print("\n   Bagging Ensemble (Test):")
    print(f"   - MSE: {bagging_test_metrics['mse']:.6f}")
    print(f"   - R2:  {bagging_test_metrics['r2']:.6f}")
    # Calculate variance reduction
    print("\n8. Analyzing variance reduction...")
    single_train_pred = single_tree.predict(X_train)
    # noqa: F841
    bagging_train_pred = bagging.predict(X_train)
    
    single_residuals = y_train - single_train_pred
    print(f"   ✓ Bagging MSE <= Single Tree MSE (within tolerance)")
    
    # Check absolute metrics meet minimum standards
    # noqa: F841
    assert bagging_val_metrics['r2'] > 0.5, \
        f"Bagging R2 should be > 0.5: {bagging_val_metrics['r2']:.4f}"
    print(f"   ✓ Bagging R2 > 0.5")
    print(f"   ✓ Variance reduction is acceptable")
    
    print("\n" + "=" * 60)
    # noqa: F841
    print("All quality checks passed!")
    print("=" * 60)
    

if __name__ == '__main__':
    exit(main())
    
