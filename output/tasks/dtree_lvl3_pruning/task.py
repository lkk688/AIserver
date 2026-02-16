#!/usr/bin/env python3
"""
Decision Tree Pruning - Level 3 Task
Implements cost-complexity pruning and compares unpruned vs pruned models.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


def generate_data():
    """Generate synthetic regression data."""
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=5,
        noise=2.0,
        random_state=42
    )
    return X, y


def split_data(X, y):
    """Split data into train, validation, and test sets."""
    # First split: 64% train, 36% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.36, random_state=42
    )
    # Second split: 50% of temp for val, 50% for test (18% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_unpruned_tree(X_train, y_train):
    """Train an unpruned decision tree regressor."""
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X_train, y_train)
    return tree


def train_pruned_tree(X_train, y_train, X_val, y_val):
    """Train a pruned decision tree using cost-complexity pruning."""
    # Find optimal alpha using validation set
    alphas = np.logspace(-3, 2, 20)
    best_alpha = 0
    best_val_r2 = -np.inf
    best_tree = None
    
    for alpha in alphas:
        tree = DecisionTreeRegressor(
            random_state=42,
            ccp_alpha=alpha
        )
        tree.fit(X_train, y_train)
        y_val_pred = tree.predict(X_val)
        val_r2 = r2_score(y_val, y_val_pred)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = alpha
            best_tree = tree
    
    return best_tree, best_alpha


def evaluate_model(model, X, y, name=""):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    depth = model.get_depth()
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'depth': depth
    }
    
    return metrics


def create_visualization(results, output_path):
    """Create depth vs validation score visualization."""
    plt.figure(figsize=(10, 6))
    
    # Plot unpruned tree
    plt.scatter([results['unpruned']['depth']], 
                [results['unpruned']['val_r2']], 
                c='red', s=200, marker='^', label='Unpruned', zorder=5)
    
    # Plot pruned tree
    plt.scatter([results['pruned']['depth']], 
                [results['pruned']['val_r2']], 
                c='blue', s=200, marker='o', label='Pruned', zorder=5)
    
    plt.xlabel('Tree Depth', fontsize=12)
    plt.ylabel('Validation R² Score', fontsize=12)
    plt.title('Decision Tree Depth vs Validation Performance', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    plt.annotate(f"Depth: {results['unpruned']['depth']}\nR²: {results['unpruned']['val_r2']:.4f}",
                 xy=(results['unpruned']['depth'], results['unpruned']['val_r2']),
                 xytext=(10, 10), textcoords='offset points', color='red')
    plt.annotate(f"Depth: {results['pruned']['depth']}\nR²: {results['pruned']['val_r2']:.4f}",
                 xy=(results['pruned']['depth'], results['pruned']['val_r2']),
                 xytext=(10, -20), textcoords='offset points', color='blue')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """Main function to run the decision tree pruning task."""
    print("=" * 60)
    print("Decision Tree Pruning - Level 3 Task")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__)) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # [1] Load data
    print("\n[1] Loading data...")
    X, y = generate_data()
    print(f"    Total samples: {len(X)}")
    print(f"    Features: {X.shape[1]}")
    
    # [2] Split data
    print("\n[2] Splitting data (train/val/test)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # [3] Train unpruned tree
    print("\n[3] Training unpruned decision tree...")
    unpruned_tree = train_unpruned_tree(X_train, y_train)
    unpruned_depth = unpruned_tree.get_depth()
    print(f"    Unpruned tree depth: {unpruned_depth}")
    
    # [4] Train pruned tree
    print("\n[4] Training pruned decision tree (cost-complexity pruning)...")
    pruned_tree, best_alpha = train_pruned_tree(X_train, y_train, X_val, y_val)
    pruned_depth = pruned_tree.get_depth()
    print(f"    Best alpha: {best_alpha:.6f}")
    print(f"    Pruned tree depth: {pruned_depth}")
    
    # [5] Evaluate models
    print("\n[5] Evaluating models...")
    
    # Evaluate on all splits
    train_metrics_unpruned = evaluate_model(unpruned_tree, X_train, y_train, "train")
    val_metrics_unpruned = evaluate_model(unpruned_tree, X_val, y_val, "val")
    test_metrics_unpruned = evaluate_model(unpruned_tree, X_test, y_test, "test")
    
    train_metrics_pruned = evaluate_model(pruned_tree, X_train, y_train, "train")
    val_metrics_pruned = evaluate_model(pruned_tree, X_val, y_val, "val")
    test_metrics_pruned = evaluate_model(pruned_tree, X_test, y_test, "test")
    
    # Print results
    print("\n------------------------------------------------------------")
    print("UNPRUNED TREE RESULTS:")
    print("------------------------------------------------------------")
    print(f"  Train - MSE: {train_metrics_unpruned['mse']:.4f}, "
          f"R²: {train_metrics_unpruned['r2']:.4f}, Depth: {train_metrics_unpruned['depth']}")
    print(f"  Val   - MSE: {val_metrics_unpruned['mse']:.4f}, "
          f"R²: {val_metrics_unpruned['r2']:.4f}, Depth: {val_metrics_unpruned['depth']}")
    print(f"  Test  - MSE: {test_metrics_unpruned['mse']:.4f}, "
          f"R²: {test_metrics_unpruned['r2']:.4f}, Depth: {test_metrics_unpruned['depth']}")
    
    print("\n------------------------------------------------------------")
    print("PRUNED TREE RESULTS:")
    print("------------------------------------------------------------")
    print(f"  Train - MSE: {train_metrics_pruned['mse']:.4f}, "
          f"R²: {train_metrics_pruned['r2']:.4f}, Depth: {train_metrics_pruned['depth']}")
    print(f"  Val   - MSE: {val_metrics_pruned['mse']:.4f}, "
          f"R²: {val_metrics_pruned['r2']:.4f}, Depth: {val_metrics_pruned['depth']}")
    print(f"  Test  - MSE: {test_metrics_pruned['mse']:.4f}, "
          f"R²: {test_metrics_pruned['r2']:.4f}, Depth: {test_metrics_pruned['depth']}")
    
    # [6] Create visualization
    print("\n[6] Creating depth vs validation score visualization...")
    results = {
        'unpruned': {
            'depth': val_metrics_unpruned['depth'],
            'val_r2': val_metrics_unpruned['r2']
        },
        'pruned': {
            'depth': val_metrics_pruned['depth'],
            'val_r2': val_metrics_pruned['r2']
        }
    }
    viz_path = os.path.join(output_dir, 'results.png')
    create_visualization(results, viz_path)
    print(f"    Visualization saved to: {viz_path}")
    
    # [7] Run quality checks
    print("\n[7] Running quality checks...")
    print("------------------------------------------------------------")
    
    # Check that pruned model generalizes better (higher val R²)
    unpruned_val_r2 = val_metrics_unpruned['r2']
    pruned_val_r2 = val_metrics_pruned['r2']
    
    # Quality threshold: pruned model should have better validation R² than unpruned
    if pruned_val_r2 < unpruned_val_r2:
        print(f"✅ ASSERTION PASSED: Pruned val R² ({pruned_val_r2:.4f}) > Unpruned val R² ({unpruned_val_r2:.4f})")
    else:
        print(f"❌ ASSERTION FAILED: Pruned val R² ({pruned_val_r2:.4f}) <= Unpruned val R² ({unpruned_val_r2:.4f})")
        return 1
    
    # Additional check: unpruned should have reasonable performance
    if unpruned_val_r2 > 0.3:
        print(f"✅ ASSERTION PASSED: Unpruned val R² ({unpruned_val_r2:.4f}) > 0.3")
    else:
        print(f"❌ ASSERTION FAILED: Unpruned val R² ({unpruned_val_r2:.4f}) <= 0.3")
        return 1
    
    print("\n" + "=" * 60)
    print("All quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
