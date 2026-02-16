#!/usr/bin/env python3
"""
Decision Tree Regression Task - MSE Evaluation
Synthetic piecewise function with decision tree regression
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import sys


def generate_synthetic_data(n_samples=200, noise=0.1, random_state=42):
    """Generate synthetic piecewise function data."""
    np.random.seed(random_state)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    
    # Create piecewise function with different behaviors in different regions
    y = np.piecewise(X.flatten(), 
                     [X.flatten() < 2, (X.flatten() >= 2) & (X.flatten() < 4), 
                      (X.flatten() >= 4) & (X.flatten() < 6), 
                      (X.flatten() >= 6) & (X.flatten() < 8), X.flatten() >= 8], 
                     [lambda x: 2 * x, 
                      lambda x: 4 + 0.5 * (x - 2), 
                      lambda x: 5 + 2 * np.sin(x - 4), 
                      lambda x: 5 + 1.5 * (x - 6), 
                      lambda x: 8 + 0.3 * (x - 8)])
    
    y += noise * np.random.randn(n_samples)
    
    return X, y
    

def compute_metrics(y_true, y_pred):
    """Compute standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
    
    
def train_decision_tree(X_train, y_train, max_depth=5):
    """Train a decision tree regressor."""
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model
    
    
def train_linear_regression(X_train, y_train):
    """Train a linear regression baseline."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
    
    
def evaluate(model, X, y, dataset_name=""):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred)
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  MSE:  {metrics['MSE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  R2:   {metrics['R2']:.6f}")
    
    return metrics
    
    
def main():
    """Main function to run the decision tree regression task."""
    print("=" * 60)
    print("Decision Tree Regression - MSE Evaluation")
    print("=" * 60)
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=200, noise=0.1, random_state=42)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nData split: {len(X_train)} training samples, {len(X_val)} validation samples")
    
    # Train decision tree
    print("\n--- Training Decision Tree Regressor ---")
    dtree_model = train_decision_tree(X_train, y_train, max_depth=5)
    
    # Train linear regression baseline
    print("\n--- Training Linear Regression Baseline ---")
    linreg_model = train_linear_regression(X_train, y_train)
    
    # Evaluate on validation set
    print("\n--- Evaluation on Validation Set ---")
    dtree_val_metrics = evaluate(dtree_model, X_val, y_val, "Decision Tree")
    linreg_val_metrics = evaluate(linreg_model, X_val, y_val, "Linear Regression")
    
    # Evaluate on training set
    print("\n--- Evaluation on Training Set ---")
    dtree_train_metrics = evaluate(dtree_model, X_train, y_train, "Decision Tree")
    linreg_train_metrics = evaluate(linreg_model, X_train, y_train, "Linear Regression")
    
    # Calculate improvement over baseline
    dtree_rmse = dtree_val_metrics['RMSE']
    linreg_rmse = linreg_val_metrics['RMSE']
    improvement_pct = ((linreg_rmse - dtree_rmse) / linreg_rmse) * 100
    
    print(f"\n--- Improvement Analysis ---")
    print(f"Linear Regression RMSE: {linreg_rmse:.6f}")
    print(f"Decision Tree RMSE:     {dtree_rmse:.6f}")
    print(f"Improvement: {improvement_pct:.2f}%")
    
    # Quality thresholds
    thresholds = {
        'R2_threshold': 0.9,
        'RMSE_threshold': 1.5,
        'min_improvement_pct': 5.0
    }
    
    print(f"\n--- Quality Thresholds ---")
    print(f"R2 > {thresholds['R2_threshold']}")
    print(f"RMSE < {thresholds['RMSE_threshold']}")
    print(f"Improvement > {thresholds['min_improvement_pct']}%")
    
    # Assertions
    print("\n--- Assertions ---")
    all_passed = True
    
    # Assert R2 threshold
    if dtree_val_metrics['R2'] < thresholds['R2_threshold']:
        print(f"❌ FAIL: R2 {dtree_val_metrics['R2']:.4f} < {thresholds['R2_threshold']}")
        all_passed = False
    else:
        print(f"✓ R2 {dtree_val_metrics['R2']:.4f} >= {thresholds['R2_threshold']}")
    
    # Assert RMSE threshold
    if dtree_val_metrics['RMSE'] > thresholds['RMSE_threshold']:
        print(f"❌ FAIL: RMSE {dtree_val_metrics['RMSE']:.4f} > {thresholds['RMSE_threshold']}")
        all_passed = False
    else:
        print(f"✓ RMSE {dtree_val_metrics['RMSE']:.4f} <= {thresholds['RMSE_threshold']}")
    
    # Assert improvement over baseline
    if improvement_pct < thresholds['min_improvement_pct']:
        print(f"❌ FAIL: Improvement {improvement_pct:.2f}% < {thresholds['min_improvement_pct']}%")
        all_passed = False
    else:
        print(f"✓ Improvement {improvement_pct:.2f}% >= {thresholds['min_improvement_pct']}%")
    
    if all_passed:
        print("\n✓ All quality thresholds passed!")
        return 0
    else:
        print("\n❌ Some quality thresholds failed!")
        return 1
    
    
if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
