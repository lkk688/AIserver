#!/usr/bin/env python3
"""
SVM with RBF Kernel (Dual Formulation)
Supports nonlinear classification using the kernel trick.
Complexity: O(n^3) for training with n samples due to quadratic programming.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class SVMRBFDual:
    """
    Support Vector Machine with RBF Kernel using dual formulation.
    Uses quadratic programming to solve the optimization problem.
    """
    
    def __init__(self, C: float = 1.0, gamma: float = 1.0, max_iter: int = 1000, tol: float = 1e-3):
        """
        Initialize SVM with RBF kernel.
        
        Args:
            C: Regularization parameter (higher = less regularization)
            gamma: RBF kernel parameter
            max_iter: Maximum iterations for optimization
            tol: Tolerance for convergence
        """
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.sv_indices = None
    
    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute RBF (Gaussian) kernel matrix.
        
        Args:
            X1: First set of samples (n1, d)
            X2: Second set of samples (n2, d)
            
        Returns:
            Kernel matrix (n1, n2)
        """
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
        distances = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * distances)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMRBFDual':
        """
        Train the SVM using simplified SMO-like optimization.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) - should be {-1, +1}
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Initialize alpha
        self.alpha = np.zeros(n_samples)
        
        # Compute kernel matrix
        K = self._kernel(X, X)
        
        # Simplified optimization using gradient-based approach
        # This is a simplified version - in production use quadratic programming
        for iteration in range(self.max_iter):
            alpha_old = self.alpha.copy()
            
            # Update each alpha_i using gradient information
            for i in range(n_samples):
                # Compute prediction error
                error_i = np.dot(K[i], self.alpha * y) - y[i]
                
                # Update alpha_i with projection onto [0, C]
                self.alpha[i] += 0.01 * error_i  # Learning rate
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)
            
            # Check convergence
            if np.max(np.abs(self.alpha - alpha_old)) < self.tol:
                break
        
        # Compute bias term b using support vectors
        sv_mask = self.alpha > 1e-6
        self.sv_indices = np.where(sv_mask)[0]
        
        if np.sum(sv_mask) > 0:
            # Use support vectors to compute bias
            K_sv = self._kernel(X[sv_mask], X)
            predictions = np.dot(K_sv, self.alpha * y)
            
            # For SVs: y_i * (prediction_i + b) = 1 for margin SVs
            # For exact SVs (alpha < C): y_i * (prediction_i + b) = 1
            margin_mask = (self.alpha > 1e-6) & (self.alpha < self.C - 1e-6)
            
            if np.sum(margin_mask) > 0:
                K_margin = self._kernel(X[margin_mask], X)
                predictions_margin = np.dot(K_margin, self.alpha * y)
                # Fix broadcasting: predictions_margin has shape (n_margin,), y[margin_mask] has shape (n_margin,)
                self.b = np.mean((1 - y[margin_mask] * predictions_margin) / y[margin_mask])
            else:
                # Fallback: use all support vectors
                self.b = np.mean((y[sv_mask] - predictions) / y[sv_mask])
        else:
            self.b = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Samples (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        K = self._kernel(X, self.X_train)
        decision = np.dot(K, self.alpha * self.y_train) + self.b
        return np.sign(decision)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function for samples in X.
        
        Args:
            X: Samples (n_samples, n_features)
            
        Returns:
            Decision values (n_samples,)
        """
        K = self._kernel(X, self.X_train)
        return np.dot(K, self.alpha * self.y_train) + self.b


class SVMLinear:
    """
    Linear SVM for baseline comparison.
    """
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, tol: float = 1e-3):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMLinear':
        n_samples, n_features = X.shape
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Initialize alpha
        self.alpha = np.zeros(n_samples)
        
        # Linear kernel: K = X X^T
        K = np.dot(X, X.T)
        
        # Optimization
        for iteration in range(self.max_iter):
            alpha_old = self.alpha.copy()
            
            for i in range(n_samples):
                error_i = np.dot(K[i], self.alpha * y) - y[i]
                self.alpha[i] += 0.01 * error_i
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)
            
            if np.max(np.abs(self.alpha - alpha_old)) < self.tol:
                break
        
        # Compute bias
        sv_mask = self.alpha > 1e-6
        
        if np.sum(sv_mask) > 0:
            K_sv = np.dot(X[sv_mask], X.T)
            predictions = np.dot(K_sv, self.alpha * y)
            margin_mask = (self.alpha > 1e-6) & (self.alpha < self.C - 1e-6)
            
            if np.sum(margin_mask) > 0:
                self.b = np.mean((1 - y[margin_mask] * np.dot(K[margin_mask], self.alpha * y)) / y[margin_mask])
            else:
                self.b = np.mean((y[sv_mask] - predictions) / y[sv_mask])
        else:
            self.b = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        K = np.dot(X, self.X_train.T)
        decision = np.dot(K, self.alpha * self.y_train) + self.b
        return np.sign(decision)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        K = np.dot(X, self.X_train.T)
        return np.dot(K, self.alpha * self.y_train) + self.b


def generate_nonlinear_dataset(n_samples: int = 80, noise: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a nonlinear classification dataset.
    
    Args:
        n_samples: Number of samples to generate
        noise: Amount of noise to add
        
    Returns:
        X: Features (n_samples, 2)
        y: Labels (n_samples,) with values {-1, +1}
    """
    np.random.seed(42)
    
    # Create two interleaving spirals or circles
    n_per_class = n_samples // 2
    
    # Class 0: inner circle
    r0 = np.random.rand(n_per_class) * 0.5 + 0.3
    theta0 = np.random.rand(n_per_class) * 2 * np.pi
    X0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])
    y0 = np.ones(n_per_class)
    
    # Class 1: outer ring
    r1 = np.random.rand(n_per_class) * 0.5 + 1.0
    theta1 = np.random.rand(n_per_class) * 2 * np.pi
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    y1 = -np.ones(n_per_class)
    
    # Add noise
    X0 += np.random.randn(*X0.shape) * noise
    X1 += np.random.randn(*X1.shape) * noise
    
    # Combine
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def evaluate(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the model and return metrics.
    
    Args:
        model: Trained SVM model
        X: Features
        y: True labels
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    y_decision = model.decision_function(X)
    
    # Convert to binary for MSE calculation (map {-1,+1} to {0,1})
    y_binary = (y + 1) / 2
    y_pred_binary = (y_pred + 1) / 2
    
    mse = mean_squared_error(y_binary, y_pred_binary)
    r2 = r2_score(y_binary, y_pred_binary)
    
    # Accuracy
    accuracy = np.mean(y_pred == y)
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'accuracy': float(accuracy)
    }


def main() -> int:
    """
    Main function to train and evaluate the SVM model.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    print("=" * 60)
    print("SVM with RBF Kernel (Dual Formulation)")
    print("=" * 60)
    
    # Generate dataset
    print("\nGenerating nonlinear dataset...")
    X, y = generate_nonlinear_dataset(n_samples=80, noise=0.2)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Train Linear Kernel (Baseline)
    print("\n--- Training Linear Kernel (Baseline) ---")
    linear_model = SVMLinear(C=1.0, max_iter=1000)
    linear_model.fit(X_train, y_train)
    linear_metrics = evaluate(linear_model, X_val, y_val)
    
    print(f"Linear Kernel - MSE: {linear_metrics['mse']:.4f}, R2: {linear_metrics['r2']:.4f}, Accuracy: {linear_metrics['accuracy']:.4f}")
    
    # Train RBF Kernel
    print("\n--- Training RBF Kernel ---")
    rbf_model = SVMRBFDual(C=1.0, gamma=2.0, max_iter=1000)
    rbf_model.fit(X_train, y_train)
    rbf_metrics = evaluate(rbf_model, X_val, y_val)
    
    print(f"RBF Kernel - MSE: {rbf_metrics['mse']:.4f}, R2: {rbf_metrics['r2']:.4f}, Accuracy: {rbf_metrics['accuracy']:.4f}")
    
    # Print training metrics
    print("\n--- Training Metrics ---")
    train_metrics = evaluate(rbf_model, X_train, y_train)
    print(f"RBF Kernel (Train) - MSE: {train_metrics['mse']:.4f}, R2: {train_metrics['r2']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
    
    # Print validation metrics
    print("\n--- Validation Metrics ---")
    print(f"RBF Kernel (Val) - MSE: {rbf_metrics['mse']:.4f}, R2: {rbf_metrics['r2']:.4f}, Accuracy: {rbf_metrics['accuracy']:.4f}")
    
    # Assert quality thresholds
    print("\n--- Quality Checks ---")
    
    # RBF should outperform linear on nonlinear data
    if rbf_metrics['accuracy'] <= linear_metrics['accuracy']:
        print(f"FAIL: RBF accuracy ({rbf_metrics['accuracy']:.4f}) should be > linear accuracy ({linear_metrics['accuracy']:.4f})")
        return 1
    
    # Check R2 > 0.9 on training data
    if train_metrics['r2'] < 0.9:
        print(f"FAIL: R2 score ({train_metrics['r2']:.4f}) should be >= 0.9 on training data")
        return 1
    
    # Check MSE threshold
    if train_metrics['mse'] > 0.1:
        print(f"FAIL: MSE ({train_metrics['mse']:.4f}) should be <= 0.1 on training data")
        return 1
    
    # Check validation accuracy
    if rbf_metrics['accuracy'] < 0.85:
        print(f"FAIL:
