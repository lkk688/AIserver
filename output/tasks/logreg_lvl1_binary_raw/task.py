"""Logistic Regression for binary classification."""

import math as math_lib
import numpy as np
from typing import Dict, Tuple

    r"""Sigmoid activation function: $\sigma(z) = \frac{1}{1+e^{-z}}$"""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


def log_loss(y_true: float, y_pred: float, eps: float = 1e-15) -> float:    
    r"""Log-loss (cross-entropy) for binary classification: $-\left(y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})\right)$"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    class1_x2 = np.random.normal(1, noise, n_class1)
    class1_labels = np.ones(n_class1)
    
    X = np.column_stack((np.concatenate([class0_x1, class1_x1]), 
                         np.concatenate([class0_x2, class1_x2])])
    y = np.concatenate([class0_labels, class1_labels])
    
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """Train the model using gradient descent."""
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for iteration in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred_proba = np.array([sigmoid(z) for z in linear_model])  
            
            dw = (2 / n_samples) * np.dot(X.T, (y_pred_proba - y))
            db = (2 / n_samples) * np.sum(y_pred_proba - y)
            
            self.weights -= self.learning_rate * dw  
            self.bias -= self.learning_rate * db
        
        return self
        return np.array([sigmoid(z) for z in linear_model])


    def predict(self, X: np.ndarray) -> np.ndarray:        
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return np.round(proba).astype(int)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """Evaluate model performance.    
    
    Returns MSE, R2 score, and accuracy for classification.
    """
        y_proba = y_pred
        y_pred = np.round(y_proba).astype(int)
    
    mse = np.mean((y_pred - y_true) ** 2)
    
    ss_res = np.sum((y_true - y_pred) ** 2)    
    baseline_pred = np.mean(y_true)
    ss_tot = np.sum((y_true - baseline_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    accuracy = np.mean(y_pred == y_true)
    
    return {
        'mse': float(mse),
        'r2': float(max(0, r2)),  # Ensure R2 is non-negative
        'accuracy': float(accuracy)
    }
def main():
    """Main function to train, evaluate, and test the model."""
    X, y = generate_gaussian_data(n_samples=200, noise=0.25)
    
    X_std, mean, std = standardize_features(X)
    
    split_idx = int(0.8 * len(X_std))
    X_train, X_val = X_std[:split_idx], X_std[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    model = LogisticRegression(learning_rate=1.0, n_iterations=1000)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)
    train_metrics = evaluate(y_train, y_train_pred, y_train_proba)
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    val_metrics = evaluate(y_val, y_val_pred, y_val_proba)
    
    print("=== Logistic Regression Results ===")
    print("\nTraining Metrics:")
    print(f"  MSE: {train_metrics['mse']:.4f}")
    print(f"  R2:  {train_metrics['r2']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    
    print("\nValidation Metrics:")
    print(f"  MSE: {val_metrics['mse']:.4f}")
    print(f"  R2:  {val_metrics['r2']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    
    print(f"\nLearned Parameters:")
    print(f"  Weights: {model.weights}")
    print(f"  Bias: {model.bias:.4f}")
    
    assert val_metrics['accuracy'] > 0.90, f"Validation accuracy {val_metrics['accuracy']:.4f} <= 0.90"
    assert val_metrics['r2'] > 0.9, f"Validation R2 {val_metrics['r2']:.4f} <= 0.9"
    assert val_metrics['mse'] < 0.1, f"Validation MSE {val_metrics['mse']:.4f} >= 0.1"
    return 0


if __name__ == '__main__':    
    exit(main())
