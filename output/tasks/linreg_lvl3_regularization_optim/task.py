#!/usr/bin/env python3
"""
Ridge Regression Task - LinReg Level 3
Implements Ridge regression with regularization for polynomial fitting.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration (cuda/cpu safe)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_data(n_samples=100, noise_std=0.5, train_ratio=0.8):
    """
    Generate nonlinear data: y = x^2 + noise
    """
    # Generate x values
    x = np.linspace(-3, 3, n_samples)
    np.random.shuffle(x)
    
    # Generate y = x^2 + noise
    y = x**2 + np.random.normal(0, noise_std, n_samples)
    
    # Split into train and validation
    split_idx = int(train_ratio * n_samples)
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    return x_train, x_val, y_train, y_val


def create_poly_features(x, degree=2):
    """
    Create polynomial features manually
    For degree=2: [1, x, x^2]
    """
    n_samples = len(x)
    features = np.ones((n_samples, degree + 1))
    for d in range(1, degree + 1):
        features[:, d] = x ** d
    return features


def ridge_objective(theta, X, y, lambda_reg):
    """
    Compute Ridge objective: J(θ) + λ∑θ_j^2
    where J(θ) = (1/2n)∑(y_pred - y)^2
    """
    n = len(y)
    y_pred = X @ theta
    mse = (1 / (2 * n)) * np.sum((y_pred - y) ** 2)
    reg_term = (lambda_reg / 2) * np.sum(theta[1:] ** 2)  # Don't regularize bias
    return mse + reg_term


def train_ridge_regression(x_train, x_val, y_train, y_val, 
                          degree=2, lambda_reg=0.1, 
                          lr=0.01, momentum=0.9, epochs=1000):
    """
    Train Ridge regression using PyTorch
    """
    # Create polynomial features
    X_train = create_poly_features(x_train, degree)
    X_val = create_poly_features(x_val, degree)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device).unsqueeze(1)
    
    # Get feature dimension
    n_features = X_train.shape[1]
    
    # Create model (Linear includes bias by default)
    model = nn.Linear(n_features, 1).to(device)
    
    # Initialize weights to small values to help with regularization
    model.weight.data.normal_(0, 0.01)
    model.bias.data.zero_()
    
    # Loss function (MSE)
    criterion = nn.MSELoss()
    
    # Optimizer with weight_decay (equivalent to Ridge regularization)
    # Note: weight_decay applies to all weights, we'll handle bias separately
    optimizer = optim.SGD([
        {'params': model.weight, 'weight_decay': lambda_reg},
        {'params': model.bias, 'weight_decay': 0}
    ], lr=lr, momentum=momentum)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Add regularization term for weight decay
        reg_loss = 0
        for param in model.weight:
            reg_loss += torch.sum(param ** 2)
        loss = loss + (lambda_reg / 2) * reg_loss
       
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record losses
        train_losses.append(loss.item())
        
        # Validation loss
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_reg_loss = 0
            for param in model.weight:
                val_reg_loss += torch.sum(param ** 2)
            val_loss = val_loss + (lambda_reg / 2) * val_reg_loss
            val_losses.append(val_loss.item())
    
    return model, train_losses, val_losses, X_train, X_val


def evaluate(model, X, y, lambda_reg=0.1):
    """
    Evaluate model and return metrics
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X).cpu().numpy().flatten()
        y_true = y.cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate Ridge objective value
    theta = np.concatenate([model.weight.cpu().numpy().flatten(), 
                           model.bias.cpu().numpy()])
    X_np = X.cpu().numpy()
    ridge_obj = ridge_objective(theta, X_np, y_true, lambda_reg)
    
    return {
        'MSE': mse,
        'R2': r2,
        'Ridge_Objective': ridge_obj
    }


def visualize_results(x_train, x_val, y_train, y_val, model, 
                     degree=2, train_losses=None, val_losses=None,
                     save_path='linreg_lvl3_fit.png'):
    """
    Create visualization of the fit and learning curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and fit
    axes[0].scatter(x_train, y_train, c='blue', label='Train data', alpha=0.6)
    axes[0].scatter(x_val, y_val, c='red', label='Val data', alpha=0.6)
    
    # Generate smooth curve for prediction
    x_smooth = np.linspace(-3, 3, 200)
    X_smooth = create_poly_features(x_smooth, degree)
    X_smooth_tensor = torch.FloatTensor(X_smooth).to(device)
    
    with torch.no_grad():
        y_smooth = model(X_smooth_tensor).cpu().numpy()
    
    axes[0].plot(x_smooth, y_smooth, 'g-', linewidth=2, label='Model prediction')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Ridge Regression Fit')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Learning curves
    if train_losses is not None:
        axes[1].plot(train_losses, label='Train Loss', linewidth=1)
        axes[1].plot(val_losses, label='Val Loss', linewidth=1)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training and Validation Loss Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def main():
    """
    Main function to run the Ridge regression task
    """
    print("=" * 60)
    print("Ridge Regression - LinReg Level 3")
    print("=" * 60)
    
    # Hyperparameters
    N_SAMPLES = 100
    NOISE_STD = 0.5
    DEGREE = 2
    LAMBDA_REG = 0.1
    LR = 0.01
    MOMENTUM = 0.9
    EPOCHS = 1000
    
    # Generate data
    print("\n[1] Generating data...")
    x_train, x_val, y_train, y_val = generate_data(
        n_samples=N_SAMPLES, 
        noise_std=NOISE_STD,
        train_ratio=0.8
    )
    print(f"    Train samples: {len(x_train)}")
    print(f"    Val samples: {len(x_val)}")
    
    # Train model
    print(f"\n[2] Training Ridge regression (λ={LAMBDA_REG})...")
    model, train_losses, val_losses, X_train, X_val = train_ridge_regression(
        x_train, x_val, y_train, y_val,
        degree=DEGREE,
        lambda_reg=LAMBDA_REG,
        lr=LR,
        momentum=MOMENTUM,
        epochs=EPOCHS
    )
    print(f"    Final training loss: {train_losses[-1]:.6f}")
    print(f"    Final validation loss: {val_losses[-1]:.6f}")
    
    # Evaluate on training data
    print("\n[3] Evaluating on training data...")
    train_metrics = evaluate(model, X_train, 
                            torch.FloatTensor(y_train).to(device), 
                            lambda_reg=LAMBDA_REG)
    print(f"    MSE: {train_metrics['MSE']:.6f}")
    print(f"    R²:  {train_metrics['R2']:.6f}")
    print(f"    Ridge Objective: {train_metrics['Ridge_Objective']:.6f}")
    
    # Evaluate on validation data
    print("\n[4] Evaluating on validation data...")
    val_metrics = evaluate(model, X_val, 
                          torch.FloatTensor(y_val).to(device), 
                          lambda_reg=LAMBDA_REG)
    print(f"    MSE: {val_metrics['MSE']:.6f}")
    print(f"    R²:  {val_metrics['R2']:.6f}")
    print(f"    Ridge Objective: {val_metrics['Ridge_Objective']:.6f}")
    
    # Check for overfitting (train vs val comparison)
    overfit_threshold = 0.1  # Allow some difference
    if abs(train_metrics['R2'] - val_metrics['R2']) > overfit_threshold:
        print(f"\n[!] Warning: Potential overfitting detected!")
        print(f"    Train R² - Val R² = {abs(train_metrics['R2'] - val_metrics['R2']):.6f}")
    else:
        print(f"\n[✓] No severe overfitting detected")
    
    # Visualize results
    print("\n[5] Creating visualization...")
    visualize_results(x_train, x_val, y_train, y_val, model, 
                     degree=DEGREE, 
                     train_losses=train_losses, 
                     val_losses=val_losses)
    
    # Quality assertions
    print("\n[6] Quality checks...")
    try:
        # R² should be > 0.9 for good fit
        assert val_metrics['R2'] > 0.9, f"Validation R² {val_metrics['R2']:.4f} < 0.9"
        print(f"    ✓ Validation R² > 0.9: {val_metrics['R2']:.4f}")
        
        # MSE should be reasonable
        assert val_metrics['MSE'] < 1.0, f"Validation MSE {val_metrics['MSE']:.4f} >= 1.0"
        print(f"    ✓ Validation MSE < 1.0: {val_metrics['MSE']:.4f}")
        
        # Check that validation loss is not much higher than training loss
        assert val_losses[-1] < train_losses[-1] * 1.5, \
            f"Validation loss too much higher than training loss"
        print(f"    ✓ No severe overfitting: val/train loss ratio = {val_losses[-1]/train_losses[-1]:.4f}")
        
        print("\n" + "=" * 60)
        print("ALL QUALITY CHECKS PASSED!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n[✗] Quality check failed: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
