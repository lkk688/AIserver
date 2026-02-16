#!/usr/bin/env python3
"""
Linear Regression with Raw Tensors - Level 1 Task
Manual implementation of linear regression using gradient descent on raw PyTorch tensors.

Mathematical Formulas:
    - Hypothesis: h_theta(x) = theta_0 + theta_1 * x
    - Cost Function (MSE): J(theta) = (1/2m) * sum((h_theta(x_i) - y_i)^2)
    - Gradient Descent Update: theta = theta - lr * grad
        where grad = (1/m) * sum((h_theta(x_i) - y_i) * x_i) for theta_1
        and grad = (1/m) * sum(h_theta(x_i) - y_i) for theta_0
"""

import torch
import sys
import numpy as np
from pathlib import Path


def generate_synthetic_data(n_samples=100, noise_std=0.5):
    """Generate synthetic data: y = 2x + 3 + noise"""
    np.random.seed(42)
    x = np.random.uniform(-5, 5, n_samples)
    noise = np.random.normal(0, noise_std, n_samples)
    y = 2 * x + 3 + noise
    return x, y


def split_data(x, y, train_ratio=0.8):
    """Split data into train and validation sets"""
    n = len(x)
    n_train = int(n * train_ratio)
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def compute_predictions(x, theta_0, theta_1):
    """Compute predictions: h_theta(x) = theta_0 + theta_1 * x"""
    return theta_0 + theta_1 * x


def compute_gradients(x, y, theta_0, theta_1):
    """Compute gradients for MSE cost function manually"""
    m = len(x)
    predictions = compute_predictions(x, theta_0, theta_1)
    errors = predictions - y
    
    # Gradient for theta_0 (intercept)
    grad_0 = torch.mean(errors)
    
    # Gradient for theta_1 (slope)
    grad_1 = torch.mean(errors * x)
    
    return grad_0, grad_1


def compute_mse(x, y, theta_0, theta_1):
    """Compute Mean Squared Error"""
    predictions = compute_predictions(x, theta_0, theta_1)
    mse = torch.mean((predictions - y) ** 2)
    return mse.item()


def compute_r2(x, y, theta_0, theta_1):
    """Compute R-squared score"""
    predictions = compute_predictions(x, theta_0, theta_1)
    ss_res = torch.sum((y - predictions) ** 2)
    ss_tot = torch.sum((y - torch.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()


def evaluate(x, y, theta_0, theta_1):
    """
    Evaluate model on given data.
    
    Returns dict with:
        - mse: Mean Squared Error
        - r2: R-squared score
        - theta_0_error: |theta_0 - 3.0|
        - theta_1_error: |theta_1 - 2.0|
    """
    mse = compute_mse(x, y, theta_0, theta_1)
    r2 = compute_r2(x, y, theta_0, theta_1)
    theta_0_error = abs(theta_0 - 3.0)
    theta_1_error = abs(theta_1 - 2.0)
    
    return {
        'mse': mse,
        'r2': r2,
        'theta_0_error': theta_0_error,
        'theta_1_error': theta_1_error
    }


def train(x_train, y_train, x_val, y_val, learning_rate=0.01, epochs=1000):
    """
    Train linear regression model using gradient descent.
    
    Args:
        x_train, y_train: Training data
        x_val, y_val: Validation data
        learning_rate: Learning rate for gradient descent
        epochs: Number of training iterations
    
    Returns:
        dict with loss_history, val_loss_history, final parameters, and metrics
    """
    # Initialize parameters
    theta_0 = torch.tensor(0.0, requires_grad=False)
    theta_1 = torch.tensor(0.0, requires_grad=False)
    
    loss_history = []
    val_loss_history = []
    
    for epoch in range(epochs):
        # Compute gradients
        grad_0, grad_1 = compute_gradients(x_train, y_train, theta_0, theta_1)
        
        # Update parameters (manual gradient descent)
        theta_0 = theta_0 - learning_rate * grad_0
        theta_1 = theta_1 - learning_rate * grad_1
        
        # Record losses
        train_loss = compute_mse(x_train, y_train, theta_0, theta_1)
        val_loss = compute_mse(x_val, y_val, theta_0, theta_1)
        loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}")
    
    # Compute final metrics
    train_metrics = evaluate(x_train, y_train, theta_
