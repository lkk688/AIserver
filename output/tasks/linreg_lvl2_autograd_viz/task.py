#!/usr/bin/env python3
"""
Linear Regression with Autograd Visualization - Level 2
Demonstrates gradient computation and parameter updates using PyTorch autograd
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score


def explain_gradient():
    """Explain the gradient of the loss function in LaTeX."""
    latex_explanation = r"""
    ### Gradient of the Loss Function
    
    For linear regression with mean squared error loss:
    
    $$
    J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
    $$
    
    where $h_\theta(x) = x^T \theta$.
    
    The gradient of the loss with respect to parameters $\theta$ is:
    
    $$
    \nabla J(\theta) = \frac{1}{m} X^T (X\theta - y)
    $$
    
    where:
    - $X$ is the feature matrix of shape $[m, n]$ (m samples, n features)
    - $y$ is the target vector of shape $[m, 1]$
    - $\theta$ is the parameter vector of shape $[n, 1]$
    - $m$ is the number of training examples
    
    This gradient tells us the direction of steepest increase in the loss function,
    so we update parameters in the opposite direction (gradient descent):
    
    $$
    \theta := \theta - \alpha \nabla J(\theta)
    $$
    
    where $\alpha$ is the learning rate.
    """
    print(latex_explanation)
    return latex_explanation


def generate_synthetic_data(n_samples=100, n_features=5, noise_std=0.5):
    """Generate synthetic multivariate linear data."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # True parameters
    true_theta = np.array([2.0, -1.5, 3.0, -0.5, 1.0])
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate targets with noise
    y = X @ true_theta + np.random.randn(n_samples) * noise_std
    
    return X, y, true_theta


def create_train_val_splits(X, y, train_ratio=0.8):
    """Split data into training and validation sets."""
    n_samples = len(y)
    n_train =
