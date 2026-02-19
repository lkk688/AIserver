"""
Linear Regression using Raw PyTorch Tensors

Mathematical Background:
- Hypothesis: h_theta(x) = theta_0 + theta_1 * x
- Cost Function (MSE): J(theta) = (1/2m) * sum((h_theta(x_i) - y_i)^2)
- Gradient Descent Update: theta = theta - lr * grad(theta)

Where:
- theta_0 is the intercept (bias)
- theta_1 is the slope (weight)
- m is the number of training examples
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any

# Set output directory
OUTPUT_DIR = 'output/tasks/linreg_lvl1_raw_tensors'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata() -> Dict[str, Any]:
    """Return metadata about the task."""
    return {
        'task_name': 'linear_regression_raw_tensors',
        'description': 'Univariate Linear Regression using raw PyTorch tensors',
        'input_dim': 1,
        'output_dim': 1,
        'model_type': 'linear',
        'loss_type': 'mse'
    }


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    """Get the appropriate device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(
    n_samples: int = 100,
    train_ratio: float = 0.8,
    noise_std: float = 0.5,
    batch_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create synthetic dataset: y = 2x + 3 + noise
    
    Args:
        n_samples: Number of samples to generate
        train_ratio: Ratio of data to use for training
        noise_std: Standard deviation of noise
        batch_size: Batch size for dataloaders (not used in this simple implementation)
    
    Returns:
        X_train, y_train, X_val, y_val
    """
    # Generate synthetic data
    x = torch.linspace(-5, 5, n_samples).unsqueeze(1)  # Shape: (n_samples, 1)
    noise = torch.randn(n_samples, 1) * noise_std
    y = 2 * x + 3 + noise  # True relationship: y = 2x + 3
    
    # Split into train and validation
    n_train = int(n_samples * train_ratio)
    X_train, y_train = x[:n_train], y[:n_train]
    X_val, y_val = x[n_train:], y[n_train:]
    
    return X_train, y_train, X_val, y_val


def build_model(device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Build the linear regression model.
    
    Returns model parameters as a dictionary.
    For univariate linear regression: h_theta(x) = theta_0 + theta_1 * x
    """
    # Initialize parameters (theta_0 = intercept, theta_1 = slope)
    # Using small random values to start
    theta_0 = torch.randn(1, requires_grad=False).to(device)
    theta_1 = torch.randn(1, requires_grad=False).to(device)
    
    return {'theta_0': theta_0, 'theta_1': theta_1}


def predict(X: torch.Tensor, model: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Make predictions using the linear regression model.
    
    Args:
        X: Input features of shape (n_samples, 1)
        model: Dictionary containing theta_0 and theta_1
    
    Returns:
        Predictions of shape (n_samples, 1)
    """
    return model['theta_0'] + model['theta_1'] * X


def compute_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute Mean Squared Error."""
    return torch.mean((y_true - y_pred) ** 2)


def compute_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute R-squared (coefficient of determination)."""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def train(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    model: Dict[str, torch.Tensor],
    lr: float = 0.01,
    epochs: int = 1000,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train the linear regression model using manual gradient descent.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model: Model parameters dictionary
        lr: Learning rate
        epochs: Number of training epochs
        verbose: Whether to print training progress
    
    Returns:
        Dictionary with loss_history, val_loss_history, and final model
    """
    # Ensure model parameters are on the same device as training data
    device = X_train.device
    theta_0 = model['theta_0'].clone().detach().to(device).requires_grad_(False)
    theta_1 = model['theta_1'].clone().detach().to(device).requires_grad_(False)
    
    loss_history = []
    val_loss_history = []
    
    n_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = predict(X_train, {'theta_0': theta_0, 'theta_1': theta_1})
        
        # Compute MSE loss
        loss = compute_mse(y_train, y_pred)
        
        # Compute gradients manually
        # d(loss)/d(theta_0) = (2/n) * sum(y_pred - y_true)
        # d(loss)/d(theta_1) = (2/n) * sum((y_pred - y_true) * x)
        
        error = y_pred - y_train
        grad_theta_0 = (2 / n_samples) * torch.sum(error)
        grad_theta_1 = (2 / n_samples) * torch.sum(error * X_train)
        
        # Update parameters using gradient descent
        theta_0 = theta_0 - lr * grad_theta_0
        theta_1 = theta_1 - lr * grad_theta_1
        
        # Record loss
        loss_history.append(loss.item())
        
        # Compute validation loss
        y_val_pred = predict(X_val, {'theta_0': theta_0, 'theta_1': theta_1})
        val_loss = compute_mse(y_val, y_val_pred)
        val_loss_history.append(val_loss.item())
        
        # Print progress every 100 epochs
        if verbose and (epoch + 1) % 100 == 0:
            r2 = compute_r2(y_train, y_pred)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, R2: {r2.item():.4f}")
    
    # Update model
    model = {'theta_0': theta_0, 'theta_1': theta_1}
    
    return {
        'model': model,
        'loss_history': loss_history,
        'val_loss_history': val_loss_history
    }


def evaluate(
    X: torch.Tensor,
    y: torch.Tensor,
    model: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Evaluate the model on given data.
    
    Args:
        X: Input features
        y: True targets
        model: Model parameters
    
    Returns:
        Dictionary with MSE, R2 score, and parameter accuracy
    """
    y_pred = predict(X, model)
    
    mse = compute_mse(y, y_pred).item()
    r2 = compute_r2(y, y_pred).item()
    
    # Parameter accuracy (how close to true values: theta_0=3.0, theta_1=2.0)
    theta_0_error = abs(model['theta_0'].item() - 3.0)
    theta_1_error = abs(model['theta_1'].item() - 2.0)
    
    return {
        'mse': mse,
        'r2': r2,
        'theta_0_error': theta_0_error,
        'theta_1_error': theta_1_error,
        'theta_0': model['theta_0'].item(),
        'theta_1': model['theta_1'].item()
    }


def save_artifacts(
    metrics: Dict[str, Any],
    model: Dict[str, torch.Tensor],
    loss_history: list,
    val_loss_history: list
) -> None:
    """
    Save model artifacts including plots and model weights.
    
    Args:
        metrics: Evaluation metrics
        model: Model parameters
        loss_history: Training loss history
        val_loss_history: Validation loss history
    """
    # Save model weights
    torch.save(model, os.path.join(OUTPUT_DIR, 'model.pt'))
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, 'metrics.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Predictions vs true values
    plt.subplot(1, 2, 2)
    # Use validation data for visualization
    # Get device from model parameters
    device = model['theta_0'].device
    X_val = torch.linspace(-5, 5, 50).unsqueeze(1).to(device)
    y_true_val = 2 * X_val + 3
    y_pred_val = predict(X_val, model)
    
    plt.scatter(X_val.cpu().numpy(), y_true_val.cpu().numpy(), label='True', alpha=0.7)
    plt.plot(X_val.cpu().numpy(), y_pred_val.cpu().numpy(), 'r-', label='Predicted', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    
    print(f"Artifacts saved to {OUTPUT_DIR}")


def main() -> int:
    """
    Main function that trains, evaluates, and validates the linear regression model.
    
    Returns:
        0 on success, non-zero on failure
    """
    print("=" * 60)
    print("Linear Regression with Raw PyTorch Tensors")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']}")
    
    # Create dataloaders (get train/val splits)
    X_train, y_train, X_val, y_val = make_dataloaders(
        n_samples=100,
        train_ratio=0.8,
        noise_std=0.5,
        batch_size=16
    )
    
    # Move data to the appropriate device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    # Build model
    model = build_model(device)
    print(f"Initial parameters: theta_0 = {model['theta_0'].item():.4f}, theta_1 = {model['theta_1'].item():.4f}")
    print("True parameters: theta_0 = 3.0, theta_1 = 2.0")
    
    # Train model
    print("\n" + "-" * 40)
    print("Training...")
    print("-" * 40)
    
    training_result = train(
        X_train, y_train,
        X_val, y_val,
        model,
        lr=0.1,
        epochs=1000,
        verbose=True
    )
    
    final_model = training_result['model']
    loss_history = training_result['loss_history']
    val_loss_history = training_result['val_loss_history']
    
    # Evaluate on training set
    print("\n" + "-" * 40)
    print("Evaluation on Training Set")
    print("-" * 40)
    train_metrics = evaluate(X_train, y_train, final_model)
    print(f"MSE: {train_metrics['mse']:.4f}")
    print(f"R2 Score: {train_metrics['r2']:.4f}")
    print(f"Learned parameters: theta_0 = {train_metrics['theta_0']:.4f}, theta_1 = {train_metrics['theta_1']:.4f}")
    print(f"Parameter errors: theta_0_error = {train_metrics['theta_0_error']:.4f}, theta_1_error = {train_metrics['theta_1_error']:.4f}")
    
    # Evaluate on validation set
    print("\n" + "-" * 40)
    print("Evaluation on Validation Set")
    print("-" * 40)
    val_metrics = evaluate(X_val, y_val, final_model)
    print(f"MSE: {val_metrics['mse']:.4f}")
    print(f"R2 Score: {val_metrics['r2']:.4f}")
    print(f"Learned parameters: theta_0 = {val_metrics['theta_0']:.4f}, theta_1 = {val_metrics['theta_1']:.4f}")
    print(f"Parameter errors: theta_0_error = {val_metrics['theta_0_error']:.4f}, theta_1_error = {val_metrics['theta_1_error']:.4f}")
    
    # Save artifacts
    print("\n" + "-" * 40)
    print("Saving Artifacts...")
    print("-" * 40)
    
    all_metrics = {
        'train_mse': train_metrics['mse'],
        'train_r2': train_metrics['r2'],
        'val_mse': val_metrics['mse'],
        'val_r2': val_metrics['r2'],
        'theta_0': val_metrics['theta_0'],
        'theta_1': val_metrics['theta_1'],
        'theta_0_error': val_metrics['theta_0_error'],
        'theta_1_error': val_metrics['theta_1_error']
    }
    
    save_artifacts(all_metrics, final_model, loss_history, val_loss_history)
    
    # Assert quality thresholds
    print("\n" + "=" * 60)
    print("Quality Thresholds Check")
    print("=" * 60)
    
    success = True
    
    # Check R2 > 0.9 on validation
    if val_metrics['r2'] > 0.9:
        print(f"✓ R2 on validation: {val_metrics['r2']:.4f} > 0.9")
    else:
        print(f"✗ R2 on validation: {val_metrics['r2']:.4f} <= 0.9 (FAILED)")
        success = False
    
    # Check parameter error < 1.0
    if val_metrics['theta_0_error'] < 1.0 and val_metrics['theta_1_error'] < 1.0:
        print(f"✓ Parameter errors: theta_0_error = {val_metrics['theta_0_error']:.4f} < 1.0, theta_1_error = {val_metrics['theta_1_error']:.4f} < 1.0")
    else:
        print(f"✗ Parameter errors: theta_0_error = {val_metrics['theta_0_error']:.4f} >= 1.0 or theta_1_error = {val_metrics['theta_1_error']:.4f} >= 1.0 (FAILED)")
        success = False
    
    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("PASS: All quality thresholds met!")
        print("=" * 60)
        return 0
    else:
        print("FAIL: Quality thresholds not met!")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
