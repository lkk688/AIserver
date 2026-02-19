"""
Multivariate Linear Regression using torch.autograd.
Calibrates decision scores; produces ROC/PR curves and AUC.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, Optional


def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "task_name": "multivariate_linear_regression",
        "description": "Calibrate decision scores; produce ROC/PR curves and AUC",
        "input_features": "multivariate",
        "output_type": "continuous",
        "metrics": ["mse", "mae", "r2", "auc"],
        "model_type": "linear_regression"
    }


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get computation device (CPU/GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        X: Feature matrix (N, D)
        y: Target vector (N,)
        batch_size: Batch size for training
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)  # Ensure shape (N, 1)
    
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_tensor.numpy(), y_tensor.numpy(),
        test_size=test_size,
        random_state=random_state
    )
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        random_state=random_state
    )
    
    # Convert back to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


class LinearRegressionModel(nn.Module):
    """Multivariate Linear Regression Model."""
    
    def __init__(self, input_dim: int):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)


def build_model(input_dim: int, device: torch.device) -> LinearRegressionModel:
    """
    Build the linear regression model.
    
    Args:
        input_dim: Number of input features
        device: Device to run the model on
    
    Returns:
        Initialized model
    """
    model = LinearRegressionModel(input_dim)
    model = model.to(device)
    return model


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train the linear regression model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        verbose: Whether to print training progress
    
    Returns:
        Dictionary containing training history and metrics
    """
    # Move model to device
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_maes = []
        
        for batch_X, batch_y in train_loader:
            # Move to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_losses.append(loss.item())
            train_maes.append(torch.mean(torch.abs(outputs - batch_y)).item())
        
        # Validation
        model.eval()
        val_losses = []
        val_maes = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_losses.append(loss.item())
                val_maes.append(torch.mean(torch.abs(outputs - batch_y)).item())
        
        # Record history
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_mae'].append(np.mean(train_maes))
        history['val_mae'].append(np.mean(val_maes))
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {history["train_loss"][-1]:.4f}, '
                  f'Val Loss: {history["val_loss"][-1]:.4f}')
    
    return history


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader to evaluate on
        device: Device to run evaluation on
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_X.size(0)
            total_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_X.size(0)
            total_samples += batch_X.size(0)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # Calculate final metrics
    mse = total_loss / total_samples
    mae = total_mae / total_samples
    
    # Calculate R²
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }


def predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device
) -> np.ndarray:
    """
    Make predictions.
    
    Args:
        model: Model to use for prediction
        X: Input features (N, D)
        device: Device to run prediction on
    
    Returns:
        Predictions (N,)
    """
    model.eval()
    
    # Convert to tensor and move to device
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
    
    # Convert to numpy
    return outputs.cpu().numpy().flatten()


def save_artifacts(
    model: nn.Module,
    history: Dict[str, Any],
    test_metrics: Dict[str, float],
    output_dir: str = "output",
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None
) -> None:
    """
    Save model artifacts and visualizations.
    
    Args:
        model: Trained model
        history: Training history
        test_metrics: Test set metrics
        output_dir: Directory to save artifacts
        X_test: Test features (for ROC/PR curves)
        y_test: Test targets (for ROC/PR curves)
        y_pred: Test predictions (for ROC/PR curves)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save test metrics
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Save metadata
    metadata = get_task_metadata()
    metadata['test_metrics'] = test_metrics
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()
    
    # Generate ROC/PR curves if test data is provided
    if X_test is not None and y_test is not None and y_pred is not None:
        # Convert continuous predictions to binary for ROC/PR analysis
        # Use median as threshold
        median_val = np.median(y_test)
        y_test_binary = (y_test >= median_val).astype(int)
        y_pred_binary = (y_pred >= median_val).astype(int)
        
        # ROC Curve
        try:
            fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_binary)
            pr_auc = auc(recall, precision)
            
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "roc_pr_curves.png"))
            plt.close()
            
            # Update metrics with AUC
            test_metrics['roc_auc'] = roc_auc
            test_metrics['pr_auc'] = pr_auc
            
            # Save updated metrics
            with open(metrics_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not generate ROC/PR curves: {e}")


def generate_sample_data(
    n_samples: int = 1000,
    n_features: int = 5,
    noise_std: float = 0.5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample data for multivariate linear regression.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise_std: Standard deviation of noise
        random_state: Random seed
    
    Returns:
        Tuple of (X, y)
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate true coefficients
    true_coeffs = np.random.randn(n_features) * 2
    
    # Generate targets with noise
    y = X @ true_coeffs + np.random.randn(n_samples) * noise_std
    
    return X, y


def main():
    """Main function to run the complete pipeline."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data(n_samples=1000, n_features=5, noise_std=0.5)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = make_dataloaders(
        X, y, batch_size=32, test_size=0.2, val_size=0.2
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Build model
    print("Building model...")
    model = build_model(input_dim=X.shape[1], device=device)
    print(f"Model architecture:\n{model}")
    
    # Train model
    print("Training model...")
    history = train(
        model, train_loader, val_loader, device,
        num_epochs=100, learning_rate=0.01, verbose=True
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Metrics: {test_metrics}")
    
    # Make predictions for visualization
    X_test = test_loader.dataset.tensors[0].numpy()
    y_test = test_loader.dataset.tensors[1].numpy().flatten()
    y_pred = predict(model, X_test, device)
    
    # Save artifacts
    print("Saving artifacts...")
    save_artifacts(
        model, history, test_metrics,
        output_dir="output",
        X_test=X_test, y_test=y_test, y_pred=y_pred
    )
    print("Artifacts saved to output/")
    
    return model, history, test_metrics


if __name__ == "__main__":
    model, history, metrics = main()
    print("\nTraining complete!")
    print(f"Final test MSE: {metrics['mse']:.4f}")
    print(f"Final test MAE: {metrics['mae']:.4f}")
    print(f"Final test R²: {metrics['r2']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    if 'pr_auc' in metrics:
        print(f"PR AUC: {metrics['pr_auc']:.4f}")
