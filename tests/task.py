"""
Deep Learning Regression: Wide vs Deep Model Comparison
Using torch.autograd for gradient computation and model training.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, mean_squared_error, mean_absolute_error
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def get_task_metadata() -> Dict[str, Any]:
    """Return metadata about the task."""
    return {
        'task_name': 'deep_learning_regression_wide_deep_comparison',
        'description': 'Compare wide and deep neural network architectures for regression',
        'framework': 'pytorch',
        'autograd_enabled': True,
        'model_types': ['wide', 'deep'],
        'metrics': ['mse', 'mae', 'auc_roc', 'auc_pr'],
        'computational_costs': ['flops', 'memory_bandwidth'],
        'created_by': 'AI Agent',
        'version': '1.0.0'
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def make_dataloaders(
    n_samples: int = 1000,
    n_features: int = 20,
    test_size: float = 0.2,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        test_size: Fraction of data for test set
        batch_size: Batch size for dataloaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, data_info)
    """
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        noise=15.0,
        random_state=42
    )
    
    # Add some non-linearity to make it more interesting
    y = y + 0.3 * X[:, 0] * X[:, 1] + 0.2 * X[:, 2]**2
    
    # Convert to float32
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).unsqueeze(1)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    data_info = {
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_features': n_features,
        'scaler': scaler
    }
    
    return train_loader, val_loader, test_loader, data_info


class WideModel(nn.Module):
    """Wide neural network model with single wide layer."""
    
    def __init__(self, n_features: int, wide_width: int = 512):
        super(WideModel, self).__init__()
        self.wide_width = wide_width
        
        # Wide layer
        self.wide = nn.Linear(n_features, wide_width)
        self.relu = nn.ReLU()
        self.output = nn.Linear(wide_width, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.wide(x))
        return self.output(x)
    
    def get_flops(self, n_samples: int) -> int:
        """Calculate FLOPs for forward pass."""
        # Wide layer: 2 * n_features * wide_width * n_samples
        wide_flops = 2 * self.wide.in_features * self.wide.out_features * n_samples
        # Output layer: 2 * wide_width * 1 * n_samples
        output_flops = 2 * self.wide_width * 1 * n_samples
        return wide_flops + output_flops
    
    def get_memory_bandwidth(self, n_samples: int) -> int:
        """Calculate memory bandwidth (bytes read/written)."""
        # Weights
        wide_weights = (self.wide.in_features * self.wide.out_features) * 4 * 2  # read + write
        output_weights = (self.wide_width * 1) * 4 * 2
        
        # Activations
        wide_activations = self.wide_width * n_samples * 4 * 3  # in, out, grad
        output_activations = 1 * n_samples * 4 * 3
        
        return wide_weights + output_weights + wide_activations + output_activations


class DeepModel(nn.Module):
    """Deep neural network model with multiple layers."""
    
    def __init__(self, n_features: int, depth: int = 6, width: int = 64):
        super(DeepModel, self).__init__()
        self.depth = depth
        self.width = width
        
        layers = []
        input_dim = n_features
        
        # Hidden layers
        for _ in range(depth):
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(width))
            input_dim = width
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        return self.output(x)
    
    def get_flops(self, n_samples: int) -> int:
        """Calculate FLOPs for forward pass."""
        # Each linear layer: 2 * in_dim * out_dim * n_samples
        flops = 0
        for layer in self.hidden[::3]:  # Get linear layers
            flops += 2 * layer.in_features * layer.out_features * n_samples
        # Output layer
        flops += 2 * self.width * 1 * n_samples
        return flops
    
    def get_memory_bandwidth(self, n_samples: int) -> int:
        """Calculate memory bandwidth (bytes read/written)."""
        memory = 0
        # Weights
        for layer in self.hidden[::3]:
            memory += layer.in_features * layer.out_features * 4 * 2
        memory += self.width * 1 * 4 * 2
        
        # Activations (each layer has in, out, grad)
        for layer in self.hidden[::3]:
            memory += layer.out_features * n_samples * 4 * 3
        memory += 1 * n_samples * 4 * 3
        
        return memory


def build_model(model_type: str = 'wide', n_features: int = 20) -> Tuple[nn.Module, Dict]:
    """
    Build a model of specified type.
    
    Args:
        model_type: 'wide' or 'deep'
        n_features: Number of input features
        
    Returns:
        Tuple of (model, model_info)
    """
    if model_type == 'wide':
        model = WideModel(n_features, wide_width=512)
        model_info = {
            'type': 'wide',
            'width': 512,
            'depth': 1,
            'n_params': sum(p.numel() for p in model.parameters())
        }
    elif model_type == 'deep':
        model = DeepModel(n_features, depth=6, width=64)
        model_info = {
            'type': 'deep',
            'width': 64,
            'depth': 6,
            'n_params': sum(p.numel() for p in model.parameters())
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model, model_info


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.001,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Train the model using torch.autograd.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Computation device
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Print training progress
        
    Returns:
        Dictionary with training history and metrics
    """
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Move model to device
    model = model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_mses = []
        
        for batch_X, batch_y in train_loader:
            # Move to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Ensure same shape and dtype
            outputs = outputs.view(-1, 1)
            batch_y = batch_y.view(-1, 1)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass with autograd
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Store metrics
            train_losses.append(loss.item())
            train_mses.append(mean_squared_error(
                batch_y.detach().cpu().numpy(),
                outputs.detach().cpu().numpy()
            ))
        
        # Validation phase
        model.eval()
        val_losses = []
        val_mses = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                outputs = outputs.view(-1, 1)
                batch_y = batch_y.view(-1, 1)
                
                loss = criterion(outputs, batch_y)
                
                val_losses.append(loss.item())
                val_mses.append(mean_squared_error(
                    batch_y.detach().cpu().numpy(),
                    outputs.detach().cpu().numpy()
                ))
        
        # Record history
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_mse'].append(np.mean(train_mses))
        history['val_mse'].append(np.mean(val_mses))
        
        # Save best model
        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            best_model_state = model.state_dict().copy()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {history['train_loss'][-1]:.4f}, "
                  f"Val Loss: {history['val_loss'][-1]:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the model.
    
    Args:
        model: Neural network model
        data_loader: Data loader for evaluation
        device: Computation device
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    criterion = nn.MSELoss()
    
    all_predictions = []
    all_targets = []
    losses = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            outputs = outputs.view(-1, 1)
            batch_y = batch_y.view(-1, 1)
            
            loss = criterion(outputs, batch_y)
            losses.append(loss.item())
            
            all_predictions.append(outputs.detach().cpu().numpy())
            all_targets.append(batch_y.detach().cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.vstack(all_predictions).flatten()
    all_targets = np.vstack(all_targets).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    
    # For ROC/PR curves, convert to binary classification problem
    # Use median as threshold
    median_target = np.median(all_targets)
    y_binary = (all_targets > median_target).astype(int)
    y_pred_binary = (all_predictions > median_target).astype(int)
    
    # Calculate AUC-ROC and AUC-PR
    try:
        auc_roc = roc_auc_score(y_binary, all_predictions)
        auc_pr = average_precision_score(y_binary, all_predictions)
    except ValueError:
        auc_roc = 0.5
        auc_pr = 0.5
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(np.sqrt(mse)),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'loss': float(np.mean(losses))
    }


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions.
    
    Args:
        model: Neural network model
        data_loader: Data loader for prediction
        device: Computation device
        
    Returns:
        Tuple of (predictions, targets)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            
            outputs = model(batch_X)
            outputs = outputs.view(-1, 1)
            
            all_predictions.append(outputs.detach().cpu().numpy())
            all_targets.append(batch_y.numpy())
    
    return np.vstack(all_predictions).flatten(), np.vstack(all_targets).flatten()


def save_artifacts(
    model: nn.Module,
    model_info: Dict,
    history: Dict,
    metrics: Dict,
    artifacts_path: str = "artifacts"
) -> None:
    """
    Save model artifacts.
    
    Args:
        model: Trained model
        model_info: Model information dictionary
        history: Training history
        metrics: Evaluation metrics
        artifacts_path: Path to save artifacts
    """
    os.makedirs(artifacts_path, exist_ok=True)
    
    # Save model state dict (CPU)
    torch.save(model.state_dict(), os.path.join(artifacts_path, "model_state.pt"))
    
    # Save model info
    with open(os.path.join(artifacts_path, "model_info.json"), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Save history (convert tensors to lists)
    history_cpu = {}
    for key, value in history.items():
        if isinstance(value, torch.Tensor):
            history_cpu[key] = value.detach().cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            history_cpu[key] = value.tolist()
        else:
            history_cpu[key] = value
    
    with open(os.path.join(artifacts_path, "history.json"), 'w') as f:
        json.dump(history_cpu, f, indent=2)
    
    # Save metrics
    with open(os.path.join(artifacts_path, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save computation costs
    costs = {
        'flops': model.get_flops(1000),  # For 1000 samples
        'memory_bandwidth': model.get_memory_bandwidth(1000)  # For 1000 samples
    }
    with open(os.path.join(artifacts_path, "costs.json"), 'w') as f:
        json.dump(costs, f, indent=2)
    
    print(f"Artifacts saved to {artifacts_path}")


def visualize_training(
    histories: Dict[str, Dict],
    save_path: str = "training_visualizations"
) -> None:
    """
    Visualize training progress and results.
    
    Args:
        histories: Dictionary of training histories for different models
        save_path: Path to save visualizations
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    for model_name, history in histories.items():
        plt.plot(history['train_loss'], label=f'{model_name} train')
        plt.plot(history['val_loss'], label=f'{model_name} val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # MSE curves
    plt.subplot(1, 2, 2)
    for model_name, history in histories.items():
        plt.plot(history['train_mse'], label=f'{model_name} train')
        plt.plot(history['val_mse'], label=f'{model_name} val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training and Validation MSE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_curves.png"), dpi=150)
    plt.close()
    
    # Plot ROC and PR curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for model_name, history in histories.items():
        if 'auc_roc' in history:
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.text(0.3, 0.7, f'{model_name}: AUC={history["auc_roc"]:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for model_name, history in histories.items():
        if 'auc_pr' in history:
            plt.text(0.1, 0.9 - 0.1*list(histories.keys()).index(model_name), 
                    f'{model_name}: AUC={history["auc_pr"]:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curves')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "roc_pr_curves.png"), dpi=150)
    plt.close()
    
    # Plot model comparison
    plt.figure(figsize=(10, 6))
    
    model_names = list(histories.keys())
    n_params = [histories[name].get('n_params', 0) for name in model_names]
    val_mses = [histories[name].get('val_mse', 0) for name in model_names]
    flops = [histories[name].get('flops', 0) for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Parameters', color=color)
    ax1.bar(x - width/2, n_params, width, label='Parameters', color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Validation MSE', color=color)
    ax2.plot(x, val_mses, 'o-', label='Val MSE', color=color, markersize=10)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title('Model Comparison: Parameters vs Performance')
    plt.xticks(x, model_names)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, "model_comparison.png"), dpi=150)
    plt.close()
    
    print(f"Visualizations saved to {save_path}")


def main():
    """Main function to run the experiment."""
    print("=" * 60)
    print("Deep Learning Regression: Wide vs Deep Model Comparison")
    print("=" * 60)
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, data_info = make_dataloaders(
        n_samples=1000,
        n_features=20,
        batch_size=32
    )
    print(f"Train samples: {data_info['n_train']}")
    print(f"Val samples: {data_info['n_val']}")
    print(f"Test samples: {data_info['n_test']}")
    
    # Train wide model
    print("\n" + "-" * 40)
    print("Training Wide Model...")
    print("-" * 40)
    
    wide_model, wide_info = build_model('wide', data_info['n_features'])
    print(f"Model info: {wide_info}")
    
    wide_history = train(
        wide_model, train_loader, val_loader, device,
        epochs=100, lr=0.001, verbose=True
    )
    
    # Add model info to history
    wide_history['n_params'] = wide_info['n_params']
    wide_history['flops'] = wide_model.get_flops(1000)
    wide_history['memory_bandwidth'] = wide_model.get_memory_bandwidth(1000)
    
    # Evaluate wide model
    wide_metrics = evaluate(wide_model, test_loader, device)
    wide_history.update(wide_metrics)
    print(f"\nWide Model Test Metrics: {wide_metrics}")
    
    # Train deep model
    print("\n" + "-" * 40)
    print("Training Deep Model...")
    print("-" * 40)
    
    deep_model, deep_info = build_model('deep', data_info['n_features'])
    print(f"Model info: {deep_info}")
    
    deep_history = train(
        deep_model, train_loader, val_loader, device,
        epochs=100, lr=0.001, verbose=True
    )
    
    # Add model info to history
    deep_history['n_params'] = deep_info['n_params']
    deep_history['flops'] = deep_model.get_flops(1000)
    deep_history['memory_bandwidth'] = deep_model.get_memory_bandwidth(1000)
    
    # Evaluate deep model
    deep_metrics = evaluate(deep_model, test_loader, device)
    deep_history.update(deep_metrics)
    print(f"\nDeep Model Test Metrics: {deep_metrics}")
    
    # Save artifacts
    print("\n" + "-" * 40)
    print("Saving Artifacts...")
    print("-" * 40)
    
    save_artifacts(wide_model, wide_info, wide_history, wide_metrics, "artifacts/wide")
    save_artifacts(deep_model, deep_info, deep_history, deep_metrics, "artifacts/deep")
    
    # Visualize
    print("\n" + "-" * 40)
    print("Generating Visualizations...")
    print("-" * 40)
    
    histories = {
        'wide': wide_history,
        'deep': deep_history
    }
    
    visualize_training(histories, "training_visualizations")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nWide Model:")
    print(f"  Parameters: {wide_info['n_params']:,}")
    print(f"  FLOPs: {wide_model.get_flops(1000):,}")
    print(f"  Memory Bandwidth: {wide_model.get_memory_bandwidth(1000):,} bytes")
    print(f"  Test MSE: {wide_metrics['mse']:.4f}")
    print(f"  Test AUC-ROC: {wide_metrics['auc_roc']:.4f}")
    print(f"  Test AUC-PR: {wide_metrics['auc_pr']:.4f}")
    
    print("\nDeep Model:")
    print(f"  Parameters: {deep_info['n_params']:,}")
    print(f"  FLOPs: {deep_model.get_flops(1000):,}")
    print(f"  Memory Bandwidth: {deep_model.get_memory_bandwidth(1000):,} bytes")
    print(f"  Test MSE: {deep_metrics['mse']:.4f}")
    print(f"  Test AUC-ROC: {deep_metrics['auc_roc']:.4f}")
    print(f"  Test AUC-PR: {deep_metrics['auc_pr']:.4f}")
    
    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
