#!/usr/bin/env python3
"""
MLP Training Tricks - Level 3
Task: Implement evaluate() returning MSE, R2, and metrics with checkpointing
Implementation using advanced training techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def generate_data(num_samples=500, num_features=10, noise=0.1):
    """Generate synthetic multivariate data with nonlinear relationships."""
    # True parameters
    true_bias = 3.0
    true_weights = np.array([1.5, -1.2, 0.8, 2.0, -0.5, 1.0, -0.8, 0.6, 1.3, -0.4])
    
    # Generate features
    X = np.random.randn(num_samples, num_features)
    
    # Generate target with nonlinear relationships and noise
    y = (true_bias + 
         np.dot(X, true_weights) + 
         0.3 * X[:, 0] ** 2 + 
         0.2 * np.sin(X[:, 1] * np.pi) +
         noise * np.random.randn(num_samples))
    
    return X, y, true_bias, true_weights


def split_data(X, y, train_ratio=0.8):
    """Split data into training and validation sets."""
    return train_test_split(X, y, train_size=train_ratio, random_state=42)


class MLPModel(nn.Module):
    """Multi-layer perceptron with training tricks."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.2):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


def train(model, X_train, y_train, X_val, y_val, learning_rate=0.001, epochs=500, save_dir='.'):
    """
    Train the MLP model with advanced techniques.
    
    Training tricks implemented:
    - Batch normalization
    - Dropout regularization
    - Learning rate scheduling
    - Early stopping
    - Checkpointing best model
    """
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 50
    
    print(f"Training for {epochs} epochs with batch size 32...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t)
            val_loss = criterion(val_predictions, y_val_t).item()
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save best model
    save_path = Path(save_dir) / 'best_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch + 1,
        'train_loss': train_losses[-1],
        'val_loss': best_val_loss,
    }, save_path)
    print(f"Saved best model to {save_path}")
    
    return train_losses, val_losses


def evaluate(model, X, y):
    """
    Evaluate the model and return metrics.
    
    Returns:
        dict: Contains MSE, R2 score, and other metrics
    """
    model.eval()
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(X_t).numpy().flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    # Additional metrics
    mae = np.mean(np.abs(y - predictions))
    rmse = np.sqrt(mse)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mae': float(mae),
        'predictions': predictions,
        'true_values': y
    }


def save_metrics(metrics, save_dir='.'):
    """Save metrics to JSON file."""
    save_path = Path(save_dir) / 'metrics.json'
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {save_path}")


def visualize_results(train_losses, val_losses, X, y, model, save_dir='.'):
    """Generate and save training curves and predictions plot."""
    save_path = Path(save_dir)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', alpha=0.7)
    axes[0].plot(val_losses, label='Val Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training and Validation Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    # Predictions vs true values
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X)).numpy().flatten()
    
    axes[1].scatter(y, predictions, alpha=0.5, label='Predictions')
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect fit')
    axes[1].set_xlabel('True Values')
    axes[1].set_ylabel('Predictions')
    axes[1].set_title('Predictions vs True Values')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_curves.png', dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path / 'training_curves.png'}")


def main():  # noqa: C901
    """Main function to run the MLP training tricks task."""
    print("=" * 60)
    print("MLP Training Tricks - Level 3")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Generating synthetic data...")
    X, y, true_bias, true_weights = generate_data(num_samples=500, num_features=10, noise=0.1)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"True parameters: bias={true_bias:.4f}, weights shape={true_weights.shape}")
    
    # 2. Split data
    print("\n2. Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = split_data(X, y, train_ratio=0.8)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 3. Create model
    print("\n3. Creating MLP model with training tricks...")
    model = MLPModel(input_dim=10, hidden_dims=[64, 32], dropout_rate=0.2)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params} parameters")
    
    # 4. Train model
    print("\n4. Training model with advanced techniques...")
    train_losses, val_losses = train(
        model, X_train, y_train, X_val, y_val,
        learning_rate=0.001,
        epochs=500,
        save_dir='.'
    )
    
    # 5. Evaluate on training data
    print("\n5. Evaluating on training data...")
    train_metrics = evaluate(model, X_train, y_train)
    print(f"Training MSE: {train_metrics['mse']:.6f}")
    print(f"Training R²: {train_metrics['r2']:.6f}")
    print(f"Training MAE: {train_metrics['mae']:.6f}")
    
    # 6. Evaluate on validation data
    print("\n6. Evaluating on validation data...")
    val_metrics = evaluate(model, X_val, y_val)
    print(f"Validation MSE: {val_metrics['mse']:.6f}")
    print(f"Validation R²: {val_metrics['r2']:.6f}")
    print(f"Validation MAE: {val_metrics['mae']:.6f}")
    
    # 7. Save metrics
    print("\n7. Saving metrics...")
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }
    save_metrics(all_metrics, save_dir='.')
    
    # 8. Generate visualizations
    print("\n8. Generating visualizations...")
    visualize_results(train_losses, val_losses, X, y, model, save_dir='.')
    
    # 9. Quality checks
    print("\n9. Quality checks...")
    
    # Check R² score (should be > 0.9)
    assert val_metrics['r2'] > 0.9, f"Validation R² should be > 0.9, got {val_metrics['r2']:.4f}"
    print(f"✓ Validation R² > 0.9: {val_metrics['r2']:.6f}")
    
    # Check MSE threshold
    assert val_metrics['mse'] < 0.5, f"Validation MSE should be < 0.5, got {val_metrics['mse']:.6f}"
    print(f"✓ Validation MSE < 0.5: {val_metrics['mse']:.6f}")
    
    # Check loss decreased
    assert train_losses[-1] < train_losses[0], "Training loss should have decreased"
    print(f"✓ Training loss decreased: {train_losses[0]:.6f} -> {train_losses[-1]:.6f}")
    
    # Check R² is reasonable on both splits
    assert train_metrics['r2'] > 0.85, f"Training R² should be > 0.85, got {train_metrics['r2']:.4f}"
    print(f"✓ Training R² > 0.85: {train_metrics['r2']:.6f}")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
