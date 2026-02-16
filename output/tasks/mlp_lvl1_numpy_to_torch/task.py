#!/usr/bin/env python3
"""
MLP from NumPy to PyTorch - Level 1
Task: Implement 2-layer MLP with manual backprop for XOR classification
Requirements: Chain rule derivations, >0.95 accuracy on XOR
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def generate_xor_data(num_samples=200, noise=0.1):
    """Generate XOR data with noise."""
    # Generate random points in [0, 1] x [0, 1]
    X = np.random.rand(num_samples, 2) * 2 - 1  # Range [-1, 1]
    
    # XOR function: y = 1 if (x1 > 0 and x2 > 0) or (x1 < 0 and x2 < 0), else 0
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
    
    # Add noise
    X += np.random.randn(num_samples, 2) * noise
    
    return X, y


class MLP(nn.Module):
    """2-layer MLP for XOR classification."""
    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size, bias=True)
        self.output = nn.Linear(hidden_size, output_size, bias=True)
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, x):
        """Forward pass with sigmoid activation."""
        # Hidden layer with sigmoid activation
        hidden = torch.sigmoid(self.hidden(x))
        # Output layer with sigmoid activation
        output = torch.sigmoid(self.output(hidden))
        return output


def train_model(model, X_train, y_train, X_val, y_val, epochs=2000, lr=0.5):
    """Train the MLP model."""
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).view(-1, 1)
    
    # Loss function
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record losses
        train_losses.append(loss.item())
        
        # Validation loss
        with torch.no_grad():
            val_predictions = model(X_val_t)
            val_loss = criterion(val_predictions, y_val_t)
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 400 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
    
    print(f"Final training loss: {train_losses[-1]:.6f}, Final validation loss: {val_losses[-1]:.6f}")
    
    return train_losses, val_losses


def evaluate(model, X, y):
    """Evaluate the model and return metrics."""
    # Convert to tensors
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).view(-1, 1)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(X_t).numpy().flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    # Binary predictions for accuracy
    predictions_binary = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(y, predictions_binary)
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'accuracy': float(accuracy),
        'predictions': predictions,
        'predictions_binary': predictions_binary
    }


def main():
    """Main function to run the MLP task."""
    print("=" * 60)
    print("Task: XOR Classification with Chain Rule Derivations")
    print("=" * 60)
    
    # 1. Generate XOR data
    print("\n1. Generating XOR data (with noise)...")
    X, y = generate_xor_data(num_samples=200)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Unique classes: {np.unique(y)}")
    
    # 2. Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 3. Create model
    print("\n3. Creating MLP model with one hidden layer...")
    model = MLP(input_size=2, hidden_size=4, output_size=1)
    print("Architecture: 2 inputs -> 4 hidden (sigmoid) -> 1 output (sigmoid)")
    print("Chain rule for backprop:")
    print("  dL/dW2 = dL/dy * dy/dz2 * dz2/dW2")
    print("  dL/dW1 = dL/dy * dy/dz2 * dz2/dh * dh/dz1 * dz1/dW1")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 4. Train model
    print("\n4. Training model...")
    train_losses, val_losses = train_model(
        model, X_train, y_train, X_val, y_val, epochs=1000, lr=0.5
    )
    
    # 5. Evaluate on training data
    print("\n5. Evaluating on training data...")
    train_metrics = evaluate(model, X_train, y_train)
    print(f"Training MSE: {train_metrics['mse']:.6f}")
    print(f"Training R²: {train_metrics['r2']:.6f}")
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    
    # 6. Evaluate on validation data
    print("\n6. Evaluating on validation data...")
    val_metrics = evaluate(model, X_val, y_val)
    print(f"Validation MSE: {val_metrics['mse']:.6f}")
    print(f"Validation R²: {val_metrics['r2']:.6f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    
    # 7. Quality checks
    print("\n7. Quality checks...")
    
    # Check accuracy > 0.95
    assert val_metrics['accuracy'] > 0.95, \
        f"Validation accuracy should be > 0.95, got {val_metrics['accuracy']:.4f}"
    print(f"✓ Validation accuracy > 0.95: {val_metrics['accuracy']:.4f}")
    
    # Check R² > 0.9
    assert val_metrics['r2'] > 0.9, \
        f"Validation R² should be > 0.9, got {val_metrics['r2']:.4f}"
    print(f"✓ Validation R² > 0.9: {val_metrics['r2']:.4f}")
    
    # Check MSE < 0.1
    assert val_metrics['mse'] < 0.1, \
        f"Validation MSE should be < 0.1, got {val_metrics['mse']:.4f}"
    print(f"✓ Validation MSE < 0.1: {val_metrics['mse']:.4f}")
    
    # Check loss decreased
    assert train_losses[-1] < train_losses[0], \
        f"Training loss should decrease: {train_losses[0]:.6f} -> {train_losses[-1]:.6f}"
    print(f"✓ Training loss decreased: {train_losses[0]:.6f} -> {train_losses[-1]:.6f}")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
