"""
Multiclass Logistic Regression with Softmax and Cross-Entropy Loss
=================================================================

This module implements a multiclass logistic regression classifier using
PyTorch's nn.Module.

Softmax Function (for probability distribution):
$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Cross-Entropy Loss:
$$L(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class MulticlassLogisticRegression(nn.Module):
    """Multiclass logistic regression model using softmax."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super(MulticlassLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    
    def predict_proba(self, X) -> torch.Tensor:
        """Get predicted probabilities."""
        with torch.no_grad():
            outputs = self.forward(X)
            probs = torch.softmax(outputs, dim=1)
        return probs
    
    def predict(self, X):
        """Get predicted class labels (tensor)."""
        probs = self.predict_proba(X)
        return torch.argmax(probs, dim=1)


def generate_spiral_data(n_samples: int = 100, n_classes: int = 3, noise: float = 0.2, random_state: int = 42):
    """Generate spiral data for multiclass classification."""
    np.random.seed(random_state)
    
    X = np.zeros((n_samples * n_classes, 2), dtype=np.float32)
    y = np.zeros(n_samples * n_classes, dtype=int)
    
    for class_idx in range(n_classes):
        start_idx = n_samples * class_idx
        end_idx = n_samples * (class_idx + 1)
        
        # Generate spiral points with good separation
        r = np.linspace(0.1, 1.0, n_samples)
        t = np.linspace(class_idx * 4.0 / n_classes * np.pi, 
                       (class_idx + 1) * 4.0 / n_classes * np.pi, n_samples) + np.random.randn(n_samples) * noise
      
        X[start_idx:end_idx] = np.column_stack((r * np.sin(t), r * np.cos(t)))
        y[start_idx:end_idx] = class_idx
  
    return X, y


def compute_cross_entropy_loss_manual(y_proba: np.ndarray, y_true: np.ndarray) -> float:
    """Compute cross entropy loss manually."""
    n_samples = len(y_true)
    eps = 1e-15
    
    # Clip probabilities to avoid log(0)
    y_proba_clipped = np.clip(y_proba, eps, 1 - eps)
    
    # Compute cross entropy for each sample
    ce_loss = -np.log(y_proba_clipped[np.arange(n_samples), y_true])
    
    return float(np.mean(ce_loss))


def evaluate(model, X, y):
    r"""
    Evaluate the model on given data and return metrics.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        
    Returns:
        dict: Dictionary containing MSE, R2 score, and task-specific metrics
    """
    model.eval()

    # Convert to tensor if needed
    if not isinstance(X, torch.Tensor):
        X_tensor = torch.FloatTensor(X)
    else:
        X_tensor = X
    
    if not isinstance(y, torch.Tensor):
        y_tensor = torch.LongTensor(y)
    else:
        y_tensor = y
  
    # Get predictions
    y_pred = model.predict(X_tensor).numpy()
    y_proba = model.predict_proba(X_tensor).numpy()
    
    n_samples = len(y)
    n_classes = 3
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y] = 1
  
    mse = mean_squared_error(y_onehot, y_proba, squared=True)
    r2 = r2_score(y_onehot, y_proba)
    
    # Compute task-specific metrics
    accuracy = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average='macro')
    ce_loss = compute_cross_entropy_loss_manual(y_proba, y)
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'cross_entropy_loss': float(ce_loss)
    }


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, title: str, save_path: str):
    """Plot decision boundary and save to file."""
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create a grid of points
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid_points)
    
    model.eval()
    with torch.no_grad():
        predictions = model.predict(grid_tensor)
    
    # Reshape predictions
    Z = predictions.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    print("=" * 70)
    print("Multiclass Logistic Regression with Softmax and Cross-Entropy Loss")
    print("=" * 70)

    # Generate data
    print("\n1. Generating 3-class spiral data...")
    X, y = generate_spiral_data(n_samples=300, n_classes=3, noise=0.25)
    print(f"   Class distribution: {np.bincount(y)}")
    
    # Split data
    print("\n2. Splitting data into training/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train size: {len(X_train)}, Validation size: {len(X_val)}")
    
    # Create tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create model
    print("\n3. Creating multiclass logistic regression model (nn.Module)...")
    model = MulticlassLogisticRegression(input_dim=2, num_classes=3)
    print(f"   Model: {model}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    # Training loop
    print("\n4. Training model...")
    num_epochs = 200
    train_losses = []
    
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
        train_losses.append(loss.item())
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Compute metrics
            train_preds = model.predict(X_train_tensor).numpy()
            val_preds = model.predict(X_val_tensor).numpy()
            
            train_acc = accuracy_score(y_train, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
            train_f1 = f1_score(y_train, train_preds, average='macro')
            val_f1 = f1_score(y_val, val_preds, average='macro')
            
            val_loss = criterion(model(X_val_tensor), y_val_tensor).item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f'Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                  f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
    
    # Final evaluation using evaluate() function
    print("\n5. Final evaluation using evaluate() function...")
    model.eval()

    # Evaluate on both splits using the evaluate() function
    train_metrics = evaluate(model, X_train_tensor, y_train_tensor)
    val_metrics = evaluate(model, X_val_tensor, y_val_tensor)
    
    print("\n   Training Metrics:")
    print(f"     MSE: {train_metrics['mse']:.4f}")
    print(f"     R2 Score: {train_metrics['r2']:.4f}")
    print(f"     Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"     Macro F1: {train_metrics['macro_f1']:.4f}")
    print(f"     Cross-Entropy Loss: {train_metrics['cross_entropy_loss']:.4f}")
    
    print("\n   Validation Metrics:")
    print(f"     MSE: {val_metrics['mse']:.4f}")
    print(f"     R2 Score: {val_metrics['r2']:.4f}")
    print(f"     Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"     Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"     Cross-Entropy Loss: {val_metrics['cross_entropy_loss']:.4f}")
    
    print("\n6. Plotting decision boundary...")
    plot_decision_boundary(model, X_train, y_train, 
                          'Decision Boundary (Training Set)', 
                          'output/logreg_lvl2_boundary.png')
    print("   Saved: output/logreg_lvl2_boundary.png")
    
    # Quality checks using evaluate() return values
    print("\n7. Quality checks...")

    # Check validation F1 score > 0.85
    assert val_metrics['macro_f1'] > 0.85, \
        f"Validation Macro F1 ({val_metrics['macro_f1']:.4f}) < 0.85"
    print(f"   ✓ Validation Macro F1 ({val_metrics['macro_f1']:.4f}) > 0.85")

    # Check validation accuracy > 0.85
    assert val_metrics['accuracy'] > 0.85, \
        f"Validation Accuracy ({val_metrics['accuracy']:.4f}) < 0.85"
    print(f"   ✓ Validation Accuracy ({val_metrics['accuracy']:.4f}) > 0.85")

    # Check training loss decreased
    assert train_losses[-1] < train_losses[0] * 0.1, \
        f"Training loss did not decrease sufficiently: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}"
    print(f"   ✓ Training loss decreased from {train_losses[0]:.4f} to {train_losses[-1]:.4f}")
    
    print("\n" + "=" * 70)
    print("All quality checks passed! Task completed successfully. ✓")
    print("=" * 70)
    
    # Exit with success
    exit(0)
