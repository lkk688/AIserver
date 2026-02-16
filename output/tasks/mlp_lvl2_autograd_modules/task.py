"""
MLP with Autograd and PyTorch Modules - Level 2
Task: Implement train, evaluate, predict, save_artifacts functions
Implementation using PyTorch nn.Module with manual training loop.
Required functions: train, evaluate, predict, save_artifacts
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
torch.manual_seed(42)
np.random.seed(42)

# Task metadata
TASK_METADATA = {
    'name': 'mlp_lvl2_autograd_modules',
    'description': 'MLP with dropout/batchnorm options using PyTorch autograd',
    'dataset': 'mnist',
    'input_shape': [1, 28, 28],
    'output_shape': 10,
    'metrics': ['accuracy', 'mse', 'r2', 'loss', 'precision', 'recall'],
    'thresholds': {
        'accuracy': 0.95,
        'r2': 0.80,
        'mse': 0.05
    }
}
    return TASK_METADATA.copy()


def set_seed(seed=42):  # noqa: F811
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(data_dir='./data', batch_size=64, train_ratio=0.8):  # noqa: C901
    """Create data loaders for train/val/test splits."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, train_subset, val_subset, test_dataset

def load_mnist_data(data_dir='./data'):
    train_losses = []
    val_losses = []
    val_accuracies = []
    test_accuracies = []

    return train_loader, val_loader, test_loader

def evaluate(model, data_loader, device):
    # Since this is classification, we'll compute metrics on the logits/probabilities
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Get predictions
            pred = output.argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets)
    
    # For MSE and R2, use one-hot encoded predictions vs targets
    num_classes = 10
    pred_onehot = np.zeros((len(all_predictions), num_classes))
    pred_onehot[np.arange(len(all_predictions)), all_predictions] = 1
    target_onehot = np.zeros((len(all_targets), num_classes))
    target_onehot[np.arange(len(all_targets)), all_targets] = 1
    
    mse = mean_squared_error(target_onehot, pred_onehot)
    r2 = r2_score(target_onehot, pred_onehot)
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'accuracy': float(accuracy),
        'loss': float(avg_loss)
    }


def predict(model, data_loader, device):
    """Make predictions using the model."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten
            output = model(data)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    return all_predictions, all_probabilities


def save_artifacts(model, metrics, save_dir='output'):
    """Save model and metrics artifacts."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model metadata
    metadata = {
        'model_type': 'MLP',
        'input_size': 784,
        'output_size': 10,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }
    metadata_path = os.path.join(save_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_path, metrics_path, metadata_path


class MLP(nn.Module):
    """Multi-Layer Perceptron for MNIST classification."""
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10,
                 dropout=0.0, use_batchnorm=False):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

def train(model, train_loader, val_loader, device, learning_rate=0.001, epochs=10):
    """Train the MLP model."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"Training for {epochs} epochs with learning rate {learning_rate}...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, device)
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy'])
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
    
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    
    return train_losses, val_losses, val_accuracies

def main():  # noqa: C901
    """Main function to run the MLP task."""
    print("=" * 60)
    print("MLP with Autograd and PyTorch Modules - Level 2")
    
    # 1. Load data
    device = get_device()
    print(f"Using device: {device}")
    
    print("\n1. Loading MNIST data...")
    train_loader, val_loader, test_loader = make_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 2. Create base model
    print("\n2. Creating base MLP model...")
    model = MLP(input_size=784, hidden_sizes=[128, 64], num_classes=10,
                dropout=0.0, use_batchnorm=False).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params}")
    
    # 3. Train model
    print("\n3. Training base model with autograd...")
    train_losses, val_losses, val_accuracies = train(
        model, train_loader, val_loader, device,
        learning_rate=0.001, epochs=10
    )
    
    # 4. Evaluate on training data
    print("\n4. Evaluating on training data...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"Training MSE: {train_metrics['mse']:.6f}")
    print(f"Training R2: {train_metrics['r2']:.6f}")
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    
    # 5. Evaluate on validation data
    print("\n5. Evaluating on validation data...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"Validation MSE: {val_metrics['mse']:.6f}")
    print(f"Validation R2: {val_metrics['r2']:.6f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    
    # 6. Evaluate on test data
    print("\n6. Evaluating on test data...")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"Test R2: {test_metrics['r2']:.6f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # 7. Save artifacts
    print("\n7. Saving artifacts...")
    save_dir = 'output/tasks/mlp_lvl2_autograd_modules'
    model_path, metrics_path, metadata_path = save_artifacts(model, test_metrics, save_dir)
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved metadata to: {metadata_path}")
    
    # 8. Quality checks
    print("\n8. Quality checks...")
    
    # Check that loss is decreasing
    assert train_losses[-1] < train_losses[0], "Training loss should decrease"
    print(f"✓ Training loss decreased: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    
    # Check validation accuracy threshold
    val_accuracy_threshold = 0.90
    assert val_metrics['accuracy'] >= val_accuracy_threshold, \
        f"Validation accuracy should be >= {val_accuracy_threshold}, got {val_metrics['accuracy']:.4f}"
    print(f"✓ Validation accuracy >= {val_accuracy_threshold}: {val_metrics['accuracy']:.4f}")
    
    # Check test accuracy threshold
    test_accuracy_threshold = 0.90
    assert test_metrics['accuracy'] >= test
