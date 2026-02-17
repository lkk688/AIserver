"""
SVM Task: Score Calibration + ROC/PR Curves
Implements SVM with probability calibration and evaluation metrics.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from typing import Dict, Any, Tuple, List

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "task_name": "svm_score_calibration",
        "description": "SVM with score calibration and ROC/PR curve generation",
        "input_type": "tabular",
        "output_type": "binary_classification",
        "requires_calibration": True
    }

def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for training and validation."""
    # Convert to torch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class SVMModel(nn.Module):
    """Simple SVM-like linear classifier for PyTorch."""
    
    def __init__(self, input_dim: int):
        super(SVMModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def build_model(input_dim: int, device: torch.device) -> SVMModel:
    """Build and return the SVM model."""
    model = SVMModel(input_dim)
    return model.to(device)

def train(
    model: SVMModel,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.01
) -> SVMModel:
    """Train the SVM model using hinge loss."""
    model.train()
    criterion = nn.HingeEmbeddingLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # Move to device
            batch_X = batch_X.to(device)
            # Convert labels from {0,1} to {-1,1} for hinge loss
            batch_y = (batch_y * 2 - 1).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return model

def evaluate(
    model: SVMModel,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            
            # Convert to probabilities using sigmoid
            scores = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_scores.extend(scores)
            all_labels.extend(batch_y.numpy().flatten())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Calculate AUC-ROC
    try:
        roc_auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        roc_auc = 0.5
    
    # Calculate AUC-PR
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    pr_auc = auc(recall, precision)
    
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc)
    }

def predict(
    model: SVMModel,
    X: np.ndarray,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions and scores."""
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        scores = torch.sigmoid(outputs).cpu().numpy().flatten()
        predictions = (scores >= 0.5).astype(int)
    
    return predictions, scores

def calibrate_scores(
    model: SVMModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device
) -> Tuple[CalibratedClassifierCV, np.ndarray, np.ndarray]:
    """Calibrate decision scores using isotonic regression."""
    # Get raw scores from the model
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    
    with torch.no_grad():
        train_scores = model(X_train_tensor).cpu().numpy().flatten()
        val_scores = model(X_val_tensor).cpu().numpy().flatten()
    
    # Use isotonic regression for calibration - fit SVC first then use prefit
    base_estimator = SVC(kernel='linear', probability=False)
    base_estimator.fit(train_scores.reshape(-1, 1), y_train)
    
    calibrated_model = CalibratedClassifierCV(
        estimator=base_estimator,
        method='isotonic',
        cv='prefit'
    )
    
    # Fit calibrated model on the same data (since cv='prefit')
    calibrated_model.fit(train_scores.reshape(-1, 1), y_train)
    
    # Get calibrated probabilities
    calibrated_probs = calibrated_model.predict_proba(val_scores.reshape(-1, 1))[:, 1]
    
    return calibrated_model, val_scores, calibrated_probs

def compute_roc_pr_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_calibrated: np.ndarray = None
) -> Dict[str, Any]:
    """Compute ROC and PR curve metrics."""
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Results dictionary
    results = {
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist()
        },
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist()
        },
        "auc_roc": float(roc_auc),
        "auc_pr": float(pr_auc)
    }
    
    if y_calibrated is not None:
        # Calibrated metrics
        fpr_cal, tpr_cal, _ = roc_curve(y_true, y_calibrated)
        roc_auc_cal = auc(fpr_cal, tpr_cal)
        precision_cal, recall_cal, _ = precision_recall_curve(y_true, y_calibrated)
        pr_auc_cal = auc(recall_cal, precision_cal)
        
        results["calibrated"] = {
            "auc_roc": float(roc_auc_cal),
            "auc_pr": float(pr_auc_cal)
        }
    
    return results

def save_artifacts(
    model: SVMModel,
    calibrated_model: CalibratedClassifierCV,
    metrics: Dict[str, Any],
    output_path: str = "output"
) -> None:
    """Save model artifacts and metrics."""
    os.makedirs(output_path, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_path, "svm_model.pt"))
    
    # Save calibrated model using pickle
    import pickle
    with open(os.path.join(output_path, "calibrated_model.pkl"), 'wb') as f:
        pickle.dump(calibrated_model, f)
    
    # Save metrics as JSON
    with open(os.path.join(output_path, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Artifacts saved to {output_path}/")

def generate_sample_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_train: int = 800
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample binary classification data."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels with some pattern
    weights = np.random.randn(n_features)
    scores = X @ weights + np.random.randn(n_samples) * 0.5
    probabilities = 1 / (1 + np.exp(-scores))
    y = (probabilities > 0.5).astype(int)
    
    # Split data
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    return X_train, y_train, X_val, y_val

def main():
    """Main function to run the SVM task."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Generate sample data
    print("Generating sample data...")
    X_train, y_train, X_val, y_val = generate_sample_data(
        n_samples=1000,
        n_features=20,
        n_train=800
    )
    
    print(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
    
    # Check class distribution safely
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    print(f"Class distribution - Train: {dict(zip(unique_train, counts_train))}, "
          f"Val: {dict(zip(unique_val, counts_val))}")
    
    # Create dataloaders
    train_loader, val_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )
    
    # Build model
    model = build_model(input_dim=X_train.shape[1], device=device)
    
    # Train model
    print("Training SVM model...")
    model = train(model, train_loader, device, epochs=100, lr=0.01)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate(model, val_loader, device)
    print(f"Metrics: {metrics}")
    
    # Calibrate scores
    print("Calibrating scores...")
    calibrated_model, raw_scores, calibrated_scores = calibrate_scores(
        model, X_train, y_train, X_val, y_val, device
    )
    
    # Compute ROC/PR metrics
    print("Computing ROC/PR metrics...")
    results = compute_roc_pr_metrics(y_val, raw_scores, calibrated_scores)
    
    # Print final results
    print(f"\nFinal Results:")
    print(f"AUC-ROC: {results['auc_roc']:.4f}")
    print(f"AUC-PR: {results['auc_pr']:.4f}")
    if 'calibrated' in results:
        print(f"Calibrated AUC-ROC: {results['calibrated']['auc_roc']:.4f}")
        print(f"Calibrated AUC-PR: {results['calibrated']['auc_pr']:.4f}")
    
    # Save artifacts
    save_artifacts(model, calibrated_model, results)

if __name__ == "__main__":
    main()
