#!/usr/bin/env python3
"""
SVM Multiclass OVR (One-vs-Rest) Classification on 3-class blobs.
Implements required functions: get_task_metadata, set_seed, get_device, make_dataloaders
Implements SVM with One-vs-Rest strategy for multiclass classification.
"""

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    classification_report
)
import random
import json
import os

    return X, y


def get_task_metadata():
    """Return metadata about the task."""
    return {
        "task_name": "svm_lvl3_multiclass_ovr",
        "description": "SVM Multiclass OVR Classification on 3-class blobs",
        "input_type": "tabular",
        "output_type": "classification",
        "num_classes": 3,
        "num_features": 2,
        "dataset": "make_blobs",
        "model_type": "SVM",
        "strategy": "One-vs-Rest",
        "kernel": "rbf",
        "validation_thresholds": {
            "f1_macro": 0.85,
            "accuracy": 0.85,
            "r2": 0.8,
            "mse": 0.5
        }
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get computation device (CPU or CUDA)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Create data loaders for training."""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and validation sets."""
    X_train, X_val, y_train, y_val = train_test_split(
    print("=" * 60)
    print("SVM Multiclass OVR Classification on 3-class Blobs")
    print("=" * 60)

    # Set seed for reproducibility
    print("\n[0] Setting random seed...")
    set_seed(42)
    print("    Seed set to 42")
    
    # Generate data
    print("\n[1] Generating 3-class blobs dataset...")
    print("    Training completed!")
    
    # Evaluate on train set
    # Note: For SVM, we use direct prediction rather than dataloaders
    # The make_dataloaders function is provided for compatibility with other models
    # but SVM training uses the direct sklearn approach
    
    print("\n[4] Evaluating on training set...")
    train_metrics = evaluate(model, X_train, y_train, dataset_name="train")
    print(f"    Train Accuracy: {train_metrics['train_accuracy']:.4f}")
    print(f"    ✓ Validation MSE < 0.5: {val_metrics['validation_mse']:.4f}")
    
    print("\n" + "=" * 60)
    print("Task Metadata:")
    metadata = get_task_metadata()
    print(f"    Task: {metadata['task_name']}")
    print(f"    Model: {metadata['model_type']} with {metadata['strategy']} strategy")
    print("All quality thresholds passed!")
    print("=" * 60)
    
        exit(1)


if __name__ == '__main__':
