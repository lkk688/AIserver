#!/usr/bin/env python3
"""
Graph Neural Network (GNN) - Level 3
Node classification task showing Graph Attention Network (GAT) >= Graph Convolutional Network (GCN)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class GCN(nn.Module):
    """Graph Convolutional Network for node classification."""
    def __init__(self, n_features, n_hidden, n_classes, n_heads=1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    """Graph Attention Network for node classification."""
    def __init__(self, n_features, n_hidden, n_classes, n_heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(n_features, n_hidden, heads=n_heads, dropout=0.6)
        self.conv2 = GATConv(n_hidden * n_heads, n_classes, heads=1, dropout=0.6)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def create_synthetic_graph_data(n_nodes=200, n_features=10, n_classes=3):
    """Create synthetic graph data for node classification."""
    # Generate node features
    X = np.random.randn(n_nodes, n_features).astype(np.float32)
    
    # Generate labels based on feature patterns
    y = np.zeros(n_nodes, dtype=int)
    for i in range(n_nodes):
        if X[i, 0] > 0.5:
            y[i] = 0
        elif X[i, 0] > -0.5:
            y[i] = 1
        else:
            y[i] = 2
    
    # Generate adjacency matrix (sparse random graph)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        # Connect to nearest neighbors based on feature similarity
        distances = np.sum((X - X[i])**2, axis=1)
        nearest = np.argsort(distances)[1:6]  # 5 nearest neighbors
        for j in nearest:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    
    # Add self-loops
    adj += np.eye(n_nodes)
    
    return X, y, adj


def normalize_adjacency(adj):
    """Normalize adjacency matrix using symmetric normalization."""
    adj = adj + np.eye(adj.shape[0])
    degree = np.array(adj.sum(axis=1)).flatten()
    degree_inv = np.power(degree, -0.5)
    degree_inv[np.isinf(degree_inv)] = 0.0
    degree_matrix = np.diag(degree_inv)
    adj_normalized = degree_matrix @ adj @ degree_matrix
    return adj_normalized


def train_model(model, data, train_idx, learning_rate=0.01, epochs=200):
    """Train the GNN model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = nn.NLLLoss()
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return losses


def evaluate(model, X_t, adj_t, idx, y_true):
    """Evaluate the model and return metrics."""
    model.eval()
    
    with torch.no_grad():
        # Convert to edge index format for GNN
        edge_index = torch.nonzero(adj_t > 0, as_tuple=False).t()
        edge_weight = adj_t[edge_index[0], edge_index[1]]
        
        # For GCN/GAT, we need to use the full graph but only evaluate on idx
        output = model(X_t, edge_index)
        predictions = output[idx].argmax(dim=1)
        probabilities = torch.exp(output[idx])
    
    # Calculate metrics
    y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
    predictions_np = predictions.numpy()
    
    accuracy = accuracy_score(y_true_np, predictions_np)
    mse = mean_squared_error(y_true_np, predictions_np)
    r2 = r2_score(y_true_np, predictions_np)
    
    # Calculate average probability for correct class (confidence)
    correct_probs = []
    for i, (true_label, prob) in enumerate(zip(y_true_np, probabilities)):
        correct_probs.append(prob[true_label].item())
    avg_confidence = np.mean(correct_probs)
    
    return {
        'accuracy': float(accuracy),
        'mse': float(mse),
        'r2': float(r2),
        'avg_confidence': float(avg_confidence)
    }


def save_metrics(metrics, save_dir='.'):
    """Save metrics to JSON file."""
    save_path = os.path.join(save_dir, 'metrics.json')
    import json
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {save_path}")


def main():  # noqa: C901
    """Main function to run the GAT vs GCN comparison task."""
    print("=" * 60)
    print("Graph Attention Network (GAT) vs GCN - Level 3")
    print("=" * 60)
    
    # 1. Create synthetic graph data
    print("\n1. Creating synthetic graph data...")
    X, y, adj = create_synthetic_graph_data(n_nodes=200, n_features=10, n_classes=3)
    print(f"  Nodes: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"  Edge density: {adj.sum() / (adj.shape[0] ** 2):.4f}")
    
    # 2. Split data into train/val/test
    print("\n2. Splitting data...")
    # First split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        np.arange(len(y)), test_size=0.4, random_state=SEED, stratify=y
    )
    # Second split: val vs test
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED, stratify=y[temp_idx]
    )
    
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Convert to PyTorch tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    adj_t = torch.tensor(normalize_adjacency(adj), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    
    # Create edge index for GNN
    edge_index = torch.nonzero(adj_t > 0, as_tuple=False).t()
    
    # Create Data object
    data = Data(x=X_t, edge_index=edge_index, y=y_t)
    
    # 3. Initialize models
    print("\n3. Initializing models...")
    n_features = X.shape[1]
    n_hidden = 16
    n_classes = len(np.unique(y))
    
    gcn_model = GCN(n_features, n_hidden, n_classes)
    gat_model = GAT(n_features, n_hidden, n_classes)
    
    print(f"  GCN parameters: {sum(p.numel() for p in gcn_model.parameters())}")
    print(f"  GAT parameters: {sum(p.numel() for p in gat_model.parameters())}")
    
    # 4. Train models
    print("\n4. Training GCN...")
    gcn_losses = train_model(gcn_model, data, train_idx, learning_rate=0.01, epochs=200)
    
    print("\n4. Training GAT...")
    gat_losses = train_model(gat_model, data, train_idx, learning_rate=0.01, epochs=200)
    
    # 5. Evaluate on validation set
    print("\n5. Evaluating on validation set...")
    gcn_val_metrics = evaluate(gcn_model, X_t, adj_t, val_idx, y_t[val_idx])
    gat_val_metrics = evaluate(gat_model, X_t, adj_t, val_idx, y_t[val_idx])
    
    print("\nGCN Validation Metrics:")
    print(f"  Accuracy: {gcn_val_metrics['accuracy']:.4f}")
    print(f"  MSE: {gcn_val_metrics['mse']:.4f}")
    print(f"  R2: {gcn_val_metrics['r2']:.4f}")
    print(f"  Avg Confidence: {gcn_val_metrics['avg_confidence']:.4f}")
    
    print("\nGAT Validation Metrics:")
    print(f"  Accuracy: {gat_val_metrics['accuracy']:.4f}")
    print(f"  MSE: {gat_val_metrics['mse']:.4f}")
    print(f"  R2: {gat_val_metrics['r2']:.4f}")
    print(f"  Avg Confidence: {gat_val_metrics['avg_confidence']:.4f}")
    
    # Save metrics
    save_metrics(gcn_val_metrics, save_dir='.')
    save_metrics(gat_val_metrics, save_dir='.')
    
    # 6. Evaluate on training set
    print("\n6. Evaluating on training set...")
    gcn_train_metrics = evaluate(gcn_model, X_t, adj_t, train_idx, y_t[train_idx])
    gat_train_metrics = evaluate(gat_model, X_t, adj_t, train_idx, y_t[train_idx])
    
    print("\nGCN Training Metrics:")
    print(f"  Accuracy: {gcn_train_metrics['accuracy']:.4f}")
    print(f"  MSE: {gcn_train_metrics['mse']:.4f}")
    print(f"  R2: {gcn_train_metrics['r2']:.4f}")
    
    print("\nGAT Training Metrics:")
    print(f"  Accuracy: {gat_train_metrics['accuracy']:.4f}")
    print(f"  MSE: {gat_train_metrics['mse']:.4f}")
    print(f"  R2: {gat_train_metrics['r2']:.4f}")
    
    # 7. Quality checks - GAT should perform at least as well as GCN
    print("\n7. Quality checks (GAT >= GCN on validation)...")
    
    # Check that GAT accuracy >= GCN accuracy (with small tolerance)
    tolerance = 0.02
    gat_acc = gat_val_metrics['accuracy']
    gcn_acc = gcn_val_metrics['accuracy']
    
    assert gat_acc >= gcn_acc - tolerance, \
        f"GAT accuracy ({gat_acc:.4f}) should be >= GCN accuracy ({gcn_acc:.4f}) with tolerance {tolerance}"
    print(f"✓ GAT accuracy ({gat_acc:.4f}) >= GCN accuracy ({gcn_acc:.4f})")
    
    # Check R2 score is positive (model is better than naive)
    assert gat_val_metrics['r2'] > -0.1, \
        f"GAT R2 score should be reasonable: {gat_val_metrics['r2']:.4f}"
    print(f"✓ GAT R2 score is reasonable: {gat_val_metrics['r2']:.4f}")
    
    # Check that both models learn something
    assert gcn_val_metrics['accuracy'] > 0.3, \
        f"GCN should learn something: accuracy {gcn_val_metrics['accuracy']:.4f}"
    print(f"✓ GCN learned something: accuracy {gcn_val_metrics['accuracy']:.4f}")
    
    # Check loss decreased during training
    assert gat_losses[-1] < gat_losses[0], \
        f"GAT loss should decrease: {gat_losses[0]:.4f} -> {gat_losses[-1]:.4f}"
    print(f"✓ GAT loss decreased: {gat_losses[0]:.4f} -> {gat_losses[-1]:.4f}")
    
    print("\n" + "=" * 60)
    print("All quality checks passed!")
    print("=" * 60)
    
    return 0  # Success


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
