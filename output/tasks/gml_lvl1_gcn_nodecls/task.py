#!/usr/bin/env python3
"""
GCN Node Classification - Level 1
A simple Graph Convolutional Network implementation for node classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from pathlib import Path
import json
    def reset_parameters(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.weight)
        if hasattr(self, 'bias') and self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, X, adj_normalized):
        """
        Forward pass: H' = A_norm @ H @ W
        where A_norm is the normalized adjacency matrix
        """
        # X @ W
class GCN(nn.Module):
    """Two-layer GCN for node classification."""
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        self.dropout = dropout
    
    def forward(self, X, adj_normalized):
        """Forward pass with ReLU activation and dropout."""
        x = self.gc1(X, adj_normalized)
        x = torch.relu(x)
        x = torch.dropout(x, self.dropout, train=self.training)
        x = self.gc2(x, adj_normalized)
        return torch.log_softmax(x, dim=1)
    """
    # Add self-loops
    adj_with_self_loops = adj + torch.eye(adj.size(0))

    # Compute degree matrix
    degree = torch.sum(adj_with_self_loops, dim=1)

    # Compute D^(-1/2)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    # Create diagonal matrix
    D_inv_sqrt = torch.diag(degree_inv_sqrt)

    # A_norm = D^(-1/2) * (A + I) * D^(-1/2)
    adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj_with_self_loops), D_inv_sqrt)

    return adj_normalized
    
    
def generate_synthetic_graph_data(n_nodes=150, n_features=10, n_classes=3):
    """
    Generate synthetic graph data for node classification.
    # Generate node features
    X = torch.randn(n_nodes, n_features)
    
    # Generate adjacency matrix (moderately connected graph for better message passing)
    adj = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes):
        # Connect to ~20% of other nodes for better connectivity
        n_connections = int(n_nodes * 0.2)
        available_nodes = [j for j in range(n_nodes) if j != i]
        if n_connections > len(available_nodes):
            n_connections = len(available_nodes)
        connected_nodes = np.random.choice(available_nodes, size=n_connections, replace=False)
        for j in connected_nodes:
            adj[i, j] = 1.0
            adj[j, i] = 1.0  # Undirected graph
    
    # Generate labels based on clear feature patterns for better separability
    labels = torch.zeros(n_nodes, dtype=torch.long)
    
    # Use multiple features for clearer class separation
    # Class 0: high feature 0 AND high feature 1
    # Class 1: high feature 0 AND low feature 1 OR low feature 0 AND high feature 1
    # Class 2: low feature 0 AND low feature 1
    feat0 = X[:, 0]
    feat1 = X[:, 1]
    median0 = torch.median(feat0)
    median1 = torch.median(feat1)
    
    class0_mask = (feat0 > median0) & (feat1 > median1)
    class2_mask = (feat0 <= median0) & (feat1 <= median1)
    class1_mask = ~(class0_mask | class2_mask)
    
    labels[class0_mask] = 0
    labels[class1_mask] = 1
    labels[class2_mask] = 2
    
    # Create balanced train/val/test masks
    train_ratio, val_ratio = 0.6, 0.2
    n_train = int(n_nodes * train_ratio)
    n_val = int(n_nodes * val_ratio)
    
    # Ensure balanced splits by class
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    for c in range(n_classes):
        class_indices = torch.where(labels == c)[0]
        perm = torch.randperm(len(class_indices))
        n_c_train = int(len(class_indices) * train_ratio)
        n_c_val = int(len(class_indices) * val_ratio)
        
        train_mask[class_indices[perm[:n_c_train]]] = True
        val_mask[class_indices[perm[n_c_train:n_c_train + n_c_val]]] = True
        test_mask[class_indices[perm[n_c_train + n_c_val:]]] = True
    
    return X, adj, labels, train_mask, val_mask, test_mask
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    print(f"Training for {epochs} epochs with learning rate {learning_rate}...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X, adj_normalized)  # Log probabilities
        
        # Compute loss only on training nodes
        loss = criterion(output[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    print(f"Final training loss: {loss.item():.4f}")
    return model


def evaluate(model, X, adj_normalized, labels, mask):
    model.eval()
    
    with torch.no_grad():
        log_probs = model(X, adj_normalized)
        predictions = log_probs[mask].argmax(dim=1)
        true_labels = labels[mask]
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels.numpy(), predictions.numpy())
        
        # Convert to probabilities for MSE/R² calculation
        probs = torch.exp(log_probs[mask])
        n_classes = probs.shape[1]
        true_onehot = F.one_hot(true_labels, num_classes=n_classes).float()
        
        mse = mean_squared_error(true_onehot.numpy(), probs.numpy())
        r2 = r2_score(true_onehot.numpy(), probs.numpy())
    
    return {
        'accuracy': float(accuracy),
        'mse': float(mse), 
        'r2': float(r2),
        'n_correct': int((predictions == true_labels).sum().item()),
        'n_total': int(len(true_labels))
    }
def main():  # noqa: C901
    """Main function to run the GCN node classification task."""
    print("=" * 60)
    print("GCN Node Classification - Level 1")
    print("=" * 60)
    
    # 1. Generate data
        n_nodes=150, n_features=10, n_classes=3
    )
    print(f"Nodes: {X.shape[0]}, Features: {X.shape[1]}, Classes: {labels.max().item() + 1}")
    print(f"Train/Val/Test split: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()} nodes")
    
    # 2. Normalize adjacency matrix
    print("\n2. Normalizing adjacency matrix...")
    adj_normalized = normalize_adjacency(adj)  # noqa: F841
    print(f"Adjacency normalization complete. Max value: {adj_normalized.max():.4f}")
    
    # 3. Initialize model
    model = GCN(nfeat=10, nhid=16, nclass=3, dropout=0.5)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 4. Train model with adjusted parameters
    print("\n4. Training model...")
    model = train(
        model, X, adj_normalized, labels, train_mask,  # noqa: F841
        learning_rate=0.02, weight_decay=1e-3, epochs=300
    )
    
    # 5. Evaluate on training data
    train_metrics = evaluate(model, X, adj_normalized, labels, train_mask)
    print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Train MSE: {train_metrics['mse']:.4f}")
    print(f"Train R²: {train_metrics['r2']:.4f}")
    
    # 6. Evaluate on validation data
    print("\n6. Evaluating on validation data...")
    val_metrics = evaluate(model, X, adj_normalized, labels, val_mask)
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['n_correct']}/{val_metrics['n_total']})")
    print(f"Validation MSE: {val_metrics['mse']:.4f}")
    print(f"Validation R²: {val_metrics['r2']:.4f}")
    
    print("\n7. Evaluating on test data (final evaluation)...")
    test_metrics = evaluate(model, X, adj_normalized, labels, test_mask)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test MSE: {test_metrics['mse']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    
    # 8. Quality checks
    print("\n8. Quality checks...")
    assert val_metrics['accuracy'] > 0.65, f"Validation accuracy should be > 0.65, got {val_metrics['accuracy']:.4f}"
    assert val_metrics['r2'] > 0.3, f"Validation R² should be > 0.3, got {val_metrics['r2']:.4f}"
    assert test_metrics['accuracy'] > 0.65, f"Test accuracy should be > 0.65, got {test_metrics['accuracy']:.4f}"
    print("✓ All quality checks passed!")
    
    # 9. Save metrics
    save_metrics(test_metrics, save_dir='.')
    
    print("\n" + "=" * 60)
    print("GCN Node Classification task completed successfully!")
    print("=" * 60)
    
    return 0  # Success
