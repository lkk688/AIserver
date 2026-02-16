#!/usr/bin/env python3
"""
GraphSAGE (Graph Sample and Aggregate) - Level 2
A simplified implementation of GraphSAGE for node classification (no external graph libs).
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from pathlib import Path


class GraphSAGE(nn.Module):
    """GraphSAGE model for node classification."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, sample_size=5):
        super(GraphSAGE, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.sample_size = sample_size
        
        # Layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))
        
        self.dropout = dropout
    
    def sample_and_aggregate(self, x, edge_index):
        """Sample neighbors and aggregate information (self + neighbor mean)."""
        num_nodes = x.size(0)
        aggregated = torch.zeros((num_nodes, x.size(1)), dtype=torch.float32)
        
        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[dst].append(src)  # dst receives from src
        
        # For each node, sample neighbors and aggregate
        for node in range(num_nodes):
            neighbors = adj_list[node]
            if len(neighbors) == 0:
                aggregated[node] = x[node]  # No neighbors, just self
            else:
                # Sample neighbors
                sample_size = min(self.sample_size, len(neighbors))
                if len(neighbors) > sample_size:
                    sampled_indices = np.random.choice(neighbors, size=sample_size, replace=False)
                else:
                    sampled_indices = neighbors
                
                # Average aggregation
                if len(sampled_indices) > 0:
                    neighbor_features = x[list(sampled_indices)]
                    aggregated[node] = neighbor_features.mean(dim=0)
                else:
                    aggregated[node] = x[node]
        
        return aggregated
    
    def forward(self, x, edge_index):
        """Forward pass through the network."""
        for i, layer in enumerate(self.layers):
            # Sample and aggregate neighbors
            x_agg = self.sample_and_aggregate(x, edge_index)
            
            # Concatenate self and aggregated features
            x = torch.cat([x, x_agg], dim=1)
            
            # Apply linear transformation
            x = layer(x)
            
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return F.log_softmax(x, dim=1)


def generate_synthetic_graph(num_nodes=500, num_features=10, num_classes=3, edge_density=0.1, noise=0.1):
    """Generate synthetic graph data for node classification."""
    # Generate node features (normalized)
    X = np.random.randn(num_nodes, num_features)
    
    # Generate true class assignments based on feature patterns (deterministic)
    class_weights = np.random.randn(num_features, num_classes)
    class_scores = X @ class_weights
    class_scores += np.random.randn(num_nodes, num_classes) * noise
    
    # Assign classes based on highest score
    y = torch.tensor(class_scores.argmax(axis=1), dtype=torch.long)
    
    # Generate edges based on node similarity (preferential attachment)
    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, min(i + 15, num_nodes)):  # Limit connections for efficiency
            # Connect nodes with similar features
            similarity = np.exp(-np.linalg.norm(X[i] - X[j]))
            if np.random.random() < edge_density * similarity * 5:
                edge_index.append([i, j])  # Undirected graph (both directions)
                edge_index.append([j, i])  # Add reverse direction for undirected
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    X = torch.tensor(X, dtype=torch.float32)
    return {'x': X, 'edge_index': edge_index, 'y': y}


def train(model, graph, train_mask, learning_rate=0.01, weight_decay=5e-4, epochs=200):
    """Train the GraphSAGE model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph['x'], graph['edge_index'])
        loss = criterion(out[train_mask], graph['y'][train_mask])
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    print(f"Final training loss: {losses[-1]:.4f}")
    return losses


def evaluate(model, graph, mask, y_true):
    """Evaluate the model on given data mask."""
    model.eval()
    with torch.no_grad():
        out = model(graph['x'], graph['edge_index'])
        pred = out[mask].argmax(dim=1).numpy()
        y_true_masked = y_true[mask]
        probs = out[mask].exp().numpy()  # Convert log_probs to probs
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_masked, pred)
    
    # For regression-like metrics, use one-hot encoding
    num_classes = out.size(1)
    y_onehot = np.zeros((len(y_true_masked), num_classes))
    y_onehot[np.arange(len(y_true_masked)), y_true_masked] = 1.0
    
    # Use probabilities for MSE and R2 (treat as regression on class probabilities)
    mse = mean_squared_error(y_onehot, probs)
    r2 = r2_score(y_onehot, probs)
    
    return {
        'accuracy': float(accuracy),
        'mse': float(mse),
        'r2': float(r2),
        'num_samples': int(len(y_true_masked))
    }


def save_metrics(metrics, save_dir='.'):
    """Save metrics to JSON file."""
    save_path = Path(save_dir) / 'metrics.json'
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {save_path}")


def main():  # noqa: C901
    """Main function to run the GraphSAGE task."""
    print("=" * 60)
    print("GraphSAGE (Graph Sample and Aggregate) - Level 2")
    print("Node Classification without external graph libraries")
    print("=" * 60)
    
    # 1. Generate synthetic graph data
    print("\n1. Generating synthetic graph data...")
    graph = generate_synthetic_graph(
        num_nodes=500,
        num_features=10,
        num_classes=3,
        edge_density=0.1,
        noise=0.1
    )
    print(f"Nodes: {graph['x'].size(0)}, Features: {graph['x'].size(1)}")
    print(f"Edges: {graph['edge_index'].size(1) // 2} (undirected)")
    print(f"Classes: {graph['y'].max().item() + 1}")
    
    # 2. Split data
    print("\n2. Preparing train/validation/test masks (70/15/15)...")
    n = graph['x'].size(0)
    indices = np.random.permutation(n)
    train_size = int(n * 0.70)
    val_size = int(n * 0.15)
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
    
    # 3. Initialize model
    print("\n3. Initializing GraphSAGE model...")
    model = GraphSAGE(
        in_channels=graph['x'].size(1),
        hidden_channels=32,
        out_channels=graph['y'].max().item() + 1,
        num_layers=2,
        dropout=0.5,
        sample_size=5
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 4. Train model
    print("\n4. Training GraphSAGE model...")
    train_losses = train(model, graph, train_mask, learning_rate=0.01, weight_decay=5e-4, epochs=100)
    
    # 5. Evaluate on training data
    print("\n5. Evaluating on training data...")
    train_metrics = evaluate(model, graph, train_mask, graph['y'].numpy())
    print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Train MSE: {train_metrics['mse']:.6f}")
    print(f"Train R²: {train_metrics['r2']:.4f}")
    
    # 6. Evaluate on validation data
    print("\n6. Evaluating on validation data...")
    val_metrics = evaluate(model, graph, val_mask, graph['y'].numpy())
    print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Val MSE: {val_metrics['mse']:.6f}")
    print(f"Val R²: {val_metrics['r2']:.6f}")
    
    # 7. Save metrics
    print("\n7. Saving metrics to metrics.json...")
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'final_train_loss': train_losses[-1]
    }
    save_metrics(metrics, save_dir='.')
    
    # 8. Quality checks
    print("\n8. Quality checks...")
    
    # Check that training loss decreased
    loss_decreased = train_losses[-1] < train_losses[0]
    print(f"✓ Training loss decreased: {train_losses[0]:.4f} -> {train_losses[-1]:.4f} ({'✓' if loss_decreased else '✗'})")
    
    # Check accuracy threshold (should be above random baseline)
    num_classes = graph['y'].max().item() + 1
    random_baseline = 1.0 / num_classes
    accuracy_ok = val_metrics['accuracy'] > random_baseline + 0.1
    print(f"✓ Validation accuracy ({val_metrics['accuracy']:.4f}) > random baseline ({random_baseline:.4f} + 0.1): {'✓' if accuracy_ok else '✗'}")
    
    # Check R² score (should be positive)
    r2_ok = val_metrics['r2'] > 0.0
    print(f"✓ Validation R² ({val_metrics['r2']:.6f}) > 0: {'✓' if r2_ok else '✗'}")
    
    # Final quality assertion
    assert loss_decreased, "Training loss should decrease"
    assert accuracy_ok, f"Validation accuracy should be above random baseline + 0.1"
    assert r2_ok, "Validation R² should be positive"
    
    print("\n✓ All quality checks passed!")
    print("=" * 60)
    
    return 0  # Success


if __name__ == '__main__':
    exit(main())
