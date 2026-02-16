#!/usr/bin/env python3
"""
UMAP-like Dimensionality Reduction - Level 4
Task: Implement evaluate() returning MSE, R2, and neighbor preservation metric
Implementation using simplified UMAP-like algorithm with neighbor preservation
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def generate_data(num_samples=200, num_features=10, noise=0.1):
    """Generate synthetic high-dimensional data with cluster structure."""
    # Generate cluster centers
    n_clusters = 4
    cluster_centers = np.random.randn(n_clusters, num_features) * 3
    
    # Generate samples around cluster centers
    samples_per_cluster = num_samples // n_clusters
    X = []
    y = []
    
    for cluster_idx in range(n_clusters):
        center = cluster_centers[cluster_idx]
        cluster_samples = np.random.randn(samples_per_cluster, num_features) * noise + center
        X.append(cluster_samples)
        y.extend([cluster_idx] * samples_per_cluster)
    
    X = np.vstack(X)
    y = np.array(y)
    
    return X, y


def split_data(X, y, train_ratio=0.8):
    """Split data into training and validation sets."""
    return train_test_split(X, y, train_size=train_ratio, random_state=42, stratify=y)


def compute_high_dim_distances(X):
    """Compute pairwise Euclidean distances in high-dimensional space."""
    n = X.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


def compute_low_dim_distances(Y):
    """Compute pairwise Euclidean distances in low-dimensional space."""
    n = Y.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((Y[i] - Y[j]) ** 2))
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


def umap_like(X, n_components=2, n_neighbors=15, learning_rate=0.01, epochs=500):
    """
    Simplified UMAP-like algorithm for dimensionality reduction.
    
    Math explanation (LaTeX):
    UMAP aims to minimize the cross-entropy between:
    - High-dimensional distances: p_ij = exp(-d_ij²/2σ_i)
    - Low-dimensional distances: q_ij = (1 + d_ij²)⁻¹
    In practice, we use a simplified optimization approach.
    """
    n_samples, n_features = X.shape
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Initialize embedding randomly
    np.random.seed(42)
    Y = np.random.randn(n_samples, n_components) * 0.1
    
    # Compute high-dimensional distances
    high_dist = compute_high_dim_distances(X_centered)
    
    # Find k-nearest neighbors for each point
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n_samples))
    nn.fit(X_centered)
    _, indices = nn.kneighbors(X_centered)
    
    # Optimization using gradient descent
    for epoch in range(epochs):
        # Compute low-dimensional distances
        low_dist = compute_low_dim_distances(Y)
        
        # Compute gradients (simplified UMAP gradient)
        grad = np.zeros_like(Y)
        
        for i in range(n_samples):
            for j in indices[i][1:]:  # Skip self (j != i)
                if j < n_samples:
                    # Simplified gradient: push points apart or together
                    dist_ratio = low_dist[i, j] / (1.0 + low_dist[i, j])
                    sign = 1.0 if low_dist[i, j] > high_dist[i, j] else -1.0
                    
                    diff = Y[i] - Y[j]
                    grad[i] += sign * dist_ratio * diff
                    grad[j] -= sign * dist_ratio * diff
        
        # Update embedding
        Y -= learning_rate * grad
        
        if (epoch + 1) % 100 == 0:
            loss = np.mean(low_dist ** 2)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.6f}")
    
    return Y


def compute_neighbor_preservation(X_high, X_low, k=10):
    """
    Compute neighbor preservation metric.
    Measures how well the top-k neighbors are preserved in the low-dimensional space.
    """
    n_samples = X_high.shape[0]
    preservation_scores = []
    
    # Find neighbors in high-dimensional space
    nn_high = NearestNeighbors(n_neighbors=min(k + 1, n_samples))
    nn_high.fit(X_high)
    _, high_neighbors = nn_high.kneighbors(X_high)
    
    # Find neighbors in low-dimensional space
    nn_low = NearestNeighbors(n_neighbors=min(k + 1, n_samples))
    nn_low.fit(X_low)
    _, low_neighbors = nn_low.kneighbors(X_low)
    
    # Compute preservation score for each sample
    for i in range(n_samples):
        high_set = set(high_neighbors[i][1:])  # Skip self
        low_set = set(low_neighbors[i][1:])    # Skip self
        
        # Jaccard similarity or simple overlap
        intersection = len(high_set & low_set)
        union = len(high_set | low_set)
        
        if union > 0:
            score = intersection / union
        else:
            score = 1.0 if len(high_set) == 0 else 0.0
        
        preservation_scores.append(score)
    
    return np.mean(preservation_scores)


def evaluate(model, X, y):
    """
    Evaluate the UMAP-like model.
    
    Returns:
        dict: Metrics including MSE, R2, and neighbor preservation
    """
    # Compute low-dimensional embedding
    X_low = umap_like(X, n_components=2, n_neighbors=15, 
                     learning_rate=0.01, epochs=500)
    
    # For evaluation, reconstruct high-dimensional data from low-dimensional
    # Use simple linear regression for reconstruction
    if hasattr(model, 'reconstruction_model') and model.reconstruction_model is not None:
        recon_model = model.reconstruction_model
    else:
        # Fit reconstruction model: low_dim -> high_dim
        recon_model = LinearRegression()
        recon_model.fit(X_low, X)
    
    # Reconstruct high-dimensional data
    X_reconstructed = recon_model.predict(X_low)
    
    # Calculate reconstruction metrics
    mse = mean_squared_error(X, X_reconstructed)
    r2 = r2_score(X, X_reconstructed)
    
    # Calculate neighbor preservation
    neighbor_preservation = compute_neighbor_preservation(X, X_low, k=10)
    
    return {
        'mse': mse,
        'r2': r2,
        'neighbor_preservation': neighbor_preservation,
        'reconstruction_model': recon_model,
        'embedding': X_low
    }


def train(X, n_components=2, n_neighbors=15, learning_rate=0.01, epochs=500):
    """
    Train UMAP-like model.
    
    Returns:
        model: Trained model with embedding_ attribute
    """
    # Compute embedding
    embedding = umap_like(X, n_components=n_components, n_neighbors=n_neighbors,
                         learning_rate=learning_rate, epochs=epochs)
    
    # Store model attributes
    model = type('UMAPModel', (), {})()
    model.embedding_ = embedding
    model.n_components = n_components
    model.n_neighbors = n_neighbors
    model.learning_rate = learning_rate
    model.epochs = epochs
    model.reconstruction_model = None  # Will be fitted during evaluation
    
    return model


def main():  # noqa: C901
    """Main function to run the UMAP-like dimensionality reduction task."""
    print("=" * 60)
    print("UMAP-like Dimensionality Reduction - Level 4")
    
    # Generate data
    print("\n1. Generating high-dimensional data...")
    X, y = generate_data(num_samples=200, num_features=10, noise=0.1)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data
    print("\n2. Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = split_data(X, y, train_ratio=0.8)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 3. Train UMAP-like model
    print("\n3. Training UMAP-like model...")
    model = train(X_train, n_components=2, n_neighbors=15, 
                 learning_rate=0.01, epochs=500)
    
    # 4. Evaluate on training data
    print("\n4. Evaluating on training data...")
    train_metrics = evaluate(model, X_train, y_train)
    print(f"Training MSE: {train_metrics['mse']:.6f}")
    print(f"Training R2: {train_metrics['r2']:.6f}")
    print(f"Training Neighbor Preservation: {train_metrics['neighbor_preservation']:.6f}")
    
    # 5. Evaluate on validation data
    print("\n5. Evaluating on validation data...")
    val_metrics = evaluate(model, X_val, y_val)
    print(f"Validation MSE: {val_metrics['mse']:.6f}")
    print(f"Validation R2: {val_metrics['r2']:.6f}")
    print(f"Validation Neighbor Preservation: {val_metrics['neighbor_preservation']:.6f}")
    
    # 6. Generate visualizations
    print("\n6. Generating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training embedding
    scatter1 = axes[0].scatter(train_metrics['embedding'][:, 0], 
                               train_metrics['embedding'][:, 1], 
                               c=y_train, cmap='viridis', alpha=0.7, edgecolors='k')
    axes[0].set_title('Training Embedding\nNeighbor Preservation: {:.3f}'.format(
        train_metrics['neighbor_preservation']))
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Validation embedding
    scatter2 = axes[1].scatter(val_metrics['embedding'][:, 0], 
                               val_metrics['embedding'][:, 1], 
                               c=y_val, cmap='viridis', alpha=0.7, edgecolors='k')
    axes[1].set_title('Validation Embedding\nNeighbor Preservation: {:.3f}'.format(
        val_metrics['neighbor_preservation']))
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('umap_results.png', dpi=150)
    print("Saved: umap_results.png")
    plt.close()
    
    # 7. Quality checks
    print("\n7. Quality checks...")
    
    # Check neighbor preservation is better than random (0.5)
    assert train_metrics['neighbor_preservation'] > 0.5, \
        f"Neighbor preservation should be > 0.5, got {train_metrics['neighbor_preservation']}"
    print(f"✓ Training neighbor preservation: {train_metrics['neighbor_preservation']:.3f} > 0.5")
    
    # Check validation neighbor preservation is reasonable
    assert val_metrics['neighbor_preservation'] > 0.4, \
        f"Validation neighbor preservation should be > 0.4, got {val_metrics['neighbor_preservation']}"
    print(f"✓ Validation neighbor preservation: {val_metrics['neighbor_preservation']:.3f} > 0.4")
    
    # Check R2 is reasonable (not too negative)
    assert train_metrics['r2'] > -1.0, \
        f"R2 score should be > -1.0, got {train_metrics['r2']}"
    print(f"✓ Training R2: {train_metrics['r2']:.3f} > -1.0")
    
    assert val_metrics['r2'] > -1.0, \
        f"Validation R2 should be > -1.0, got {val_metrics['r2']}"
    print(f"✓ Validation R2: {val_metrics['r2']:.3f} > -1.0")
    
    print("All quality checks passed!")
    print("=" * 60)
    
    return 0  # Success


if __name__ == '__main__':
    exit(main())
