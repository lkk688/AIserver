#!/usr/bin/env python3
"""
DBSCAN Clustering - Level 3
Task: Precompute pairwise distances for efficiency, find non-convex clusters on moons dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)


class DBSCAN:
    """DBSCAN clustering algorithm with precomputed pairwise distances."""
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize DBSCAN.
        
        Args:
            eps: Maximum distance between two samples for them to be considered neighbors
            min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.pairwise_distances_ = None
    
    def _compute_pairwise_distances(self, X):
        """Precompute pairwise Euclidean distances."""
        n_samples = X.shape[0]
        # Efficient computation using broadcasting
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        sq_norms = np.sum(X ** 2, axis=1)
        distances = sq_norms.reshape(-1, 1) + sq_norms.reshape(1, -1) - 2 * np.dot(X, X.T)
        # Handle numerical errors (small negative values)
        distances = np.maximum(distances, 0)
        distances = np.sqrt(distances)
        return distances
    
    def _get_neighbors(self, point_idx):
        """Get indices of points within eps distance."""
        return np.where(self.pairwise_distances_[point_idx] <= self.eps)[0]
    
    def fit(self, X):
        """
        Perform DBSCAN clustering.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        n_samples = X.shape[0]
        
        # Precompute pairwise distances
        self.pairwise_distances_ = self._compute_pairwise_distances(X)
        
        # Initialize labels: -1 means noise, -2 means unvisited
        self.labels_ = np.full(n_samples, -2, dtype=int)
        
        cluster_id = 0
        
        for point_idx in range(n_samples):
            # Skip if already processed
            if self.labels_[point_idx] != -2:
                continue
            
            # Get neighbors
            neighbors = self._get_neighbors(point_idx)
            
            # Check if core point (enough neighbors)
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1  # Mark as noise
                continue
            
            # Start a new cluster
            self.labels_[point_idx] = cluster_id
            
            # Expand cluster
            seed_set = list(neighbors)
            i = 0
            while i < len(seed_set):
                current_point = seed_set[i]
                
                if self.labels_[current_point] == -1:
                    # Was marked as noise, now it's part of cluster (border point)
                    self.labels_[current_point] = cluster_id
                elif self.labels_[current_point] == -2:
                    # Unvisited point - add to cluster
                    self.labels_[current_point] = cluster_id
                    
                    # Get neighbors of this point
                    current_neighbors = self._get_neighbors(current_point)
                    
                    # If this point is also a core point, add its neighbors to seed set
                    if len(current_neighbors) >= self.min_samples:
                        for neighbor in current_neighbors:
                            if neighbor not in seed_set:
                                seed_set.append(neighbor)
                
                i += 1
           
            cluster_id += 1
       
        return self
   
    def fit_predict(self, X):
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


def generate_moons_data(n_samples=300, noise=0.1):
    """Generate moons dataset."""
    X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y_true


def evaluate(model, X, y_true):
    """
    Evaluate DBSCAN clustering performance.
    
    Since DBSCAN is unsupervised, we use:
    - Silhouette-like metric based on intra-cluster and inter-cluster distances
    - Noise ratio
    - Number of clusters found
    
    Args:
        model: Fitted DBSCAN model
        X: Data
        y_true: True labels (for comparison when available)
        
    Returns:
        dict with metrics
    """
    labels = model.labels_
    n_samples = len(labels)
    
    # Count clusters (excluding noise label -1)
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    
    # Calculate noise ratio
    noise_count = np.sum(labels == -1)
    noise_ratio = noise_count / n_samples if n_samples > 0 else 0
   
    # Calculate compactness (average intra-cluster distance)
    compactness = 0.0
    valid_clusters = 0
    
    for cluster_label in unique_labels:
        if cluster_label == -1:
            continue
       
        cluster_mask = labels == cluster_label
        cluster_points = X[cluster_mask]
        
        if len(cluster_points) < 2:
            continue
       
        # Average distance between points in cluster
        if len(cluster_points) > 1:
            cluster_distances = model.pairwise_distances_[cluster_mask][:, cluster_mask]
            # Get upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(len(cluster_points), k=1)
            avg_dist = np.mean(cluster_distances[triu_indices])
            compactness += avg_dist
            valid_clusters += 1
   
    compactness = compactness / valid_clusters if valid_clusters > 0 else 0
   
    # Separation (average inter-cluster distance for nearest cluster pairs)
    separation = 0.0
    cluster_centers = []
    
    for cluster_label in unique_labels:
        if cluster_label == -1:
            continue
        cluster_mask = labels == cluster_label
        cluster_points = X[cluster_mask]
        if len(cluster_points) > 0:
            cluster_centers.append(np.mean(cluster_points, axis=0))
    
    if len(cluster_centers) >= 2:
        min_separation = float('inf')
        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                min_separation = min(min_separation, dist)
        separation = min_separation
   
    # If true labels available, compute adjusted metrics
    mse = 0.0
    r2 = 0.0
    if y_true is not None and len(np.unique(y_true)) <= 10:  # Only for small number of classes
        # Create predicted labels for comparison
        # Map cluster labels to true labels (simple approach)
        predicted = np.zeros_like(labels, dtype=int)
        for cluster_label in unique_labels:
            if cluster_label == -1:
                predicted[labels == cluster_label] = -1
            else:
                cluster_mask = labels == cluster_label
                # Find most common true label in cluster
                true_labels_in_cluster = y_true[cluster_mask]
                if len(true_labels_in_cluster) > 0:
                    most_common = np.bincount(true_labels_in_cluster).argmax()
                    predicted[cluster_mask] = most_common
       
        # Calculate MSE and R2 against true labels (treating as regression)
        mse = mean_squared_error(y_true, predicted)
        r2 = r2_score(y_true, predicted)
    
    return {
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'compactness': compactness,
        'separation': separation,
        'mse': mse,
        'r2': r2,
        'n_samples': n_samples
    }


def main():  # noqa: C901
    """Main function to run DBSCAN clustering task."""
    print("=" * 60)
    print("DBSCAN Clustering - Level 3")
    print("Task: Precompute pairwise distances, find non-convex clusters")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Generating moons dataset...")
    X, y_true = generate_moons_data(n_samples=300, noise=0.1)
    print(f"X shape: {X.shape}, y_true shape: {y_true.shape}")
    
    # 2. Split data
    print("\n2. Splitting data into train and validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_true, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 3. Train DBSCAN
    print("\n3. Training DBSCAN with precomputed pairwise distances...")
    model = DBSCAN(eps=0.3, min_samples=5)
    model.fit(X_train)
    print(f"Found {len(set(model.labels_) - {-1})} clusters")
    print(f"Noise points: {np.sum(model.labels_ == -1)}")
    
    # 4. Evaluate on training data
    print("\n4. Evaluating on training data...")
    train_metrics = evaluate(model, X_train, y_train)
    print(f"Training Metrics:")
    print(f"  - Number of clusters: {train_metrics['n_clusters']}")
    print(f"  - Noise ratio: {train_metrics['noise_ratio']:.4f}")
    print(f"  - Compactness: {train_metrics['compactness']:.4f}")
    print(f"  - Separation: {train_metrics['separation']:.4f}")
    print(f"  - MSE: {train_metrics['mse']:.4f}")
    print(f"  - R2: {train_metrics['r2']:.4f}")
    
    # 5. Evaluate on validation data
    print("\n5. Evaluating on validation data...")
    val_metrics = evaluate(model, X_val, y_val)
    print(f"Validation Metrics:")
    print(f"  - Number of clusters: {val_metrics['n_clusters']}")
    print(f"  - Noise ratio: {val_metrics['noise_ratio']:.4f}")
    print(f"  - Compactness: {val_metrics['compactness']:.4f}")
    print(f"  - Separation: {val_metrics['separation']:.4f}")
    print(f"  - MSE: {val_metrics['mse']:.4f}")
    print(f"  - R2: {val_metrics['r2']:.4f}")
    
    # 6. Generate visualization
    print("\n6. Generating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training data
    unique_labels = set(model.labels_)
    for label in unique_labels:
        mask = model.labels_ == label
        color = 'red' if label == -1 else plt.cm.tab10(label % 10)
        axes[0].scatter(X_train[mask, 0], X_train[mask, 1], 
                       c=color, s=20, label=f'Cluster {label}' if label != -1 else 'Noise')
    axes[0].set_title('Training Data - DBSCAN Clusters')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].legend()
    
    # Validation data
    for label in unique_labels:
        mask = model.labels_ == label
        color = 'red' if label == -1 else plt.cm.tab10(label % 10)
        axes[1].scatter(X_val[mask, 0], X_val[mask, 1], 
                       c=color, s=20, label=f'Cluster {label}' if label != -1 else 'Noise')
    axes[1].set_title('Validation Data - DBSCAN Clusters')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('dbscan_results.png', dpi=150, bbox_inches='tight')
    print("Saved: dbscan_results.png")
    plt.close()
    
    # 7. Quality checks
    print("\n7. Quality checks...")
    
    # Check that we found meaningful clusters (not all noise)
    assert val_metrics['n_clusters'] >= 2, f"Should find at least 2 clusters, got {val_metrics['n_clusters']}"
    print(f"✓ Found {val_metrics['n_clusters']} clusters (expected >= 2)")
    
    # Check noise ratio is reasonable (< 20%)
    assert val_metrics['noise_ratio'] < 0.2, f"Noise ratio too high: {val_metrics['noise_ratio']:.4f}"
    print(f"✓ Noise ratio acceptable: {val_metrics['noise_ratio']:.4f} (< 0.2)")
    
    # Check compactness is reasonable
    assert val_metrics['compactness'] < 1.0, f"Compactness too high: {val_metrics['compactness']:.4f}"
    print(f"✓ Compactness acceptable: {val_metrics['compactness']:.4f}")
    
    # Check R2 is reasonable (for moons, should be decent)
    assert val_metrics['r2'] > 0.5, f"R2 score too low: {val_metrics['r2']:.4f}"
    print(f"✓ R2 score acceptable: {val_metrics['r2']:.4f} (> 0.5)")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
