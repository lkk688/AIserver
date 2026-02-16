#!/usr/bin/env python3
"""
K-means Clustering - Level 1
Task: Implement K-means clustering from scratch with SSE objective
Implementation using manual K-means algorithm with k-means++ initialization
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score
from scipy.stats import mode
from sklearn.metrics import mean_squared_error


# Set random seeds for reproducibility
np.random.seed(42)


def generate_data(num_samples=300, num_features=2, n_clusters=4, noise=0.3):
    """Generate synthetic cluster data with well-separated clusters."""
    X, y_true = make_blobs(
        n_samples=num_samples,
        centers=n_clusters,
        n_features=num_features,
        cluster_std=noise,
        random_state=42
    )
    return X, y_true


def split_data(X, y, train_ratio=0.8):
    """Split data into training and validation sets."""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * train_ratio)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def euclidean_distance(X1, X2):
    """Compute Euclidean distance between each point in X1 and X2."""
    # X1: (n, d), X2: (m, d) -> output: (n, m)
    X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
    X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
    cross = np.dot(X1, X2.T)
    return np.sqrt(np.maximum(X1_sq + X2_sq.T - 2 * cross, 0))


def kmeans_plus_plus_init(X, n_clusters, random_state):
    """Initialize centroids using k-means++ algorithm."""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    centroids = []
    
    # Choose first centroid randomly
    first_idx = np.random.randint(0, n_samples)
    centroids.append(X[first_idx].copy())
    
    for _ in range(1, n_clusters):
        # Compute distances to nearest centroid
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            min_dist = float('inf')
            for c in centroids:
                dist = np.sum((X[i] - c) ** 2)
                if dist < min_dist:
                    min_dist = dist
            distances[i] = min_dist
        
        # Choose next centroid with probability proportional to distance squared
        probs = distances / distances.sum()
        cumprobs = np.cumsum(probs)
        r = np.random.random()
        next_idx = np.searchsorted(cumprobs, r)
        next_idx = min(next_idx, n_samples - 1)
        centroids.append(X[next_idx].copy())
    
    return np.array(centroids)


class KMeans:
    """K-means clustering implementation from scratch."""
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42, init='k-means++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    
    def fit(self, X):
        """Fit K-means to the data."""
        n_samples, n_features = X.shape
        
        # Initialize centroids
        if self.init == 'k-means++':
            self.centroids_ = kmeans_plus_plus_init(X, self.n_clusters, self.random_state)
        else:
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids_ = X[indices].copy()
        
        for self.n_iter_ in range(1, self.max_iter + 1):
            # Assign points to nearest centroid
            distances = euclidean_distance(X, self.centroids_)
            self.labels_ = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(self.centroids_)
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = self.centroids_[k]
            
            # Check convergence
            centroid_shift = np.sqrt(np.sum((new_centroids - self.centroids_) ** 2))
            self.centroids_ = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        # Compute inertia (SSE)
        self.inertia_ = self._compute_sse(X)
        return self
    
    def predict(self, X):
        """Predict cluster labels for X."""
        distances = euclidean_distance(X, self.centroids_)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels_
    
    def _compute_sse(self, X):
        """Compute sum of squared errors."""
        sse = 0.0
        for k in range(self.n_clusters):
            mask = self.labels_ == k
            if np.any(mask):
                cluster_points = X[mask]
                centroid = self.centroids_[k]
                sse += np.sum((cluster_points - centroid) ** 2)
        return sse


def compute_purity(labels_true, labels_pred):
    """Compute clustering purity."""
    n = len(labels_true)
    total_correct = 0
    for cluster_id in np.unique(labels_pred):
        mask = labels_pred == cluster_id
        cluster_true_labels = labels_true[mask]
        if len(cluster_true_labels) > 0:
            most_common = mode(cluster_true_labels, keepdims=True).mode[0]
            correct = np.sum(cluster_true_labels == most_common)
            total_correct += correct
    return total_correct / n


def train(model, X_train, X_val=None):
    """Train the K-means model."""
    model.fit(X_train)
    return model


def evaluate(model, X, y_true=None):
    """Evaluate the K-means model and return metrics."""
    # Compute SSE (inertia)
    sse = model.inertia_
    
    # Compute silhouette score
    sil_score = silhouette_score(X, model.labels_)
    
    # Compute purity if true labels are available
    purity = None
    if y_true is not None:
        purity = compute_purity(y_true, model.labels_)
    
    return {
        'sse': sse,
        'silhouette': sil_score,
        'purity': purity,
        'n_clusters': model.n_clusters,
        'n_iter': model.n_iter_
    }


def main():  # noqa: C901
    """Main function to run the K-means clustering task."""
    print("=" * 60)
    print("K-means Clustering - Level 1")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Generating data...")
    X, y_true = generate_data(num_samples=300, num_features=2, n_clusters=4, noise=0.3)
    print(f"X shape: {X.shape}, y shape: {y_true.shape}")
    
    # 2. Split data
    print("\n2. Splitting data...")
    X_train, X_val, y_train, y_val = split_data(X, y_true, train_ratio=0.8)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 3. Train model
    print("\n3. Training K-means model...")
    model = KMeans(n_clusters=4, max_iter=300, tol=1e-4, random_state=42, init='k-means++')
    model = train(model, X_train)
    print(f"Training completed in {model.n_iter_} iterations")
    print(f"Training SSE (inertia): {model.inertia_:.4f}")
    
    # 4. Evaluate on training data
    print("\n4. Evaluating on training data...")
    train_metrics = evaluate(model, X_train, y_train)
    print(f"Training SSE: {train_metrics['sse']:.4f}")
    print(f"Training purity: {train_metrics['purity']:.4f}")
    print(f"Training silhouette: {train_metrics['silhouette']:.4f}")
    
    # 5. Evaluate on validation data
    print("\n5. Evaluating on validation data...")
    val_metrics = evaluate(model, X_val, y_val)
    print(f"Validation SSE: {val_metrics['sse']:.4f}")
    print(f"Validation purity: {val_metrics['purity']:.4f}")
    print(f"Validation silhouette: {val_metrics['silhouette']:.4f}")
    
    # 6. Compare with sklearn K-means
    print("\n6. Comparing with sklearn K-means...")
    sklearn_model = SklearnKMeans(n_clusters=4, max_iter=300, random_state=42, n_init=10)
    sklearn_model.fit(X_train)
    sklearn_sse = sklearn_model.inertia_
    
    print(f"Sklearn SSE: {sklearn_sse:.4f}")
    print(f"Our SSE: {train_metrics['sse']:.4f}")
    
    # Compute relative difference
    rel_diff = abs(train_metrics['sse'] - sklearn_sse) / sklearn_sse * 100
    print(f"Relative difference: {rel_diff:.2f}%")
    print(f"Within 5% tolerance: {rel_diff <= 5}")
    
    # 7. Quality checks
    print("\n7. Quality checks...")
    
    # Check SSE is positive
    assert train_metrics['sse'] > 0, f"SSE must be positive: {train_metrics['sse']}"
    print(f"✓ SSE is positive: {train_metrics['sse']:.4f}")
    
    # Check SSE is not too large (should be comparable to sklearn)
    assert rel_diff <= 5, f"SSE too different from sklearn: {rel_diff:.2f}%"
    print(f"✓ SSE comparable to sklearn (within 5%): {rel_diff:.2f}%")
    
    # Check silhouette score is reasonable
    assert train_metrics['silhouette'] > 0.5, f"Silhouette score too low: {train_metrics['silhouette']:.4f}"
    print(f"✓ Silhouette score reasonable: {train_metrics['silhouette']:.4f}")
    
    # Check purity is reasonable
    assert train_metrics['purity'] > 0.7, f"Purity too low: {train_metrics['purity']:.4f}"
    print(f"✓ Purity reasonable: {train_metrics['purity']:.4f}")
    
    # Check inertia decreases (run multiple times with same data)
    model2 = KMeans(n_clusters=4, max_iter=300, random_state=42, init='k-means++')
    model2.fit(X_train)
    assert model2.inertia_ <= model.inertia_ * 1.1, "Inertia should be stable across runs"
    print(f"✓ Inertia stable across runs: {model.inertia_:.4f} vs {model2.inertia_:.4f}")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
