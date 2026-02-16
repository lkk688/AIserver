#!/usr/bin/env python3
"""
t-SNE Dimensionality Reduction - Level 3 (Simplified)
Task: Implement t-SNE algorithm for dimensionality reduction with visualization
and validation that KL divergence decreases during optimization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)


def generate_data(num_samples=500, num_features=50, n_classes=5):
    """Generate synthetic high-dimensional data with class structure."""
    samples_per_class = num_samples // n_classes
    X_list = []
    y_list = []
    
    for class_idx in range(n_classes):
        # Create class-specific center in high-dimensional space
        class_center = np.random.randn(num_features) * 5 + class_idx * 3
        
        # Generate samples around the class center
        class_samples = class_center + np.random.randn(samples_per_class, num_features) * 0.5
        X_list.append(class_samples)
        y_list.append(np.full(samples_per_class, class_idx))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y


def euclidean_distance(X):
    """Compute pairwise Euclidean distances."""
    n = X.shape[0]
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * np.dot(X, X.T)
    return np.maximum(D, 0)  # Ensure non-negative due to numerical errors


def compute_perplexity(distance_matrix, beta, epsilon=1e-10):
    """Compute perplexity for a given beta value."""
    P = np.exp(-distance_matrix * beta)
    np.fill_diagonal(P, 0)
    P = P / (np.sum(P, axis=1)[:, np.newaxis] + epsilon)
    return P


def binary_search_beta(distance_row, target_perplexity, epsilon=1e-10):
    """Find optimal beta using binary search for given perplexity."""
    min_beta = 1e-10
    max_beta = 1e10
    
    for _ in range(50):  # Binary search iterations
        beta = (min_beta + max_beta) / 2
        P = compute_perplexity(distance_row[np.newaxis, :], beta, epsilon)
        entropy = -np.sum(P * np.log(P + epsilon))
        perplexity = np.exp(entropy)
        
        if np.abs(perplexity - target_perplexity) < 1e-5:
            break
        
        if perplexity < target_perplexity:
            min_beta = beta
        else:
            max_beta = beta
    
    return beta


def compute_P(X, perplexity=30):
    """Compute joint probability matrix P."""
    n = X.shape[0]
    D = euclidean_distance(X)
    P = np.zeros((n, n))
    
    for i in range(n):
        distance_row = D[i]
        beta = binary_search_beta(distance_row, perplexity)
        P[i] = compute_perplexity(distance_row[np.newaxis, :], beta)
    
    # Make symmetric: P = (P + P.T) / (2n)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)  # Avoid log(0)
    
    return P


def compute_Q(Y):
    """Compute joint probability matrix Q from low-dimensional embeddings."""
    n = Y.shape[0]
    D = euclidean_distance(Y)
    # t-SNE uses Student's t-distribution with 1 degree of freedom (Cauchy)
    Q = 1 / (1 + D)
    np.fill_diagonal(Q, 0)
    Q = Q / np.sum(Q)
    Q = np.maximum(Q, 1e-12)
    return Q


def compute_gradient(Y, P, Q):
    """Compute gradient of KL divergence."""
    n = Y.shape[0]
    # Compute squared distances in embedding space
    D = euclidean_distance(Y)
    
    # Compute the difference matrix
    diff = P - Q
    
    # Compute gradient
    grad = np.zeros_like(Y)
    
    for i in range(n):
        # Weighted difference
        diff_row = diff[i] * (1 + D[i]) ** -1
        
        # Gradient computation
        for j in range(n):
            if i != j:
                diff_vec = Y[i] - Y[j]
                grad[i] += 4 * diff_row[j] * diff_vec
    
    return grad


def tsne(X, n_components=2, perplexity=30, learning_rate=200, n_iter=1000, verbose=False):
    """
    t-SNE algorithm for dimensionality reduction.
    
    Args:
        X: High-dimensional data (n_samples, n_features)
        n_components: Number of dimensions in embedding
        perplexity: Effective number of neighbors
        learning_rate: Learning rate for optimization
        n_iter: Number of iterations
        verbose: Print progress
    
    Returns:
        Y: Low-dimensional embedding (n_samples, n_components)
        kl_divergences: List of KL divergence values during optimization
    """
    n_samples, n_features = X.shape
    
    # Initialize embeddings randomly
    Y = np.random.randn(n_samples, n_components) * 0.0001
    
    # Compute high-dimensional probabilities
    P = compute_P(X, perplexity)
    
    # Optimization parameters
    momentum = 0.5
    final_momentum = 0.8
    switch_iter = 250
    gain = np.ones_like(Y)
    
    kl_divergences = []
    
    for iter_num in range(n_iter):
        # Compute low-dimensional probabilities
        Q = compute_Q(Y)
        
        # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
        kl_div = np.sum(P * np.log(P / Q))
        kl_divergences.append(kl_div)
        
        if verbose and iter_num % 100 == 0:
            print(f"Iteration {iter_num}: KL divergence = {kl_div:.6f}")
        
        # Compute gradient
        grad = compute_gradient(Y, P, Q)
        
        # Update with momentum
        if iter_num < switch_iter:
            current_momentum = momentum
        else:
            current_momentum = final_momentum
        
        # Adaptive learning rate (gain)
        grad_sign = np.sign(grad)
        gain = gain * (1.0 + 0.2 * grad_sign) * (grad > 0) + gain * (1.0 - 0.2 * grad_sign) * (grad < 0)
        gain = np.maximum(gain, 0.01)
        
        # Update embeddings
        Y = Y - learning_rate * gain * grad
        
        # Add momentum term
        if iter_num > 0:
            Y = Y + current_momentum * (Y - Y_prev)
        
        Y_prev = Y.copy()
    
    return Y, kl_divergences


def evaluate(model, X, y, Y_embedded=None):
    """
    Evaluate the t-SNE embedding quality.
    
    Returns metrics including MSE, R2, and t-SNE specific metrics.
    """
    metrics = {}
    
    # If we have the original data, compute reconstruction MSE (approximate)
    if X is not None and Y_embedded is not None:
        # Compute pairwise distances in original and embedded space
        D_orig = euclidean_distance(X)
        D_embed = euclidean_distance(Y_embedded)
        
        # Normalize distances
        D_orig_norm = D_orig / (np.max(D_orig) + 1e-10)
        D_embed_norm = D_embed / (np.max(D_embed) + 1e-10)
        
        # MSE between distance matrices
        metrics['mse'] = np.mean((D_orig_norm - D_embed_norm) ** 2)
        
        # R2 score for distance preservation
        ss_res = np.sum((D_orig_norm - D_embed_norm) ** 2)
        ss_tot = np.sum((D_orig_norm - np.mean(D_orig_norm)) ** 2)
        metrics['r2_score'] = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Class separation in embedded space
        unique_classes = np.unique(y)
        class_centers = np.array([np.mean(Y_embedded[y == c], axis=0) for c in unique_classes])
        
        # Between-class scatter
        overall_center = np.mean(class_centers, axis=0)
        between_class_scatter = np.sum([len(y[y == c]) * np.outer(class_centers[i] - overall_center, 
                                                                   class_centers[i] - overall_center) 
                                        for i, c in enumerate(unique_classes)])
        
        # Within-class scatter
        within_class_scatter = np.zeros((Y_embedded.shape[1], Y_embedded.shape[1]))
        for c in unique_classes:
            class_samples = Y_embedded[y == c]
            class_center = np.mean(class_samples, axis=0)
            within_class_scatter += np.cov(class_samples.T) * len(class_samples)
        
        # Separation ratio (trace(between) / trace(within))
        sep_ratio = np.trace(between_class_scatter) / (np.trace(within_class_scatter) + 1e-10)
        metrics['separation_ratio'] = sep_ratio
    
    return metrics


def visualize_results(Y, y, kl_divergences, save_dir='.'):
    """Save 2D embedding plot and KL divergence plot."""
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: t-SNE embedding
    plt.figure(figsize=(10, 8))
    unique_classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    
    for i, c in enumerate(unique_classes):
        mask = y == c
        plt.scatter(Y[mask, 0], Y[mask, 1], c=colors[i], label=f'Class {int(c)}', alpha=0.7, s=50)
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE 2D Embedding')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'tsne_embedding.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: KL divergence during optimization
    plt.figure(figsize=(10, 6))
    plt.plot(kl_divergences, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence During t-SNE Optimization')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'tsne_kl_divergence.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():  # noqa: C901
    """Main function to run the t-SNE task."""
    print("=" * 60)
    print("t-SNE Dimensionality Reduction - Level 3 (Simplified)")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Generating synthetic high-dimensional data...")
    X, y = generate_data(num_samples=500, num_features=50, n_classes=5)
    print(f"Generated data: X shape = {X.shape}, y shape = {y.shape}")
    
    # 2. Split data
    print("\n2. Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 3. Run t-SNE on training data
    print("\n3. Running t-SNE on training data...")
    print("   Parameters: perplexity=30, learning_rate=200, iterations=500")
    Y_train, kl_divergences = tsne(
        X_train, n_components=2, perplexity=30, 
        learning_rate=200, n_iter=500, verbose=True
    )
    
    # 4. Run t-SNE on validation data
    print("\n4. Running t-SNE on validation data...")
    Y_val, val_kl_divergences = tsne(
        X_val, n_components=2, perplexity=30,
        learning_rate=200, n_iter=500, verbose=False
    )
    
    # 5. Evaluate on both splits
    print("\n5. Evaluating embeddings...")
    train_metrics = evaluate(None, X_train, y_train, Y_train)
    val_metrics = evaluate(None, X_val, y_val, Y_val)
    
    print(f"\nTraining Metrics:")
    print(f"  MSE (distance preservation): {train_metrics['mse']:.6f}")
    print(f"  R2 Score (distance preservation): {train_metrics['r2_score']:.6f}")
    print(f"  Separation Ratio: {train_metrics['separation_ratio']:.6f}")
    
    print(f"\nValidation Metrics:")
    print(f"  MSE (distance preservation): {val_metrics['mse']:.6f}")
    print(f"  R2 Score (distance preservation): {val_metrics['r2_score']:.6f}")
    print(f"  Separation Ratio: {val_metrics['separation_ratio']:.6f}")
    
    # 6. Generate visualizations
    print("\n6. Generating visualizations...")
    visualize_results(Y_train, y_train, kl_divergences, save_dir='.')
    print("Saved: tsne_embedding.png, tsne_kl_divergence.png")
    
    # 7. Quality checks
    print("\n7. Quality checks...")
    
    # Check KL divergence decreases
    assert kl_divergences[0] > kl_divergences[-1], \
        f"KL divergence should decrease: {kl_divergences[0]:.6f} -> {kl_divergences[-1]:.6f}"
    print(f"✓ KL divergence decreases: {kl_divergences[0]:.6f} -> {kl_divergences[-1]:.6f}")
    
    # Check R2 score is reasonable (should be positive)
    assert train_metrics['r2_score'] > 0.1, \
        f"R2 score should be positive: {train_metrics['r2_score']:.6f}"
    print(f"✓ R2 score is positive: {train_metrics['r2_score']:.6f}")
    
    # Check MSE is reasonable
    assert train_metrics['mse'] < 1.0, \
        f"MSE should be less than 1.0: {train_metrics['mse']:.6f}"
    print(f"✓ MSE is reasonable: {train_metrics['mse']:.6f}")
    
    # Check separation ratio is positive
    assert train_metrics['separation_ratio'] > 0, \
        f"Separation ratio should be positive: {train_metrics['separation_ratio']:.6f}"
    print(f"✓ Separation ratio is positive: {train_metrics['separation_ratio']:.6f}")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
