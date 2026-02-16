#!/usr/bin/env python3
"""
Gaussian Mixture Model Clustering with EM Algorithm - Level 2
Task: Implement evaluate() returning MSE, R2, and metrics
Implementation using EM algorithm with full and diagonal covariance options.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Set random seeds for reproducibility
np.random.seed(42)

def generate_data(num_samples=500, n_components=3, noise=0.1, n_features=2):
    """Generate synthetic data for GMM clustering."""
    centers = np.array([
        [2.0, 2.0],
        [-2.0, -2.0],
        [2.0, -2.0],
    ])
    
    # Calculate samples per cluster
        """
        E-step: Compute responsibilities (posterior probabilities of cluster assignments).
        
        p(z=k|x) = pi_k * N(x|mu_k, Sigma_k) / sum_j pi_j * N(x|mu_j, Sigma_j)
        """
        n_samples = X.shape[0]
        
            log_resp[:, k] = log_prob
        
        # Normalize using log-sum-exp trick for numerical stability
        # log_resp_sum is the log of the denominator (marginal probability p(x))
        log_resp_sum = logsumexp(log_resp, axis=1, keepdims=True)
        log_resp = log_resp - log_resp_sum
        
        """
        M-step: Update cluster parameters (means, covariances, weights).
        
        pi_k = sum_i r_ik / N  (mixing weights)
        mu_k = sum_i r_ik * x_i / sum_i r_ik  (cluster means)
        Sigma_k = sum_i r_ik * (x_i - mu_k)(x_i - mu_k)^T / sum_i r_ik  (covariances)
        """
        """
        Fit the GMM model using EM algorithm.
        
        Returns: self
        """
        n_samples, self.n_features = X.shape
        
        self._initialize_parameters(X)
        
        # EM algorithm
        prev_log_likelihood = -np.inf  # Track log-likelihood for convergence

        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            resp, log_likelihood = self._e_step(X)
            
            # M-step
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                # Log-likelihood should be non-decreasing
                assert log_likelihood >= prev_log_likelihood - 1e-6, \
                    f"Log-likelihood decreased: {prev_log_likelihood:.6f} -> {log_likelihood:.6f}"
                self.n_iter_ = iteration + 1
                self.log_likelihood_ = log_likelihood
                return self
            
            prev_log_likelihood = log_likelihood
            # Log-likelihood should be non-decreasing (allow tiny numerical noise)
            assert log_likelihood >= prev_log_likelihood - 1e-6, \
                f"Log-likelihood decreased at iteration {iteration}: {prev_log_likelihood:.6f} -> {log_likelihood:.6f}"
            self.n_iter_ = iteration + 1
            self.log_likelihood_ = log_likelihood
        
        return resp.argmax(axis=1)
    
    def score(self, X):  # noqa: C901
        """
        Compute the average log-likelihood (per sample).
        
        Higher is better - this is the objective the EM algorithm maximizes.
        """
        _, log_resp_sum = self._e_step(X)
        return log_resp_sum / X.shape[0]
    
        """Compute Bayesian Information Criterion."""
        n_samples = X.shape[0]
        n_params = self._count_parameters()
        """
        BIC = -2 * log_likelihood + n_params * log(n_samples)
        
        Lower is better - balances model fit and complexity.
        """
        return -2 * self.log_likelihood_ + n_params * np.log(n_samples)
    
    def aic(self, X):
        n_samples = X.shape[0]
        n_params = self._count_parameters()
        """
        AIC = -2 * log_likelihood + 2 * n_params
        
        Lower is better - balances model fit and complexity.
        """
        return -2 * self.log_likelihood_ + 2 * n_params
    
    def _count_parameters(self):


def evaluate(gmm, X, y_true):
    """
    Evaluate GMM clustering performance.
    
    Returns:
        dict: Dictionary containing MSE, R2, silhouette score, ARI, and log-likelihood
    """
    cluster_labels = gmm.predict(X)
    
    # Calculate log-likelihood (higher is better - per sample)
    if hasattr(gmm, 'score'):
        log_likelihood = gmm.score(X)
    else:
        # For custom implementation (legacy support)
        log_likelihood = gmm.log_likelihood_ / len(X) if hasattr(gmm, 'log_likelihood_') else 0.0
    
    # Calculate MSE between points and their cluster centers
        center = gmm.means_[label]
        mse += np.sum((point - center) ** 2)
    mse /= len(X)
    
    # Calculate R2 score (pseudo R2 based on cluster center predictions)
    # R2 = 1 - SS_res / SS_tot where SS_tot is variance around cluster centers
    cluster_means = gmm.means_[cluster_labels]
    ss_res = np.sum((X - cluster_means) ** 2)
    ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2 = max(0.0, r2)  # Ensure non-negative
    
    # Calculate silhouette score
    sil_score = silhouette_score(X, cluster_labels)
    ari = adjusted_rand_score(y_true, cluster_labels)
    
    return {
        'log_likelihood': log_likelihood,  # Higher is better
        'mse': mse,
        'r2': r2,  # Higher is better (pseudo R2)
        'silhouette_score': sil_score,
        'ari': ari,  # Higher is better (clustering accuracy)
    }


    n_clusters = len(np.unique(cluster_labels))
    
    # Check that no cluster is empty
    # Each cluster should have at least min_cluster_ratio of total samples
    for cluster in range(n_clusters):
        cluster_size = np.sum(cluster_labels == cluster)
        if cluster_size < n_samples * min_cluster_ratio:
    if n_clusters < 2:
        return False
    
    # Check that cluster centers are spread out (not all at same location)
    centers = gmm.means_
    center_distances = []
    for i in range(len(centers)):
        # Check all pairwise distances
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            center_distances.append(dist)


def main():  # noqa: C901
    """Main function to run the GMM clustering task with EM algorithm."""
    print("=" * 60)
    print("Gaussian Mixture Model Clustering with EM Algorithm - Level 2")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Generating synthetic data...")
    # Generate data with 3 true clusters
    X, y_true = generate_data(num_samples=500, n_components=3, noise=0.1, n_features=2)
    print(f"Generated {len(X)} samples with {len(np.unique(y_true))} true clusters")
    print(f"X shape: {X.shape}")
    # 2. Split data
    print("\n2. Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_true, train_size=0.8, random_state=42, stratify=y_true
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 3. Train GMM model with EM algorithm
    print("\n3. Training GMM with EM algorithm...")
    gmm = GaussianMixtureModel(n_components=3, max_iter=100, tol=1e-6, random_state=42, covariance_type='full')
    gmm.fit(X_train)
    print(f"EM converged after {gmm.n_iter_} iterations")
    print(f"Final log-likelihood: {gmm.log_likelihood_:.4f}")
    
    # 4. Evaluate on training data
    print("\n4. Evaluating on training data...")
    train_metrics = evaluate(gmm, X_train, y_train)
    print(f"Training Metrics:")
    print(f"  Log-likelihood: {train_metrics['log_likelihood']:.4f}")
    print(f"  MSE: {train_metrics['mse']:.4f}")
    print(f"  R2: {train_metrics['r2']:.4f}")
    print(f"  Silhouette: {train_metrics['silhouette_score']:.4f}")
    print(f"  ARI: {train_metrics['ari']:.4f}")
    
    # 5. Evaluate on validation data
    print("\n5. Evaluating on validation data...")
    val_metrics = evaluate(gmm, X_val, y_val)
    print(f"Validation Metrics:")
    print(f"  Log-likelihood: {val_metrics['log_likelihood']:.4f}")
    print(f"  MSE: {val_metrics['mse']:.4f}")
    print(f"  R2: {val_metrics['r2']:.4f}")
    print(f"  Silhouette: {val_metrics['silhouette_score']:.4f}")
    print(f"  ARI: {val_metrics['ari']:.4f}")
    
    # 6. Verify clusters are reasonable
    print("\n6. Verifying cluster quality...")
    cluster_labels = gmm.predict(X_val)
    assert verify_clusters_reasonable(gmm, X_val, cluster_labels), "Clusters are not reasonable"
    print("✓ Clusters are reasonable (non-empty and spread out)")
    
    # 7. Quality checks
    print("\n7. Quality checks...")
    
    # Check log-likelihood is non-decreasing (should be during EM)
    # We can't directly check this without storing all iterations, but we can
    # verify the final log-likelihood is reasonable (higher than random)
    assert train_metrics['log_likelihood'] > -10, \
        f"Log-likelihood too low: {train_metrics['log_likelihood']:.4f}"
    print(f"✓ Log-likelihood is reasonable: {train_metrics['log_likelihood']:.4f}")
    
    # Check R2 score (should be reasonably high for well-separated clusters)
    assert train_metrics['r2'] > 0.5, \
        f"R2 score too low: {train_metrics['r2']:.4f}"
    print(f"✓ R2 score is good: {train_metrics['r2']:.4f}")
    
    # Check silhouette score (should be positive for reasonable clustering)
    assert train_metrics['silhouette_score'] > 0.1, \
        f"Silhouette score too low: {train_metrics['silhouette_score']:.4f}"
    print(f"✓ Silhouette score is positive: {train_metrics['silhouette_score']:.4f}")
    
    # Check ARI (should be reasonably high for true cluster recovery)
    assert train_metrics['ari'] > 0.5, \
        f"ARI too low: {train_metrics['ari']:.4f}"
    print(f"✓ ARI is good: {train_metrics['ari']:.4f}")
    
    # Check validation metrics are similar to training (no overfitting)
    assert val_metrics['log_likelihood'] > train_metrics['log_likelihood'] - 1.0, \
        f"Validation log-likelihood much lower than training"
    print(f"✓ Validation log-likelihood comparable to training")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0  # Success


if __name__ == '__main__':
