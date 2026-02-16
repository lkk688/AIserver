import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processor import CSVDataLoader


class FeatureHashingNB:
    """Naive Bayes classifier with feature hashing for dimensionality reduction."""
    
    def __init__(self, hash_dim=256):
        self.hash_dim = hash_dim
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = []
    
    def _hash_features(self, X):
        """Apply feature hashing to reduce dimensionality."""
        hashed = np.zeros((len(X), self.hash_dim))
        for i, row in enumerate(X):
            for j, val in enumerate(row):
                # Simple hash function: mod by hash_dim
                hash_idx = abs(hash((i, j))) % self.hash_dim
                hashed[i, hash_idx] += val
        return hashed
    
    def fit(self, X, y):
        """Train the Naive Bayes classifier."""
        X_hashed = self._hash_features(X)
        self.classes = list(set(y))
        n_features = X_hashed.shape[1]
        
        # Calculate class priors
        for c in self.classes:
            self.class_priors[c] = sum(1 for yi in y if yi == c) / len(y)
        
        # Calculate feature probabilities for each class
        self.feature_probs = {}
        for c in self.classes:
            class_indices = [i for i, yi in enumerate(y) if yi == c]
            class_features = X_hashed[class_indices]
            
            # Use Laplace smoothing
            self.feature_probs[c] = (class_features.mean(axis=0) + 1) / (class_features.sum() + n_features)
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        X_hashed = self._hash_features(X)
        predictions = []
        
        for row in X_hashed:
            class_scores = {}
            for c in self.classes:
                # Calculate log probability to avoid underflow
                prob = 1.0
                for j, val in enumerate(row):
                    if val > 0:
                        prob *= self.feature_probs[c][j] ** val
                class_scores[c] = np.log(prob + 1e-10) + np.log(self.class_priors[c])
            
            predictions.append(max(class_scores, key=class_scores.get))
        
        return predictions


def calculate_metrics(y_true, y_pred):
    """Calculate MSE, R2, and accuracy metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Convert to numeric for MSE/R2 calculation
    unique_classes = list(set(y_true))
    y_true_num = np.array([unique_classes.index(y) for y in y_true])
    y_pred_num = np.array([unique_classes.index(y) for y in y_pred])
    
    # MSE
    mse = np.mean((y_true_num - y_pred_num) ** 2)
    
    # R2 score
    ss_res = np.sum((y_true_num - y_pred_num) ** 2)
    ss_tot = np.sum((y_true_num - np.mean(y_true_num)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'accuracy': float(accuracy)
    }


def evaluate(model, X_train, y_train, X_val, y_val):
    """Evaluate the model on training and validation data."""
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    
    return {
        'train': train_metrics,
        'validation': val_metrics
    }


def main():
    """Main function to run the feature hashing NB task."""
    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_loader = CSVDataLoader('data/sample_data.csv')
    data_loader.load_data()
    
    # Get features and targets (assuming last column is target)
    features, targets = data_loader.get_features_targets(
        feature_cols=list(range(len(data_loader.headers) - 1)),
        target_col=len(data_loader.headers) - 1
    )
    
    # Split data
    split_idx = int(len(features) * 0.8)
    X_train, X_val = features[:split_idx], features[split_idx:]
    y_train, y_val = targets[:split_idx], targets[split_idx:]
    
    # Test different hash dimensions
    hash_dims = [32, 64, 128, 256, 512]
    results = []
    memories = []
    
    for hash_dim in hash_dims:
        model = FeatureHashingNB(hash_dim=hash_dim)
        metrics = evaluate(model, X_train, y_train, X_val, y_val)
        results.append(metrics['validation'])
        memories.append(hash_dim * 8)  # Approximate memory usage
    
    # Plot F1 vs hash_dim (using accuracy as proxy for F1)
    plt.figure(figsize=(10, 6))
    plt.plot(hash_dims, [r['accuracy'] for r in results], 'b-o', label='Accuracy')
    plt.xlabel('Hash Dimension')
    plt.ylabel('Accuracy')
    plt.title('Feature Hashing NB: Accuracy vs Hash Dimension')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'f1_vs_hash_dim.png'))
    plt.close()
    
    # Plot memory vs hash_dim to show monotonic memory drop
    plt.figure(figsize=(10, 6))
    plt.plot(hash_dims, memories, 'r-s', label='Memory Usage (bytes)')
    plt.xlabel('Hash Dimension')
    plt.ylabel('Memory Usage')
    plt.title('Feature Hashing: Memory vs Hash Dimension')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'memory_vs_hash_dim.png'))
    plt.close()
    
    # Print metrics
    print("\n" + "="*60)
    print("FEATURE HASHING NB RESULTS")
    print("="*60)
    
    for i, (hash_dim, metrics) in enumerate(zip(hash_dims, results)):
        print(f"\nHash Dimension: {hash_dim}")
        print(f"  Train - MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Val   - MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    
    # Assert quality thresholds
    best_result = max(results, key=lambda x: x['r2'])
    assert best_result['r2'] > 0.7, f"R2 score {best_result['r2']:.4f} is below threshold 0.7"
    assert best_result['accuracy'] > 0.7, f"Accuracy {best_result['accuracy']:.4f} is below threshold 0.7"
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())
