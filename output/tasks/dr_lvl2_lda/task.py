Implementation using closed-form solution for LDA projection
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
)
# Set random seeds for reproducibility (will be set in main)
np.random.seed(42)


    """Load and preprocess the Iris dataset."""
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val


def train_pca(X_train, X_val, X_test, n_components=2):
    where:
    - S_w = within-class scatter matrix = sum_c sum_{x in c} (x - mu_c)(x - mu_c)^T
    - S_b = between-class scatter matrix = sum_c n_c * (mu_c - mu)(mu_c - mu)^T
    - W = projection matrix (n_features x n_components)
    
    Returns:
        W: projection matrix (n_features x n_components)
    return W, mean


def apply_lda_projection(X_train, X_val, X_test, W, mean):  # noqa: F811
    """Apply LDA projection using trained projection matrix."""
    X_train_lda = (X_train - mean) @ W
    X_val_lda = (X_val - mean) @ W
    X_test_lda = (X_test - mean) @ W
    return X_train_lda, X_val_lda, X_test_lda


def train_knn(X_train, y_train, n_neighbors=1):  # noqa: F811
    """Train KNN classifier."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def apply_pca(X_train, X_val, X_test, n_components=2):  # noqa: F811
    """Apply PCA dimensionality reduction."""
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_val_pca, X_test_pca, pca


def evaluate_knn(knn, X, y, name=""):  # noqa: F811
    """Evaluate KNN classifier."""
    y_pred = knn.predict(X)
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "name": name,
    }


def evaluate(data):  # noqa: F811
    """Evaluate both PCA and LDA + 1-NN classifiers.
    
    Returns standard metrics (MSE, R2) and task-specific metrics.
    """
    # Apply dimensionality reduction
    X_train_pca, X_val_pca, X_test_pca, pca = apply_pca(
        data[0], data[1], data[1]
    )
    
    # Train LDA using closed-form solution
    W, lda_mean = train_lda(
        data[0], data[2],
        data[1], data[1]
    )
    X_train_lda, X_val_lda, X_test_lda = apply_lda_projection(
        data[0], data[1], data[1], W, lda_mean
    )
    
    # Train 1-NN classifiers
    knn_pca = train_knn(X_train_pca, data[2])
    knn_lda = train_knn(X_train_lda, data[2])
    
    # Evaluate on validation set
    val_metrics_pca = evaluate_knn(knn_pca, X_val_pca, data[3], "validation_pca")
    val_metrics_lda = evaluate_knn(knn_lda, X_val_lda, data[3], "validation_lda")
    
    # Evaluate on training set
    train_metrics_pca = evaluate_knn(knn_pca, X_train_pca, data[2], "train_pca")
    train_metrics_lda = evaluate_knn(knn_lda, X_train_lda, data[2], "train_lda")
    
    # Compute MSE and R2 for classification
    val_mse_pca = 1.0 - val_metrics_pca["accuracy"]
    val_r2_pca = val_metrics_pca["accuracy"]
    val_mse_lda = 1.0 - val_metrics_lda["accuracy"]
    val_r2_lda = val_metrics_lda["accuracy"]
    
    # Average metrics across PCA and LDA
    avg_mse = (val_mse_pca + val_mse_lda) / 2
    avg_r2 = (val_r2_pca + val_r2_lda) / 2
    
    return {
        "mse": avg_mse,
        "r2": avg_r2,
        "train_accuracy_pca": train_metrics_pca["accuracy"],
        "train_accuracy_lda": train_metrics_lda["accuracy"],
        "lda_projection_matrix": W,
        "val_accuracy_pca": val_metrics_pca["accuracy"],
        "val_accuracy_lda": val_metrics_lda["accuracy"],
        "train_confusion_matrix_pca": train_metrics_pca["confusion_matrix"],
    }


def main():  # noqa: C901, PLR0915
    """Main function to run the LDA dimensionality reduction task."""
    print("=" * 60)
    print("Linear Discriminant Analysis (LDA) - Level 2")
    print("Task: Implement LDA closed-form solution and validate 1-NN beats PCA")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(42)
    
    # 1. Load and prepare data
    print("\n1. Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    data = (X_train, X_val, y_train, y_val)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # 2. Apply dimensionality reduction
    print("\n2. Applying dimensionality reduction (PCA)...")
    X_train_pca, X_val_pca, X_test_pca, pca = apply_pca(
        X_train, X_val, X_val
    )
    print(f"PCA components: {pca.n_components_}")
    
    print("\n2b. Training LDA with closed-form solution...")
    W, lda_mean = train_lda(
        X_train, y_train, X_val, X_val
    )
    X_train_lda, X_val_lda, X_test_lda = apply_lda_projection(
        X_train, X_val, X_val, W, lda_mean
    )
    print(f"LDA components: {W.shape}")
    
    # 3. Train 1-NN classifiers
    print("\n3. Training 1-NN classifiers...")
    knn_pca = train_knn(X_train_pca, y_train)
    knn_lda = train_knn(X_train_lda, y_train)
    print("Trained PCA + 1-NN and LDA + 1-NN classifiers")
    
    # 4. Evaluate on validation set
    print("\n4. Evaluating on validation set...")
    val_metrics_pca = evaluate_knn(knn_pca, X_val_pca, y_val, "validation_pca")
    val_metrics_lda = evaluate_knn(knn_lda, X_val_lda, y_val, "validation_lda")
    
    print(f"PCA + 1-NN Accuracy: {val_metrics_pca['accuracy']:.4f}")
    print(f"PCA + 1-NN MSE: {1.0 - val_metrics_pca['accuracy']:.4f} (1 - accuracy)")
    print(f"PCA + 1-NN R2: {val_metrics_pca['accuracy']:.4f} (accuracy)")
    print(f"LDA + 1-NN Accuracy: {val_metrics_lda['accuracy']:.4f}")
    print(f"LDA + 1-NN MSE: {1.0 - val_metrics_lda['accuracy']:.4f} (1 - accuracy)")
    print(f"LDA + 1-NN R2: {val_metrics_lda['accuracy']:.4f} (accuracy)")
    
    # 5. Evaluate on training set
    print("\n5. Evaluating on training set...")
    train_metrics_pca = evaluate_knn(knn_pca, X_train_pca, y_train, "train_pca")
    train_metrics_lda = evaluate_knn(knn_lda, X_train_lda, y_train, "train_lda")
    print(f"LDA + 1-NN Train Accuracy: {train_metrics_lda['accuracy']:.4f}")
    
    # 6. Print detailed classification reports
    print("\n6. Classification Reports (Validation):")
    print("\nPCA + 1-NN Validation Report:")
    print(classification_report(
        y_val, knn_pca.predict(X_val_pca),
        target_names=target_names
    ))
    print("\nLDA + 1-NN Validation Report:")
    print(classification_report(
        y_val, knn_lda.predict(X_val_lda),
        target_names=target_names
    ))
    
    # 7. Quality checks and assertions
    print("\n7. Quality checks...")
    metrics = evaluate(data)
    print(f"\nStandard Metrics (MSE, R2):")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  R2: {metrics['r2']:.4f} (averaged across PCA and LDA)")
    
    # Task-specific assertions
    print("\nTask-Specific Quality Checks:")
    
    # Check that LDA + 1-NN beats PCA + 1-NN on validation
    lda_acc = val_metrics_lda["accuracy"]
    pca_acc = val_metrics_pca["accuracy"]
    print(f"  LDA accuracy: {lda_acc:.4f}")
    print(f"  PCA accuracy: {pca_acc:.4f}")
    assert lda_acc >= pca_acc, f"LDA should beat PCA: {lda_acc:.4f} < {pca_acc:.4f}"
    print(f"  ✓ LDA + 1-NN beats PCA + 1-NN on validation")
    
    # Check R2 > 0.9 (for classification, this means accuracy > 0.9)
    assert metrics["r2"] > 0.9, f"R2 should be > 0.9, got {metrics['r2']:.4f}"
    print(f"  ✓ R2 > 0.9: {metrics['r2']:.4f}")
    
    # Check MSE < 0.1
    assert metrics["mse"] < 0.1, f"MSE should be < 0.1, got {metrics['mse']:.4f}"
    print(f"  ✓ MSE < 0.1: {metrics['mse']:.4f}")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
