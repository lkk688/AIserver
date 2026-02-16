"""Decision Tree Classifier using Gini Impurity on Iris Dataset."""

import numpy as np
import pickle
from collections import Counter
from typing import Tuple, List, Dict, Any
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


class DecisionTreeClassifierScratch:
    """Custom Decision Tree Classifier using Gini Impurity."""
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity for a set of labels."""
        if len(y) == 0:
            return 0.0
        
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Calculate information gain based on Gini impurity."""
        if len(y) == 0:
            return 0.0
        
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        parent_gini = self._gini_impurity(y)
        child_gini = (n_left / n) * self._gini_impurity(y_left) + (n_right / n) * self._gini_impurity(y_right)
        
        return parent_gini - child_gini
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """Find the best split for the data."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Dict[str, Any]:
        """Recursively build the decision tree."""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            return {'leaf': True, 'class': self._most_common_label(y)}
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            return {'leaf': True, 'class': self._most_common_label(y)}
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _most_common_label(self, y: np.ndarray) -> int:
        """Return the most common label in y."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the decision tree to the training data."""
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: Dict[str, Any]) -> int:
        """Predict class for a single sample."""
        if node['leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for all samples in X."""
        return np.array([self._predict_sample(x, self.tree) for x in X])


def evaluate(
    model: DecisionTreeClassifierScratch,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    
    return {
        'mse': mse,
        'r2': r2,
        'accuracy': accuracy
    }


def make_dataloaders(
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and split the Iris dataset."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_val, y_train, y_val


if __name__ == '__main__':
    # Set seed for reproducibility
    set_seed(42)
    
    # Load Iris dataset
    X_train, X_val, y_train, y_val = make_dataloaders(test_size=0.3, random_state=42)
    
    # Train custom decision tree
    custom_tree = DecisionTreeClassifierScratch(max_depth=5, min_samples_split=2)
    custom_tree.fit(X_train, y_train)
    
    # Train sklearn decision tree for comparison
    sklearn_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
    sklearn_tree.fit(X_train, y_train)
    
    # Make predictions
    custom_train_pred = custom_tree.predict(X_train)
    custom_val_pred = custom_tree.predict(X_val)
    sklearn_train_pred = sklearn_tree.predict(X_train)
    sklearn_val_pred = sklearn_tree.predict(X_val)
    
    # Calculate metrics
    custom_train_acc = accuracy_score(y_train, custom_train_pred)
    custom_val_acc = accuracy_score(y_val, custom_val_pred)
    sklearn_train_acc = accuracy_score(y_train, sklearn_train_pred)
    sklearn_val_acc = accuracy_score(y_val, sklearn_val_pred)
    
    # Evaluate on validation set
    val_metrics = evaluate(custom_tree, X_val, y_val)
    
    # Print results
    print("=" * 60)
    print("Decision Tree Classifier Results on Iris Dataset")
    print("=" * 60)
    print(f"Custom Tree - Train Accuracy: {custom_train_acc:.4f}")
    print(f"Custom Tree - Validation Accuracy: {custom_val_acc:.4f}")
    print(f"Sklearn Tree - Train Accuracy: {sklearn_train_acc:.4f}")
    print(f"Sklearn Tree - Validation Accuracy: {sklearn_val_acc:.4f}")
    print(f"MSE: {val_metrics['mse']:.4f}")
    print(f"R2 Score: {val_metrics['r2']:.4f}")
    print("=" * 60)
    
    # Assert quality thresholds
    # Custom tree should match sklearn within 5%
    acc_diff = abs(custom_val_acc - sklearn_val_acc)
    print(f"Accuracy difference between custom and sklearn: {acc_diff:.4f}")
    assert acc_diff <= 0.05, f"Custom tree accuracy differs from sklearn by more than 5%: {acc_diff}"
    assert custom_val_acc >= 0.90, f"Custom tree validation accuracy should be >= 0.90: {custom_val_acc}"
    assert val_metrics['r2'] >= 0.90, f"R2 score should be >= 0.90: {val_metrics['r2']}"
    
    print("All quality thresholds passed!")
    print("=" * 60)
    
    # Save model
    with open('dtree_model.pkl', 'wb') as f:
        pickle.dump(custom_tree, f)
    
    exit(0)
