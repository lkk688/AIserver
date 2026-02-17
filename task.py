import torch
import numpy as np


class UnivariateLinearRegression:
    """Univariate Linear Regression using only PyTorch tensors."""
    
    def __init__(self, device='cpu'):
        self.device = device
        # Initialize weights and bias
        self.w = torch.randn(1, device=device)
        self.b = torch.randn(1, device=device)
    
    def forward(self, X):
        """Forward pass: y = wx + b"""
        return self.w * X + self.b
    
    def compute_loss(self, y_pred, y_true):
        """Compute Mean Squared Error loss"""
        return torch.mean((y_pred - y_true) ** 2)
    
    def backward(self, X, y_pred, y_true):
        """Compute gradients manually"""
        n = X.shape[0]
        
        # dL/dw = (2/n) * sum((y_pred - y_true) * x)
        grad_w = torch.mean(2 * (y_pred - y_true) * X)
        
        # dL/db = (2/n) * sum(y_pred - y_true)
        grad_b = torch.mean(2 * (y_pred - y_true))
        
        return grad_w, grad_b
    
    def update_parameters(self, grad_w, grad_b, learning_rate):
        """Update parameters using gradient descent"""
        with torch.no_grad():
            self.w -= learning_rate * grad_w
            self.b -= learning_rate * grad_b
    
    def predict(self, X):
        """Make predictions"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        return self.forward(X_tensor)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            learning_rate=0.01, epochs=1000, verbose=True):
        """Train the model"""
        # Convert to tensors and move to device
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X_train)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y_train)
            
            # Backward pass
            grad_w, grad_b = self.backward(X_train, y_pred, y_train)
            
            # Update parameters
            self.update_parameters(grad_w, grad_b, learning_rate)
            
            # Validation loss if validation data provided
            if X_val is not None and y_val is not None and verbose and epoch % 100 == 0:
                with torch.no_grad():
                    y_val_pred = self.forward(X_val)
                    val_loss = self.compute_loss(y_val_pred, y_val)
                print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")
            elif verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model and return predictions and metrics"""
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        with torch.no_grad():
            y_pred = self.forward(X_test)
            loss = self.compute_loss(y_pred, y_test)
        
        # Convert to numpy for additional metrics
        y_pred_np = y_pred.detach().cpu().numpy()
        y_test_np = y_test.detach().cpu().numpy()
        
        # Calculate R-squared
        ss_res = np.sum((y_test_np - y_pred_np) ** 2)
        ss_tot = np.sum((y_test_np - np.mean(y_test_np)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate MAE
        mae = np.mean(np.abs(y_test_np - y_pred_np))
        
        return {
            'loss': loss.item(),
            'predictions': y_pred_np,
            'r_squared': r_squared,
            'mae': mae
        }


def generate_synthetic_data(n_samples=100, noise_level=0.5, train_ratio=0.7, val_ratio=0.15):
    """Generate synthetic data for univariate linear regression"""
    np.random.seed(42)
    
    # Generate x values
    X = np.random.uniform(-10, 10, n_samples)
    
    # True relationship: y = 3x + 5 + noise
    true_w = 3.0
    true_b = 5.0
    noise = np.random.normal(0, noise_level, n_samples)
    y = true_w * X + true_b + noise
    
    # Split data
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    X_train, X_test, X_val = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_test, y_val = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    
    return X_train, X_test, X_val, y_train, y_test, y_val


def main():
    """Main function to demonstrate univariate linear regression"""
    print("=" * 60)
    print("Univariate Linear Regression using PyTorch Tensors")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X_train, X_test, X_val, y_train, y_test, y_val = generate_synthetic_data(
        n_samples=150, noise_level=1.0, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}\n")
    
    # Initialize model
    model = UnivariateLinearRegression(device=device)
    print(f"Initial weights: w = {model.w.item():.4f}, b = {model.b.item():.4f}\n")
    
    # Train model
    print("Training model...")
    model.fit(
        X_train, y_train, 
        X_val=X_val, y_val=y_val,
        learning_rate=0.01, 
        epochs=1000, 
        verbose=True
    )
    
    print(f"\nFinal weights: w = {model.w.item():.4f}, b = {model.b.item():.4f}")
    print("True weights: w = 3.0000, b = 5.0000\n")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    results = model.evaluate(X_test, y_test)
    
    print(f"Test Loss (MSE): {results['loss']:.6f}")
    print(f"R-squared: {results['r_squared']:.6f}")
    print(f"MAE: {results['mae']:.6f}\n")
    
    # Validation results
    print("Validation Results:")
    val_results = model.evaluate(X_val, y_val)
    print(f"Validation Loss (MSE): {val_results['loss']:.6f}")
    print(f"Validation R-squared: {val_results['r_squared']:.6f}\n")
    
    # Sample predictions
    print("Sample Predictions (first 5 test samples):")
    for i in range(5):
        print(f"  True: {y_test[i]:.2f}, Predicted: {results['predictions'][i]:.2f}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
