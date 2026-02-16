#!/usr/bin/env python3
"""Linear Autoencoder - Level 1: Implement a linear autoencoder model."""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class LinearAutoencoder(nn.Module):
    """Linear autoencoder with encoder and decoder."""
    
    def __init__(self, input_dim, latent_dim):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def split_data(X, train_ratio=0.8):
    """Split data into train and validation sets."""
    return train_test_split(X, train_size=train_ratio, random_state=42)


def train_ae(model, X_train, X_val, learning_rate=0.01, epochs=500, batch_size=32):
    """Train the linear autoencoder."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    
    train_losses = []
    val_losses = []
    
    n_samples = len(X_train)
    n_batches = max(1, n_samples // batch_size)
    
    print(f"Training for {epochs} epochs with learning rate {learning_rate}...")
    
    for epoch in range(epochs):
        # Mini-batch training
        epoch_train_loss = 0.0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_X = X_train_t[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        train_loss = epoch_train_loss / n_batches
        train_losses.append(train_loss)
        
        # Validation loss
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, X_val_t)
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss.item():.6f}")
    
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    
    return train_losses, val_losses


def evaluate(model, X):
    """Evaluate the autoencoder and return metrics."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        reconstructed = model(X_t).numpy()
    
    # Calculate metrics
    mse = mean_squared_error(X, reconstructed)
    r2 = r2_score(X.flatten(), reconstructed.flatten())
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'reconstructed': reconstructed
    }


def save_model(model, save_dir='.'):
    """Save the model to file."""
    save_path = Path(save_dir) / 'model.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


def save_metrics(metrics, save_dir='.'):
    """Save metrics to JSON file."""
    save_path = Path(save_dir) / 'metrics.json'
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {save_path}")


def compute_pca_reconstruction(X, n_components):
    """Compute PCA reconstruction and return MSE."""
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_reconstructed = pca.inverse_transform(pca.transform(X))
    mse = mean_squared_error(X, X_reconstructed)
    r2 = r2_score(X.flatten(), X_reconstructed.flatten())
    return mse, r2, X_reconstructed


def main():  # noqa: C901
    """Main function to run the linear autoencoder task."""
    print("=" * 60)
    print("Linear Autoencoder - Level 1")
    print("=" * 60)
    
    # 1. Generate synthetic data with strong correlations
    print("\n1. Generating synthetic data with strong correlations...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Create correlated data for better reconstruction
    X = np.random.randn(n_samples, n_features)
    # Add correlations to make reconstruction easier
    X[:, 1] = 0.8 * X[:, 0] + 0.2 * np.random.randn(n_samples)
    X[:, 2] = 0.7 * X[:, 0] + 0.5 * X[:, 1] + 0.2 * np.random.randn(n_samples)
    X[:, 3] = 0.9 * X[:, 2] + 0.2 * np.random.randn(n_samples)
    X[:, 4] = 0.85 * X[:, 3] + 0.15 * np.random.randn(n_samples)
    
    # Normalize data
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    print(f"X shape: {X.shape}")
    
    # 2. Split data
    print("\n2. Splitting data into train and validation...")
    X_train, X_val = split_data(X, train_ratio=0.8)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 3. Define linear autoencoder model
    print("\n3. Defining linear autoencoder...")
    input_dim = n_features
    latent_dim = 3  # Lower dimensional representation
    model = LinearAutoencoder(input_dim, latent_dim)
    print(f"Model: {input_dim} -> {latent_dim} -> {input_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 4. Train autoencoder model
    print("\n4. Training autoencoder...")
    train_losses, val_losses = train_ae(
        model, X_train, X_val,
        learning_rate=0.01, epochs=1000, batch_size=64
    )
    
    # 5. Evaluate on training data
    print("\n5. Evaluating on training data split...")
    train_metrics = evaluate(model, X_train)
    print(f"Training MSE: {train_metrics['mse']:.6f}")
    print(f"Training R²: {train_metrics['r2']:.6f}")
    
    # 6. Evaluate on validation data
    print("\n6. Evaluating on validation data split...")
    val_metrics = evaluate(model, X_val)
    print(f"Validation MSE: {val_metrics['mse']:.6f}")
    print(f"Validation R²: {val_metrics['r2']:.6f}")
    
    # 7. Compare with PCA baseline
    print("\n7. Comparing reconstruction error with PCA baseline...")
    pca_mse, pca_r2, _ = compute_pca_reconstruction(X_val, n_components=latent_dim)
    print(f"PCA MSE: {pca_mse:.6f}")
    print(f"PCA R²: {pca_r2:.6f}")
    ae_vs_pca_ratio = val_metrics['mse'] / pca_mse
    print(f"AE/PCA reconstruction error ratio: {ae_vs_pca_ratio:.4f}")
    
    # 8. Save model and metrics
    print("\n8. Saving model and metrics...")
    save_model(model)
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'pca_baseline': {'mse': pca_mse, 'r2': pca_r2},
        'ae_vs_pca_ratio': ae_vs_pca_ratio,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    save_metrics(all_metrics)
    
    # 9. Run quality checks
    print("\n9. Running quality checks...")
    
    # Check R² score on validation (should be high, > 0.9 as per requirements)
    assert val_metrics['r2'] > 0.9, f"R² should be > 0.9, got {val_metrics['r2']:.4f}"
    print(f"✓ Validation R² > 0.9: {val_metrics['r2']:.4f}")
    
    # Check MSE is reasonable
    assert val_metrics['mse'] < 0.3, f"MSE should be < 0.3, got {val_metrics['mse']:.4f}"
    print(f"✓ Validation MSE < 0.3: {val_metrics['mse']:.4f}")
    
    # Check AE reconstruction is close to PCA (within 20%)
    assert ae_vs_pca_ratio < 1.2, f"AE reconstruction should be close to PCA, ratio {ae_vs_pca_ratio:.4f} >= 1.2"
    print(f"✓ AE reconstruction close to PCA: ratio {ae_vs_pca_ratio:.4f} < 1.2")
    
    # Check training loss decreased
    assert train_losses[-1] < train_losses[0], "Training loss should decrease"
    print(f"✓ Training loss decreased: {train_losses[0]:.6f} -> {train_losses[-1]:.6f}")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
