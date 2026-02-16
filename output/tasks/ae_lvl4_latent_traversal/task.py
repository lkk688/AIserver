"""Latent Traversal Task - Autoencoder Level 4"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class Autoencoder(nn.Module):
    """Simple autoencoder for latent traversal."""
    def __init__(self, input_dim=10, latent_dim=2):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


def generate_synthetic_data(n_samples=1000, input_dim=10):
    """Generate synthetic data with latent structure."""
    np.random.seed(42)
    
    # Generate latent variables
    z1 = np.random.randn(n_samples, 1)
    z2 = np.random.randn(n_samples, 1)
    z = np.hstack([z1, z2])
    
    # Generate input data from latent variables
    X = np.hstack([
        0.8 * z1 + 0.6 * z2 + 0.1 * np.random.randn(n_samples, 1),
        0.7 * z1 - 0.7 * z2 + 0.1 * np.random.randn(n_samples, 1),
        0.9 * z1 + 0.1 * z2 + 0.1 * np.random.randn(n_samples, 1),
        0.6 * z1 + 0.8 * z2 + 0.1 * np.random.randn(n_samples, 1),
        0.5 * z1 - 0.5 * z2 + 0.1 * np.random.randn(n_samples, 1),
        0.4 * z1 + 0.9 * z2 + 0.1 * np.random.randn(n_samples, 1),
        0.3 * z1 - 0.8 * z2 + 0.1 * np.random.randn(n_samples, 1),
        0.2 * z1 + 0.7 * z2 + 0.1 * np.random.randn(n_samples, 1),
        0.1 * z1 - 0.6 * z2 + 0.1 * np.random.randn(n_samples, 1),
        0.0 * z1 + 0.5 * z2 + 0.1 * np.random.randn(n_samples, 1),
    ])
    
    return X.astype(np.float32), z.astype(np.float32)


def train(model, X_train, y_train, X_val, y_val, learning_rate=0.001, epochs=500):
    """Train the autoencoder."""
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        # Forward pass
        recon_train, _ = model(X_train_t)
        loss = criterion(recon_train, y_train_t)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record training loss
        train_losses.append(loss.item())
        
        # Compute validation loss
        with torch.no_grad():
            recon_val, _ = model(X_val_t)
            val_loss = criterion(recon_val, y_val_t)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
    
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_loss.item():.6f}")
    
    return train_losses, val_losses


def evaluate(model, X, y):
    """Evaluate model and compute metrics."""
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)
    
    # Get reconstructions and latent codes
    with torch.no_grad():
        reconstructions, latents = model(X_t)
    
    # Calculate reconstruction metrics
    recon_np = reconstructions.numpy()
    y_np = y
    
    mse = mean_squared_error(y_np.flatten(), recon_np.flatten())
    r2 = r2_score(y_np.flatten(), recon_np.flatten())
    
    # Calculate latent space statistics
    latents_np = latents.numpy()
    latent_variance = np.var(latents_np, axis=0)
    latent_mean = np.mean(latents_np, axis=0)
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'latent_mean': latent_mean.tolist(),
        'latent_variance': latent_variance.tolist(),
        'latent_std': np.std(latents_np, axis=0).tolist()
    }


def generate_latent_traversal(model, X_val, save_dir='.', n_traversal=10, n_samples=5):
    """Generate latent traversal images."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Get a sample from validation data
    sample_idx = 0
    with torch.no_grad():
        sample = X_val[sample_idx:sample_idx+1]
        _, z_sample = model(torch.FloatTensor(sample))
        z_sample = z_sample.numpy()
    
    # Generate traversal images for each latent dimension
    n_latent = z_sample.shape[1]
    fig, axes = plt.subplots(n_latent, n_samples, figsize=(3*n_samples, 3*n_latent))
    
    if n_latent == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_latent):
        # Create traversal range
        z_min = z_sample[0, i] - 2 * np.std(z_sample[:, i])
        z_max = z_sample[0, i] + 2 * np.std(z_sample[:, i])
        z_values = np.linspace(z_min, z_max, n_samples)
        
        for j, z_val in enumerate(z_values):
            # Modify latent vector
            z_modified = z_sample.copy()
            z_modified[0, i] = z_val
            
            # Decode
            z_t = torch.FloatTensor(z_modified)
            with torch.no_grad():
                reconstruction = model.decoder(z_t).numpy()
            
            # Plot
            ax = axes[i, j]
            ax.imshow(reconstruction.reshape(5, 2), cmap='viridis', aspect='auto')
            ax.set_title(f'Latent {i}: {z_val:.2f}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'latent_traversal.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved latent traversal to {save_dir / 'latent_traversal.png'}")
    
    return z_sample


def generate_sample_reconstructions(model, X_val, save_dir='.', n_samples=8):
    """Generate sample reconstructions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Select samples
    indices = np.random.choice(len(X_val), min(n_samples, len(X_val)), replace=False)
    samples = X_val[indices]
    
    with torch.no_grad():
        samples_t = torch.FloatTensor(samples)
        reconstructions, _ = model(samples_t)
    
    # Plot
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    
    for i, idx in enumerate(indices):
        # Original
        axes[0, i].imshow(samples[i].reshape(5, 2), cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Original {idx}')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstructions[i].numpy().reshape(5, 2), cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_reconstructions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved sample reconstructions to {save_dir / 'sample_reconstructions.png'}")


def main():  # noqa: C901
    """Main function to run the latent traversal task."""
    print("=" * 60)
    print("Autoencoder Latent Traversal - Level 4")
    
    # Generate data
    print("\n1. Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=1000, input_dim=10)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create model
    print("\n3. Creating autoencoder model...")
    model = Autoencoder(input_dim=10, latent_dim=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\n4. Training model...")
    train_losses, val_losses = train(
        model, X_train, y_train, X_val, y_val,
        learning_rate=0.001, epochs=500
    )
    
    # Evaluate on training data
    print("\n5. Evaluating on training data...")
    train_metrics = evaluate(model, X_train, y_train)
    print(f"Train MSE: {train_metrics['mse']:.6f}")
    print(f"Train R²: {train_metrics['r2']:.6f}")
    
    # Evaluate on validation data
    print("\n6. Evaluating on validation data...")
    val_metrics = evaluate(model, X_val, y_val)
    print(f"Validation MSE: {val_metrics['mse']:.6f}")
    print(f"Validation R²: {val_metrics['r2']:.6f}")
    
    # Generate visualizations
    print("\n7. Generating visualizations...")
    save_dir = Path('output/tasks/ae_lvl4_latent_traversal')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: training_curves.png")
    
    # Generate sample reconstructions
    print("\n8. Generating sample reconstructions...")
    generate_sample_reconstructions(model, X_val, save_dir=str(save_dir))
    
    # Generate latent traversal
    print("\n9. Generating latent traversal...")
    generate_latent_traversal(model, X_val, save_dir=str(save_dir))
    
    # Save metrics
    print("\n10. Saving metrics...")
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    import json
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {save_dir / 'metrics.json'}")
    
    # Quality checks
    print("\n11. Quality checks...")
    
    # Check R² score
    assert val_metrics['r2'] > 0.8, f"Validation R² should be > 0.8, got {val_metrics['r2']:.4f}"
    print(f"✓ Validation R² > 0.8: {val_metrics['r2']:.4f}")
    
    # Check MSE is reasonable
    assert val_metrics['mse'] < 0.5, f"Validation MSE should be < 0.5, got {val_metrics['mse']:.4f}"
    print(f"✓ Validation MSE < 0.5: {val_metrics['mse']:.4f}")
    
    # Check loss decreased
    assert train_losses[-1] < train_losses[0], "Training loss should decrease"
    print(f"✓ Training loss decreased: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
