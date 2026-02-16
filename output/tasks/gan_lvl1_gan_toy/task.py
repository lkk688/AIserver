"""GAN for Toy Data Generation - Level 1"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import json

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def generate_target_data(n_samples=1000):
    """Generate target distribution data (mixture of Gaussians)."""
    # Create a mixture of 3 Gaussians
    n_per_component = n_samples // 3
    component1 = np.random.randn(n_per_component, 2) * 0.5 + np.array([2, 2])
    component2 = np.random.randn(n_per_component, 2) * 0.5 + np.array([-2, -2])
    component3 = np.random.randn(n_samples - 2 * n_per_component, 2) * 0.5 + np.array([2, -2])
    return np.vstack([component1, component2, component3]).astype(np.float32)


class Generator(nn.Module):
    """Generator network for GAN."""
    def __init__(self, input_dim=10, output_dim=2):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """Discriminator network for GAN."""
    def __init__(self, input_dim=2):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


def train(generator, discriminator, X_real, epochs=1000, batch_size=64, lr=0.0002):
    """Train GAN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Training tracking
    g_losses = []
    d_losses = []
    
    X_real = torch.FloatTensor(X_real).to(device)
    n_samples = len(X_real)
    
    print(f"Training GAN for {epochs} epochs...")
    
    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Sample random batch
        idx = np.random.randint(0, n_samples, batch_size)
        real_samples = X_real[idx]
        
        # Sample noise for generator
        z = torch.FloatTensor(np.random.randn(batch_size, 10)).to(device)
        
        # Generate fake samples
        fake_samples = generator(z)
        
        # Real and fake labels
        valid = torch.FloatTensor(batch_size, 1).fill_(1.0).to(device)
        fake = torch.FloatTensor(batch_size, 1).fill_(0.0).to(device)
        
        # Discriminator loss
        real_loss = adversarial_loss(discriminator(real_samples), valid)
        fake_loss = adversarial_loss(discriminator(fake_samples.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        
        # Sample noise
        z = torch.FloatTensor(np.random.randn(batch_size, 10)).to(device)
        
        # Generate fake samples
        gen_samples = generator(z)
        
        # Generator loss
        g_loss = adversarial_loss(discriminator(gen_samples), valid)
        
        g_loss.backward()
        optimizer_G.step()
        
        # Track losses
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
        # Print progress
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    print(f"Final D Loss: {d_losses[-1]:.4f}, Final G Loss: {g_losses[-1]:.4f}")
    
    return g_losses, d_losses


def evaluate(model, X):
    """
    Evaluate the model on data.
    For GAN, we measure how well the generated data matches the target distribution.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate samples from the generator
    n_samples = len(X)
    z = torch.FloatTensor(np.random.randn(n_samples, 10)).to(device)
    model.eval()
    
    with torch.no_grad():
        generated = model(z).cpu().numpy()
    
    # Compute MMD-like metric (simplified)
    # Compare means and covariances
    real_mean = np.mean(X, axis=0)
    real_cov = np.cov(X.T)
    gen_mean = np.mean(generated, axis=0)
    gen_cov = np.cov(generated.T)
    
    # MSE between means
    mean_mse = mean_squared_error(real_mean, gen_mean)
    
    # MSE between covariance matrices (flattened upper triangle)
    real_cov_triu = real_cov[np.triu_indices(2)]
    gen_cov_triu = gen_cov[np.triu_indices(2)]
    cov_mse = mean_squared_error(real_cov_triu, gen_cov_triu)
    
    # R2 score for mean comparison
    r2_mean = r2_score(real_mean, gen_mean)
    
    return {
        'mean_mse': float(mean_mse),
        'cov_mse': float(cov_mse),
        'r2_mean': float(r2_mean),
        'generated_mean': gen_mean.tolist(),
        'generated_cov': gen_cov.tolist()
    }


def save_metrics(metrics, save_dir='.'):
    """Save metrics to JSON file."""
    save_path = Path(save_dir) / 'metrics.json'
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {save_path}")


def main():  # noqa: C901
    """Main function to run the GAN task."""
    print("=" * 60)
    print("GAN for Toy Data Generation - Level 1")
    print("=" * 60)
    
    # 1. Generate target data
    print("\n1. Generating target data...")
    X_real = generate_target_data(n_samples=1000)
    print(f"Generated {len(X_real)} samples with shape {X_real.shape}")
    
    # 2. Split data into train and validation
    print("\n2. Splitting data...")
    split_idx = int(0.8 * len(X_real))
    X_train = X_real[:split_idx]
    X_val = X_real[split_idx:]
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 3. Initialize models
    print("\n3. Initializing GAN models...")
    generator = Generator(input_dim=10, output_dim=2)
    discriminator = Discriminator(input_dim=2)
    
    n_params_g = sum(p.numel() for p in generator.parameters())
    n_params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {n_params_g}, Discriminator parameters: {n_params_d}")
    
    # 4. Train GAN
    print("\n4. Training GAN...")
    g_losses, d_losses = train(
        generator, discriminator, X_train,
        epochs=1000, batch_size=64, lr=0.0002
    )
    
    # 5. Evaluate on training data
    print("\n5. Evaluating on training data...")
    train_metrics = evaluate(generator, X_train)
    print(f"Train Mean MSE: {train_metrics['mean_mse']:.4f}")
    print(f"Train Cov MSE: {train_metrics['cov_mse']:.4f}")
    print(f"Train R2 (mean): {train_metrics['r2_mean']:.4f}")
    
    # 6. Evaluate on validation data
    print("\n6. Evaluating on validation data...")
    val_metrics = evaluate(generator, X_val)
    print(f"Validation Mean MSE: {val_metrics['mean_mse']:.4f}")
    print(f"Validation Cov MSE: {val_metrics['cov_mse']:.4f}")
    print(f"Validation R2 (mean): {val_metrics['r2_mean']:.4f}")
    
    # 7. Save metrics
    print("\n7. Saving metrics...")
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'g_losses': g_losses,
        'd_losses': d_losses
    }
    save_metrics(metrics, save_dir='.')
    
    # 8. Quality checks
    print("\n8. Quality checks...")
    
    # Check R2 score is reasonable
    assert val_metrics['r2_mean'] > 0.5, f"R2 score too low: {val_metrics['r2_mean']:.4f}"
    print(f"✓ R2 score is reasonable: {val_metrics['r2_mean']:.4f} > 0.5")
    
    # Check MSE is low
    assert val_metrics['mean_mse'] < 1.0, f"Mean MSE too high: {val_metrics['mean_mse']:.4f}"
    print(f"✓ Mean MSE is low: {val_metrics['mean_mse']:.4f} < 1.0")
    
    # Check covariance MSE is reasonable
    assert val_metrics['cov_mse'] < 1.0, f"Covariance MSE too high: {val_metrics['cov_mse']:.4f}"
    print(f"✓ Covariance MSE is reasonable: {val_metrics['cov_mse']:.4f} < 1.0")
    
    # Check that GAN loss decreased
    assert g_losses[-1] < g_losses[0], f"Generator loss should decrease: {g_losses[0]:.4f} -> {g_losses[-1]:.4f}"
    print(f"✓ Generator loss decreased: {g_losses[0]:.4f} -> {g_losses[-1]:.4f}")
    
    print("\n" + "=" * 60)
    print("All quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
