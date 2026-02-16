#!/usr/bin/env python3
"""
WGAN-GP (Wasserstein GAN with Gradient Penalty) implementation for toy data generation.
Level 3 GAN task - implements WGAN-GP with proper gradient penalty.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad as torch_grad
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    """Generator network for WGAN-GP."""
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=2):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 4, output_dim)
        )
    
    def forward(self, z):
        return self.model(z)


class Critic(nn.Module):
    """Critic (Discriminator) network for WGAN-GP."""
    def __init__(self, input_dim=2, hidden_dim=128):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.model(x)


def generate_toy_data(n_samples=1000):
    """Generate synthetic 2D toy data with complex structure."""
    # Create a moon-like distribution
    n_samples_per_moon = n_samples // 2
    
    # First moon
    theta1 = np.linspace(0, np.pi, n_samples_per_moon)
    x1 = np.cos(theta1) + np.random.normal(0, 0.1, n_samples_per_moon)
    y1 = np.sin(theta1) + np.random.normal(0, 0.1, n_samples_per_moon)
    
    # Second moon (shifted and inverted)
    theta2 = np.linspace(0, np.pi, n_samples_per_moon)
    x2 = np.cos(theta2) + 1 + np.random.normal(0, 0.1, n_samples_per_moon)
    y2 = -np.sin(theta2) + 0.5 + np.random.normal(0, 0.1, n_samples_per_moon)
    
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    return X.astype(np.float32)


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Compute gradient penalty for WGAN-GP."""
    batch_size = real_samples.size(0)
    
    # Random weight for interpolation
    alpha = torch.rand(batch_size, 1).to(device)
    
    # Interpolate between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # Get critic score for interpolated samples
    d_interpolates = critic(interpolates)
    
    # Compute gradients
    gradients = torch_grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


def train_wgan_gp(
    generator,
    critic,
    real_data,
    n_epochs=1000,
    batch_size=64,
    n_critic=5,
    lr_g=0.0001,
    lr_c=0.0001,
    lambda_gp=10,
    noise_dim=100
):
    """Train WGAN-GP model."""
    real_data = torch.from_numpy(real_data).to(device)
    n_samples = real_data.size(0)
    
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
    optimizer_c = optim.Adam(critic.parameters(), lr=lr_c, betas=(0.5, 0.9))
    
    # Training tracking
    losses = {'g_losses': [], 'c_losses': [], 'gp_losses': []}
    
    print(f"Training WGAN-GP for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        # Sample random noise
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_data = generator(z)
        
        # Sample real data
        idx = torch.randint(0, n_samples, (batch_size,))
        real_batch = real_data[idx]
        
        # Train Critic (multiple times per generator update)
        for _ in range(n_critic):
            optimizer_c.zero_grad()
            
            # Real samples
            real_score = critic(real_batch)
            
            # Fake samples
            fake_score = critic(fake_data.detach())
            
            # Compute critic loss (Wasserstein distance)
            c_loss = -(torch.mean(real_score) - torch.mean(fake_score))
            
            # Compute gradient penalty
            gp = compute_gradient_penalty(critic, real_batch, fake_data.detach(), device)
            gp_loss = lambda_gp * gp
            
            # Total critic loss
            total_c_loss = c_loss + gp_loss
            
            total_c_loss.backward()
            optimizer_c.step()
        
        # Train Generator
        optimizer_g.zero_grad()
        
        # Generate new fake samples
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_data = generator(z)
        fake_score = critic(fake_data)
        
        # Generator loss (minimize negative mean of critic scores)
        g_loss = -torch.mean(fake_score)
        g_loss.backward()
        optimizer_g.step()
        
        # Track losses
        losses['c_losses'].append(c_loss.item())
        losses['g_losses'].append(g_loss.item())
        losses['gp_losses'].append(gp_loss.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], "
                  f"Critic Loss: {c_loss.item():.4f}, "
                  f"Generator Loss: {g_loss.item():.4f}, "
                  f"GP Loss: {gp_loss.item():.4f}")
    
    return losses


def evaluate(generator, real_data, noise_dim=100, n_samples=500):
    """Evaluate generator performance."""
    generator.eval()
    
    # Generate samples
    with torch.no_grad():
        z = torch.randn(n_samples, noise_dim).to(device)
        generated_data = generator(z).cpu().numpy()
    
    # Compute statistics
    real_mean = np.mean(real_data, axis=0)
    real_std = np.std(real_data, axis=0)
    gen_mean = np.mean(generated_data, axis=0)
    gen_std = np.std(generated_data, axis=0)
    
    # Compute metrics
    mse = mean_squared_error(real_data.flatten(), generated_data.flatten())
    r2 = r2_score(real_data.flatten(), generated_data.flatten())
    
    # Parameter accuracy
    mean_diff = np.mean(np.abs(real_mean - gen_mean))
    std_diff = np.mean(np.abs(real_std - gen_std))
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'real_mean': real_mean.tolist(),
        'gen_mean': gen_mean.tolist(),
        'real_std': real_std.tolist(),
        'gen_std': gen_std.tolist(),
        'generated_data': generated_data
    }


def visualize_results(real_data, generated_data, losses, save_dir='.'):
    """Generate and save visualizations."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Real vs Generated data
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(real_data[:, 0], real_data[:, 1], c='blue', alpha=0.6, label='Real')
    plt.title('Real Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(generated_data[:, 0], generated_data[:, 1], c='red', alpha=0.6, label='Generated')
    plt.title('Generated Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'gan_wgan_gp_samples.png', dpi=150)
    plt.close()
    
    # Plot 2: Training curves
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses['g_losses'], label='Generator Loss')
    plt.plot(losses['c_losses'], label='Critic Loss')
    plt.plot(losses['gp_losses'], label='GP Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(losses['g_losses'], label='Generator Loss')
    plt.title('Generator Loss (Zoomed)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'gan_wgan_gp_training.png', dpi=150)
    plt.close()
    
    print(f"Saved visualizations to {save_path}")


def main():  # noqa: C901
    """Main function to run the WGAN-GP task."""
    print("=" * 60)
    print("WGAN-GP (Wasserstein GAN with Gradient Penalty)")
    print("=" * 60)
    
    # 1. Generate real data
    print("\n1. Generating real data...")
    real_data = generate_toy_data(n_samples=1000)
    print(f"Real data shape: {real_data.shape}")
    
    # Split data for validation
    split_idx = int(0.8 * len(real_data))
    train_data = real_data[:split_idx]
    val_data = real_data[split_idx:]
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # 2. Initialize models
    print("\n2. Initializing models...")
    generator = Generator(input_dim=100, hidden_dim=128, output_dim=2).to(device)
    critic = Critic(input_dim=2, hidden_dim=128).to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters())}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters())}")
    
    # 3. Train model
    print("\n3. Training WGAN-GP model...")
    losses = train_wgan_gp(
        generator,
        critic,
        train_data,
        n_epochs=1000,
        batch_size=64,
        n_critic=5,
        lr_g=0.0001,
        lr_c=0.0001,
        lambda_gp=10,
        noise_dim=100
    )
    
    # 4. Evaluate on training data
    print("\n4. Evaluating on training data...")
    train_metrics = evaluate(generator, train_data, noise_dim=100, n_samples=500)
    print(f"Train MSE: {train_metrics['mse']:.6f}")
    print(f"Train R²: {train_metrics['r2']:.6f}")
    print(f"Train Mean Diff: {train_metrics['mean_diff']:.6f}")
    print(f"Train Std Diff: {train_metrics['std_diff']:.6f}")
    
    # 5. Evaluate on validation data
    print("\n5. Evaluating on validation data...")
    val_metrics = evaluate(generator, val_data, noise_dim=100, n_samples=500)
    print(f"Validation MSE: {val_metrics['mse']:.6f}")
    print(f"Validation R²: {val_metrics['r2']:.6f}")
    print(f"Validation Mean Diff: {val_metrics['mean_diff']:.6f}")
    print(f"Validation Std Diff: {val_metrics['std_diff']:.6f}")
    
    # 6. Generate visualizations
    print("\n6. Generating visualizations...")
    visualize_results(train_data, train_metrics['generated_data'], losses, save_dir='.')
    
    # 7. Quality checks
    print("\n7. Quality checks...")
    
    # Check that critic loss is reasonable (Wasserstein distance property)
    assert losses['c_losses'][-1] < 0, \
        f"Critic loss should be negative (Wasserstein distance): {losses['c_losses'][-1]:.4f}"
    
    # Check that generator loss is decreasing (general trend)
    assert losses['g_losses'][-1] < losses['g_losses'][0], \
        f"Generator loss should decrease: {losses['g_losses'][0]:.4f} -> {losses['g_losses'][-1]:.4f}"
    
    # Check R² is reasonable (generated data should capture some structure)
    assert val_metrics['r2'] > 0.5, \
        f"R² score should be > 0.5 for reasonable generation: {val_metrics['r2']:.4f}"
    
    # Check mean difference is small
    assert val_metrics['mean_diff'] < 0.5, \
        f"Mean difference should be small: {val_metrics['mean_diff']:.4f}"
    
    # Check std difference is small
    assert val_metrics['std_diff'] < 0.5, \
        f"Std difference should be small: {val_metrics['std_diff']:.4f}"
    
    # Check that losses are stable (not exploding)
    assert not np.isnan(losses['g_losses'][-1]), \
        "Generator loss contains NaN values"
    assert not np.isnan(losses['c_losses'][-1]), \
        "Critic loss contains NaN values"
    
    print("All quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
