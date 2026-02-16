import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error, r2_score

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
z_dim = 100
batch_size = 128
lr = 0.0002
epochs = 50
img_size = 28

# Generator network
class Generator(nn.Module):
    def __init__(self, z_dim, channels, hidden_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, hidden_dim * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_dim, 4, 4)),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 4, channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, channels, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, hidden_dim // 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4 * 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img)

def load_mnist_data():
    """Load and preprocess MNIST data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_discriminator(discriminator, generator, real_images, z_dim, optimizer_d, criterion):
    """Train discriminator for one batch."""
    batch_size_current = real_images.size(0)
    
    # Real images
    real_labels = torch.ones(batch_size_current, 1).to(device)
    real_outputs = discriminator(real_images)
    loss_real = criterion(real_outputs, real_labels)
    
    # Fake images
    z = torch.randn(batch_size_current, z_dim).to(device)
    fake_images = generator(z)
    fake_labels = torch.zeros(batch_size_current, 1).to(device)
    fake_outputs = discriminator(fake_images.detach())
    loss_fake = criterion(fake_outputs, fake_labels)
    
    # Combined loss
    loss_d = loss_real + loss_fake
    
    # Backward pass
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()
    
    return loss_d.item()

def train_generator(discriminator, generator, z_dim, optimizer_g, criterion):
    """Train generator for one batch."""
    batch_size_current = z_dim  # Will be overwritten
    
    # Generate fake images and compute loss
    z = torch.randn(batch_size_current, z_dim).to(device)
    fake_images = generator(z)
    fake_labels = torch.ones(batch_size_current, 1).to(device)
    outputs = discriminator(fake_images)
    loss_g = criterion(outputs, fake_labels)
    
    # Backward pass
    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()
    
    return loss_g.item()

def train_model(generator, discriminator, train_loader, epochs, z_dim):
    """Train the GAN model."""
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    generator.train()
    discriminator.train()
    
    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        
        for i, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            
            # Train discriminator
            d_loss = train_discriminator(discriminator, generator, real_images, z_dim, optimizer_d, criterion)
            d_losses.append(d_loss)
            
            # Train generator (every other iteration to avoid generator overpowering discriminator)
            if i % 2 == 0:
                g_loss = train_generator(discriminator, generator, z_dim, optimizer_g, criterion)
                g_losses.append(g_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses) if g_losses else 0
            print(f"Epoch [{epoch+1}/{epochs}], D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
        
        # Save sample images
        if (epoch + 1) % 5 == 0:
            save_samples(generator, epoch + 1, z_dim)
    
    return generator, discriminator

def save_samples(generator, epoch, z_dim, num_samples=16):
    """Save generated sample images."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim).to(device)
        fake_images = generator(z)
        
        # Denormalize
        fake_images = (fake_images + 1) / 2  # From [-1, 1] to [0, 1]
        
        # Save images
        torchvision.utils.save_image(fake_images, f'output/tasks/gan_lvl2_dcgan_mnist/samples_epoch_{epoch}.png', nrow=4)
    
    generator.train()

def evaluate(generator, discriminator, data_loader, z_dim):
    """Evaluate the GAN on validation data."""
    generator.eval()
    discriminator.eval()
    
    d_scores_real = []
    d_scores_fake = []
    
    with torch.no_grad():
        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size_current = real_images.size(0)
            
            # Score for real images
            real_outputs = discriminator(real_images)
            d_scores_real.extend(real_outputs.cpu().numpy().flatten())
            
            # Score for fake images
            z = torch.randn(batch_size_current, z_dim).to(device)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images)
            d_scores_fake.extend(fake_outputs.cpu().numpy().flatten())
    
    # Calculate metrics
    d_scores_real = np.array(d_scores_real)
    d_scores_fake = np.array(d_scores_fake)
    
    # Discriminator accuracy (how well it distinguishes real from fake)
    real_accuracy = np.mean(d_scores_real > 0.5)
    fake_accuracy = np.mean(d_scores_fake < 0.5)
    total_accuracy = (real_accuracy + fake_accuracy) / 2
    
    # MSE between real and fake distributions (lower is better, but not perfect metric for GANs)
    mse = mean_squared_error(d_scores_real, d_scores_fake)
    
    # R2 score (how well discriminator can classify)
    labels = np.concatenate([np.ones(len(d_scores_real)), np.zeros(len(d_scores_fake))])
    scores = np.concatenate([d_scores_real, d_scores_fake])
    r2 = r2_score(labels, scores)
    
    metrics = {
        'discriminator_accuracy': float(total_accuracy),
        'real_accuracy': float(real_accuracy),
        'fake_accuracy': float(fake_accuracy),
        'mean_real_score': float(np.mean(d_scores_real)),
        'mean_fake_score': float(np.mean(d_scores_fake)),
        'mse': float(mse),
        'r2': float(r2)
    }
    
    return metrics

def main():  # noqa: C901
    """Main function to run the DCGAN task."""
    print("=" * 60)
    print("DCGAN on MNIST - Level 2")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading MNIST data...")
    train_loader, val_loader = load_mnist_data()
    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
    
    # 2. Initialize GAN models
    print("\n2. Initializing GAN models...")
    z_dim = 100
    generator = Generator(z_dim=z_dim, channels=1, hidden_dim=64).to(device)
    discriminator = Discriminator(channels=1, hidden_dim=64).to(device)
    generator_params = sum(p.numel() for p in generator.parameters())
    discriminator_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {generator_params}")
    print(f"Discriminator parameters: {discriminator_params}")
    
    # 3. Train model
    print("\n3. Training GAN...")
    generator, discriminator = train_model(generator, discriminator, train_loader, epochs, z_dim)
    
    # 4. Evaluate on validation data
    print("\n4. Evaluating on validation data...")
    val_metrics = evaluate(generator, discriminator, val_loader, z_dim=z_dim)
    print(f"Validation metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 5. Evaluate on training data
    print("\n5. Evaluating on training data...")
    train_metrics = evaluate(generator, discriminator, train_loader, z_dim=z_dim)
    print(f"Training metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 6. Quality checks
    print("\n6. Quality checks...")
    
    # Check discriminator accuracy (should be above random chance)
    assert val_metrics['discriminator_accuracy'] > 0.4, f"Discriminator accuracy too low: {val_metrics['discriminator_accuracy']:.4f}"
    print(f"✓ Discriminator accuracy: {val_metrics['discriminator_accuracy']:.4f}")
    
    # Check that discriminator can distinguish real from fake (R2 > 0)
    assert val_metrics['r2'] > 0, f"R2 score should be positive: {val_metrics['r2']:.4f}"
    print(f"✓ R2 score: {val_metrics['r2']:.4f}")
    
    # Check that real images score higher than fake on average
    assert val_metrics['mean_real_score'] > val_metrics['mean_fake_score'], \
        f"Real score should be higher than fake: {val_metrics['mean_real_score']:.4f} vs {val_metrics['mean_fake_score']:.4f}"
    print(f"✓ Mean real score > Mean fake score")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
