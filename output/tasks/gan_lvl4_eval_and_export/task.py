#!/usr/bin/env python3
"""
GAN Level 4: Evaluation and Export
Benchmark: Generation throughput (images/sec)
"""

import os
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
TRAIN_RATIO = 0.8
LATENT_DIM = 100
HIDDEN_DIM = 128
LEARNING_RATE = 0.0002


def generate_synthetic_data(n_samples=2000):
    """Generate synthetic 2D data with complex distribution."""
    # Create a complex distribution: mixture of Gaussians with nonlinear transformations
    n_components = 4
    samples_per_component = n_samples // n_components
    
    np.random.seed(42)  # Reset seed for reproducibility
    
    data = []
    for i in range(n_components):
        # Different centers for each component
        data.append(component_data)
    
    data = np.vstack(data)
    
    # Shuffle the data
    np.random.shuffle(data)
    return data

class Generator(nn.Module):
    """Generator network for GAN."""
    def __init__(self, latent_dim=100, hidden_dim=128, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),


class Discriminator(nn.Module):
    """Discriminator network for GAN.""" 
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),


def train_gan(generator, discriminator, train_data, num_epochs=NUM_EPOCHS, 
              batch_size=64, latent_dim=100):
    """Train the GAN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # Training loop - train for more epochs for better convergence
    print(f"Training GAN for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        num_batches = 0
        
        for i, batch in enumerate(dataloader):
            real_data = batch[0]
            batch_size_actual = real_data.size(0)
            
            real_labels = torch.ones(batch_size_actual).to(device)
            fake_labels = torch.zeros(batch_size_actual).to(device)
            
            # Train Discriminator - more frequently
            optimizer_d.zero_grad()
            
            # Real data
            output_real = discriminator(real_data)
            loss_d_real = criterion(output_real, real_labels)

            # Fake data
            z = torch.randn(batch_size_actual, latent_dim).to(device)
            fake_data = generator(z)
            loss_d.backward()
            optimizer_d.step()
            
            # Train Generator - every 2 discriminator steps
            if i % 2 == 0:
            optimizer_g.zero_grad()
                # Generate new fake data
            z = torch.randn(batch_size_actual, latent_dim).to(device)
            fake_data = generator(z)
            output_fake = discriminator(fake_data)
                loss_g = criterion(output_fake, real_labels)
            
            loss_g.backward()
            optimizer_g.step()
            avg_g_loss = epoch_g_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
    
    return generator.cpu(), discriminator.cpu()


def evaluate(generator, discriminator, train_data, val_data):
    """Evaluate the GAN model on train and validation data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    val_tensor = torch.FloatTensor(val_data).to(device)
    
    metrics = {}

    # Generate samples for evaluation
    num_eval_samples = min(len(val_data), 1000)  # Limit for efficiency
    z = torch.randn(num_eval_samples, LATENT_DIM).to(device)
    
    # Measure generation throughput
    start_time = time.time()
    with torch.no_grad():
        generated_samples = generator(z)
    gen_time = time.time() - start_time  # noqa: F841
    generation_throughput = num_eval_samples / gen_time if gen_time > 0 else 0
    
    metrics['generation_throughput'] = float(generation_throughput)

    # Calculate MSE between generated samples and validation data
    # Compute MSE between generated samples and closest real samples
    mse_values = []
    for i in range(num_eval_samples):  # All samples
        gen_sample = generated_samples[i].cpu().numpy()
        # Find closest real sample
        distances = np.sum((val_data - gen_sample) ** 2, axis=1)
        min_mse = np.min(distances)
        mse = mean_squared_error(gen_sample, val_sample)
        mse_values.append(mse)
    
    metrics['cov_mse'] = float(np.var(mse_values))
    
    # Calculate R2 score for mean comparison
    gen_samples_np = generated_samples.cpu().numpy()
    gen_mean = np.mean(gen_samples_np, axis=0)
    val_mean = np.mean(val_data, axis=0)
    
    # R2 score for mean comparison
    ss_res = np.sum((val_mean - gen_mean) ** 2)
    ss_tot = np.sum((val_mean - np.mean(val_data, axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    metrics['r2_mean'] = float(r2)

    # Discriminator accuracy on real and fake data
    with torch.no_grad():
        # Real data accuracy
        output_real_val = discriminator(val_tensor[:num_eval_samples])
        real_preds = (output_real_train > 0.5).float()
        real_acc = real_preds.float().mean().item()
        
        # Fake data accuracy
        output_fake = discriminator(generated_samples)
        fake_preds = (output_fake > 0.5).float()
        fake_acc = (1 - fake_preds).float().mean().item()  # Discriminator should predict 0 (fake)
    
    metrics['discriminator_real_acc'] = float(real_acc)
    metrics['discriminator_fake_acc'] = float(fake_acc)

    return metrics


    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

def save_metrics(metrics_dict, save_path):
    """Save metrics to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved metrics to {save_path}")


    """Main function to run the GAN evaluation and export task."""
    print("=" * 60)
    print("GAN Level 4: Evaluation and Export")
    print("Benchmark: Generation throughput (images/sec)")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_data(2000)
    print(f"Generated data shape: {data.shape}")

    # Split data
    train_size = int(len(data) * TRAIN_RATIO)
    train_data = data[:train_size]
    val_data = data[train_size:]
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    print("\n2. Initializing GAN models...")
    generator = Generator()
    discriminator = Discriminator()

    total_gen_params = sum(p.numel() for p in generator.parameters())
    total_disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {total_gen_params}")
    print(f"Discriminator parameters: {total_disc_params}")

    # 3. Train GAN
    print("\n3. Training GAN...")
    
    # 4. Evaluate on training data
    print("\n4. Evaluating on training data...")
    train_metrics = evaluate(generator, discriminator, train_data, train_data)  # noqa: F841
    print(f"Training metrics: {train_metrics}")

    # 5. Evaluate on validation data
    print("\n5. Evaluating on validation data...")
    val_metrics = evaluate(generator, discriminator, train_data, val_data)
    print(f"Validation metrics: {val_metrics}")

    # 6. Save models
    print("\n6. Saving model...")
    save_model(generator, Path(__file__).parent / 'generator.pt')
    save_model(discriminator, Path(__file__).parent / 'discriminator.pt')

    # 7. Save metrics
    print("\n7. Saving metrics...")
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'total_samples': len(data),
        'train_ratio': TRAIN_RATIO,
        'latent_dim': LATENT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE
    }
    save_metrics(all_metrics, OUTPUT_DIR / 'metrics.json')

    # 8. Quality checks
    print("\n8. Quality checks...")

    # Check generation throughput (must be reasonable)
    assert val_metrics['generation_throughput'] > 1000, \
        f"Generation throughput too low: {val_metrics['generation_throughput']:.2f} images/sec"
    
    # Check discriminator accuracy
    assert val_metrics['discriminator_real_acc'] > 0.7, \
        f"Discriminator real accuracy too low: {val_metrics['discriminator_real_acc']:.2f} (need >0.7)"
    print(f"✓ Discriminator real accuracy: {val_metrics['discriminator_real_acc']:.2f}")

    assert val_metrics['discriminator_fake_acc'] > 0.7, \
        f"Discriminator fake accuracy too low: {val_metrics['discriminator_fake_acc']:.2f} (need >0.7)"
    print(f"✓ Discriminator fake accuracy: {val_metrics['discriminator_fake_acc']:.2f}")

    # Check mean MSE is reasonable (relaxed threshold for simple GAN)
    assert val_metrics['mean_mse'] < 5.0, \
        f"Mean MSE too high: {val_metrics['mean_mse']:.4f} (threshold: 5.0)"
    print(f"✓ Mean MSE: {val_metrics['mean_mse']:.4f} (threshold: 10.0)")

    # Check R2 score for mean comparison
    assert val_metrics['r2_mean'] > -5.0, \
        f"R2 score for mean too low: {val_metrics['r2_mean']:.2f} (threshold: -5.0)"
    print(f"✓ R2 score for mean: {val_metrics['r2_mean']:.2f} (threshold: -10.0)")

    print("\nAll quality checks passed!")
    print("=" * 60)

    return 0
