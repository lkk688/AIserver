#!/usr/bin/env python3
"""Denoising Autoencoder for MNIST - Level 2"""

import os
    """Simple denoising autoencoder for MNIST."""
    
    def __init__(self, input_size=784, hidden_size=128):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(  # noqa: C400
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
    """Add Gaussian noise to images."""
    noisy_images = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy_images, 0.0, 1.0)
    

def load_mnist_data():
    """Load MNIST data and prepare noisy/clean pairs."""
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # noqa: S108
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Extract data
    X_test_clean = X_test_flat
    
    # Add noise
    X_train_noisy = add_noise(X_train_clean, noise_factor=0.25)
    X_val_noisy = add_noise(X_val_clean, noise_factor=0.3)
    X_test_noisy = add_noise(X_test_clean, noise_factor=0.3)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = TensorDataset(X_train_noisy, X_train_clean)  # noqa: PLW2901
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_losses = []
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
            val_loss = criterion(val_outputs, X_val_clean).item()
            val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return train_losses, val_losses
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        reconstructions = model(X_noisy)  # noqa: PLW2901
        mse_loss = criterion(reconstructions, X_clean).item()
        
        # Calculate per-pixel metrics
        r2_per_pixel = r2_score(X_clean.cpu().numpy().flatten(), 
                                reconstructions.cpu().numpy().flatten())
        
        # Calculate per-image metrics  # noqa: PLW2901
        mse_per_image = mean_squared_error(X_clean.cpu().numpy(), 
                                           reconstructions.cpu().numpy(), 
                                           multioutput='raw_values')
        'clean_images': X_clean.cpu().numpy(),
        'noisy_images': X_noisy.cpu().numpy()
    }
    

def visualize_results(X_noisy, X_reconstructed, X_clean, save_dir='.', n_images=10):
    """Save grid of noisy vs reconstructed vs clean images."""
    fig, axes = plt.subplots(3, n_images, figsize=(2*n_images, 6))
    
    for i in range(n_images):
        # Noisy images  # noqa: PLW2901
        axes[0, i].imshow(X_noisy[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
        axes[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', rotation=0, labelpad=20)  # noqa: PLW2901
       
        # Clean images
        axes[2, i].imshow(X_clean[i].reshape(28, 28), cmap='gray')
    plt.close()
    print(f"Saved visualization to {save_path}")
    

def save_metrics(metrics, save_dir='.'):
    """Save metrics to JSON file."""
    
    # Convert numpy arrays to lists for JSON serialization
    metrics_serializable = {}
    for k, v in metrics.items():  # noqa: PLW2901
        if isinstance(v, np.ndarray):
            metrics_serializable[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
        else:
            metrics_serializable[k] = v
    
    with open(save_path, 'w') as f:  # noqa: PLW2901
        json.dump(metrics_serializable, f, indent=2)
    print(f"Saved metrics to {save_path}")
    

def main():  # noqa: C901
    """Main function to run the denoising autoencoder task."""
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading MNIST data...")  # noqa: PLW2901
    (X_train_noisy, X_train_clean), (X_val_noisy, X_val_clean), (X_test_noisy, X_test_clean) = load_mnist_data()
    print(f"Training data: {X_train_noisy.shape}")
    print(f"Validation data: {X_val_noisy.shape}")
    print(f"Test data: {X_test_noisy.shape}")
    
    # 2. Initialize model
    print("\n2. Initializing model...")  # noqa: PLW2901
    model = DenoisingAutoencoder(input_size=784, hidden_size=256)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("\n3. Training model...")
    train_losses, val_losses = train(
        model, X_train_noisy, X_train_clean,
        X_val_noisy, X_val_clean,  # noqa: PLW2901
        learning_rate=0.001, epochs=100, batch_size=256
    )
    
    # 4. Evaluate on training data
    print(f"Training R² (per pixel): {train_metrics['r2_per_pixel']:.6f}")
    
    # 5. Evaluate on validation data
    print("\n5. Evaluating on validation data...")  # noqa: PLW2901
    val_metrics = evaluate(model, X_val_noisy, X_val_clean)
    print(f"Validation MSE (per pixel): {val_metrics['mse_per_pixel']:.6f}")
    print(f"Validation R² (per pixel): {val_metrics['r2_per_pixel']:.6f}")
    print("\n6. Generating visualizations...")
    visualize_results(
        val_metrics['noisy_images'][:10],
        val_metrics['reconstructions'][:10],  # noqa: PLW2901
        val_metrics['clean_images'][:10],  # noqa: PLW2901
        save_dir='.'
    )
    
    print("\n7. Saving metrics...")
    all_metrics = {
        'train': {
            'mse_per_pixel': train_metrics['mse_per_pixel'],  # noqa: PLW2901
            'r2_per_pixel': train_metrics['r2_per_pixel']  # noqa: PLW2901
        },
        'validation': {
    save_metrics(all_metrics, save_dir='.')
    
    # 8. Quality checks
    print("\n8. Quality checks...")  # noqa: PLW2901
    
    # Check R² score is good (denoising should be better than random)
    assert val_metrics['r2_per_pixel'] > 0.85, f"R² score should be > 0.85, got {val_metrics['r2_per_pixel']:.4f}"
    print(f"✓ R² score is excellent: {val_metrics['r2_per_pixel']:.4f} (> 0.85)")
    
    # Check MSE is reasonable
    assert val_metrics['mse_per_pixel'] < 0.02, f"MSE should be < 0.02, got {val_metrics['mse_per_pixel']:.6f}"
    
    # Check loss decreased
    assert train_losses[-1] < train_losses[0], "Training loss should decrease"
    print(f"✓ Training loss decreased: {train_losses[0]:.6f} -> {train_losses[-1]:.6f}")  # noqa: PLW2901
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    return 0


if __name__ == '__main__':  # noqa: PLW2901
    sys.exit(main())
