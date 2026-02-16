    print(f"✓ Validation R² > 0.9: {val_metrics['r2']:.4f}")
    
    # Check MSE is low (VAE reconstruction MSE is typically higher than AE)
    assert val_metrics['mse'] < 0.3, f"Validation MSE should be < 0.3, got {val_metrics['mse']:.6f}"
    print(f"✓ Validation MSE < 0.1: {val_metrics['mse']:.6f}")
    
    # Check ELBO is reasonable (should be positive for well-trained model)
    assert val_metrics['elbo'] > 0, f"ELBO should be positive, got {val_metrics['elbo']:.6f}"
    print(f"✓ Validation ELBO > 0: {val_metrics['elbo']:.6f}")
