"""Transformer Encoder Classifier for sequence classification."""

import numpy as np
import warnings
import os
import torch
import torch.nn as nn
    """Train the transformer encoder classifier."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            # Ensure shapes match: flatten both to 1D
            loss = criterion(outputs.view(-1), batch_y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss)
            
            train_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    
    return train_losses, val_losses


def evaluate(model, X, y, device='cpu'):
    """Evaluate the model and return metrics."""
    model.eval()
    model = model.to(device)
    
    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.FloatTensor(y).to(device).view(-1)
    
    # Get predictions
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = model(X_t).cpu().numpy().flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    # For classification-like evaluation on regression task
    target_mean = np.mean(y)
    target_binary = (y > target_mean).astype(int)
    pred_binary = (predictions.flatten() > target_mean).astype(int)
    accuracy = np.mean(target_binary == pred_binary)
    
    return {
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data into train/val sets
    print("\n2. Splitting data into train/val sets...")
    X_train, X_val, y_train, y_val = split_data(X, y.reshape(-1), train_ratio=0.8)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create dataloaders
    # Train model
    print("\n5. Training model...")
    losses, val_losses = train(
        model, train_loader, val_loader,  # noqa: F841
        learning_rate=0.01, epochs=100, device=device
    )
    
    # Evaluate on training data
    print("\n10. Quality checks...")
    
    # Check loss decreases
    assert losses[-1] < losses[0] * 0.9, f"Training loss should decrease significantly: {losses[0]:.6f} -> {losses[-1]:.6f}"
    print(f"✓ Loss decreasing significantly: {losses[0]:.6f} -> {losses[-1]:.6f}")
    
    # Check R² score (should be positive and reasonably high)
    assert val_metrics['r2'] > 0.3, f"Validation R² should be > 0.3: {val_metrics['r2']:.4f}"
    print(f"✓ Validation R² > 0.3: {val_metrics['r2']:.4f}")
    
    # Check MSE is reasonable
    assert val_metrics['mse'] < 0.5, f"Validation MSE should be < 0.5: {val_metrics['mse']:.4f}"
    print(f"✓ Validation MSE < 0.5: {val_metrics['mse']:.4f}")
    
    # Check accuracy is better than random
    assert val_metrics['accuracy'] > 0.55, f"Validation accuracy should be > 0.55: {val_metrics['accuracy']:.4f}"
    print(f"✓ Validation accuracy > 0.55: {val_metrics['accuracy']:.4f}")
    
    print("\nAll quality checks passed!")
    print("=" * 70)
