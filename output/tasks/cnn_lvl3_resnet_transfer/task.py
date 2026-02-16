        y_train_device = y_train.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train_device)
        loss = criterion(outputs, y_train_device)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        model.eval()
        X_val_device = X_val.to(device)
        y_val_device = y_val.to(device)
        
        with torch.no_grad():
            val_outputs = model(X_val_device)
            val_losses.append(criterion(val_outputs, y_val_device).item())  # noqa: F841
        if use_scheduler:
            scheduler.step(val_loss)
        
    model.eval()
    
    with torch.no_grad():
        X_device = X.to(device)
        predictions = model(X_device).cpu().numpy().flatten()
    
    # Calculate metrics
        generator=torch.Generator().manual_seed(42)  # noqa: F841
    )
    
    # Create data loaders - use proper collate function for tuples
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1].unsqueeze(0) if isinstance(item[1], torch.Tensor) else torch.tensor([item[1]]) for item in batch])
        return images, targets.squeeze()
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Extract tensors for evaluation
    X_train, y_train = next(iter(train_loader))
    X_val, y_val = next(iter(val_loader))
    
    # Ensure proper dtypes
    X_train, X_val = X_train.float(), X_val.float()
    y_train, y_val = y_train.float().view(-1), y_val.float().view(-1)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, dtype: {X_train.dtype}")
    print("\n4. Performance Comparison:")
    val_improvement = scratch_val_metrics['mse'] - transfer_val_metrics['mse']
    print(f"  Validation MSE improvement: {val_improvement:.6f}")
    print(f"  Transfer learning is better: {transfer_val_metrics['mse'] <= scratch_val_metrics['mse']}")
    
    # 5. Save metrics
    print("\n5. Saving metrics...")
    print("\n7. Quality checks...")
    
    # Check that transfer learning beats from-scratch
    assert transfer_val_metrics['mse'] < scratch_val_metrics['mse'], \
        f"Transfer learning should beat from-scratch: {transfer_val_metrics['mse']:.6f} >= {scratch_val_metrics['mse']:.6f}"
    print(f"✓ Transfer learning beats from-scratch (MSE: {transfer_val_metrics['mse']:.6f} < {scratch_val_metrics['mse']:.6f})")
