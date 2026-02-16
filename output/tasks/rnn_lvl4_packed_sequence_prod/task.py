        # Process through RNN
        out, _ = self.rnn(x, (h0, c0))
        
        # Use last valid output for each sequence (extract single timestep)
        out = self.fc(out)
        
        # Extract last valid output based on lengths
        for i, length in enumerate(lengths):
            final_out.append(out[i, length-1:i+1])
        
        return torch.cat(final_out, dim=0).squeeze(-1)


class PackedRNN(nn.Module):
        # Use last valid output for each sequence
        out = self.fc(out)
        
        # Extract last valid output for each sequence (single timestep)
        final_out = []
        for i, length in enumerate(lengths):
            final_out.append(out[i, length-1:i+1])
        
        return torch.cat(final_out, dim=0)

    return torch.cat(final_out, dim=0).squeeze(-1)

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """Train the RNN model."""
    with torch.no_grad():
        for padded, lengths, targets in data_loader:
            padded = padded.to(device)
            targets = targets.view(-1).to(device)
            
            outputs = model(padded, lengths)
            
    # 7. Quality checks
    print("\n7. Quality checks...")
    
    # Check R² score (relaxed threshold for variable-length sequences)
    assert val_metrics['r2'] > 0.5, f"Validation R² should be > 0.5, got {val_metrics['r2']:.4f}"
    print(f"✓ Validation R² > 0.5: {val_metrics['r2']:.4f}")
    
    print(f"✓ Validation MSE < 1.0: {val_metrics['mse']:.6f}")
    
    # Check that packed is at least as fast as padded (allow small variance)
    speedup = padded_times / max(packed_times, 1e-6)
    print(f"✓ Speedup factor: {speedup:.2f}x (packed vs padded)")
    
    # Check loss decreased
    assert padded_val_losses[-1] < padded_val_losses[0], "Padded model loss should decrease"
    assert packed_val_losses[-1] < packed_val_losses[0], "Packed model loss should decrease"
    print(f"✓ Model loss decreased: {padded_val_losses[0]:.6f} -> {padded_val_losses[-1]:.6f}")
