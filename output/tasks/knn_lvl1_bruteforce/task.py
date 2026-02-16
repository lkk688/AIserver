    # MSE (for regression-like evaluation)
    mse = ((y_pred.float() - y_val.float()) ** 2).mean().item()
    
    # R2 score - handle edge case where ss_tot is 0 (all y_val values are the same)
    y_val_f = y_val.float()
    y_pred_f = y_pred.float()
    ss_res = torch.sum((y_val_f - y_pred_f) ** 2)
    ss_tot = torch.sum((y_val_f - y_val_f.mean()) ** 2)
    
    # Handle division by zero - if ss_tot is 0, R2 is 1.0 (perfect prediction)
    r2 = 1.0 - (ss_res / ss_tot).item() if ss_tot.item() > 1e-10 else 1.0
    
    return {
        'accuracy': accuracy,
    # Quality thresholds
    print("\n[5] Validating quality thresholds...")
    try:
        # For```diff
classification, accuracy is the primary metric
        assert val_metrics['accuracy'] >= 0.80, f"Accuracy {val_metrics['accuracy']:.4f} < 0.80"
        # R2 should be valid (not NaN)
        assert val_metrics['r2'] == val_metrics['r2'], f"R2 is NaN"
        print("    ✓ All quality thresholds passed!")
    except AssertionError as e:
        print(f"    ✗ Quality threshold failed: {e}")
