    # 7. Evaluate on training data
    print("\n7. Evaluating on training data...")
    train_metrics = evaluate(model, train_loader)
    print(f"Training Metrics:")
    print(f"  MSE: {train_metrics['mse']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  R2: {train_metrics['r2']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall: {train_metrics['recall']:.4f}")
    print(f"  F1: {train_metrics['f1']:.4f}")
    print(f"  Avg Loss: {train_metrics['avg_loss']:.4f}")
    
    # 8. Evaluating on validation data
    print("\n8. Evaluating on validation data...")
    val_metrics = evaluate(model, val_loader)
    print(f"Validation Metrics:")
    print(f"  F1: {val_metrics['f1']:.4f}")
    print(f"  Avg Loss: {val_metrics['avg_loss']:.4f}")
    
    # 9. Test padding/masking correctness
    print("\n9. Testing padding/masking correctness...")
    padding_accuracy = test_padding_masking(model, max_len=50)
    print(f"Padding/masking test accuracy: {padding_accuracy:.4f}")
    # 10. Generate visualizations
    print("\n10. Generating visualizations...")
    visualize_results(train_losses, val_losses, val_accuracies, save_dir='.')

    # 11. Quality checks
    print("\n11. Quality checks...")
    
    # Check R2 score (should be positive and reasonably high)
    assert val_metrics['r2'] > 0.5, f"R2 score should be > 0.5, got {val_metrics['r2']:.4f}"
    print(f"✓ R2 score: {val_metrics['r2']:.4f} > 0.5")

    # Check accuracy (should be > 70%)
    assert val_metrics['accuracy'] > 0.70, f"Accuracy should be > 0.70, got {val_metrics['accuracy']:.4f}"
    print(f"✓ Accuracy: {val_metrics['accuracy']:.4f} > 0.70")
    # Check padding/masking test
    assert padding_accuracy >= 0.80, f"Padding/masking test should be > 0.80, got {padding_accuracy:.4f}"
    print(f"✓ Padding/masking test: {padding_accuracy:.4f} > 0.80")

    print("\nAll quality checks passed!")
    print("=" * 60)
    


if __name__ == '__main__':
    main()
