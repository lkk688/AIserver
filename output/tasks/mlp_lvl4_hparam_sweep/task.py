                    # Record result
                    config = {
                        'learning_rate': lr,
                        'hidden_sizes': list(hidden_sizes),
                        'weight_decay': weight_decay,
                        'batch_size': batch_size
                    }
                    
                    sweep_results.append({
                        'config': config,
                        'metrics': val_metrics,
                        'val_loss': val_loss
                    })
    return 0  # Success (exit code 0)


if __name__ == '__main__':
    exit(main())
