#!/usr/bin/env python3
"""
Random Forest Ensemble - Level 2
Task: Implement evaluate() returning MSE, R2, and OOB score
Implementation using sklearn's RandomForestRegressor with OOB scoring
"""

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        oob_score=True,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model, X, y):
    """
    Evaluate the model and return metrics.
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"True parameters: bias={true_bias:.4f}, weights={true_weights}")
    
    # 2. Split data (use stratified-like split for regression)
    print("\n2. Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = split_data(X, y, train_ratio=0.8)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    # 3. Train model
    print("\n3. Training Random Forest Regressor...")
    model = train_model(X_train, y_train, n_estimators=200, max_depth=8, random_state=42)
    print(f"Model trained with {model.n_estimators} estimators, max_depth={model.max_depth}")
    
    # 4. Evaluate on training data
    print("\n4. Evaluating on training data...")
    train_metrics = evaluate(model, X_train, y_train)
    print(f"Training MSE: {train_metrics['mse']:.6f}")
    print(f"Training R2: {train_metrics['r2']:.6f} (OOB: {train_metrics['oob_score']:.6f})")
    print(f"Training OOB Score: {train_metrics['oob_score']:.6f}")
    
    # 5. Evaluate on validation data
    print(f"Validation R2: {val_metrics['r2']:.6f}")
    print(f"Validation OOB Score: {val_metrics['oob_score']:.6f}")
    
    # 6. Quality checks (OOB should be close to validation R2)
    print("\n6. Quality checks...")
    
    # Check OOB score is close to validation R2 (difference < 0.05) - PRIMARY REQUIREMENT
    oob_val_diff = abs(val_metrics['oob_score'] - val_metrics['r2'])
    assert oob_val_diff < 0.05, f"OOB score should be close to validation R2, diff={oob_val_diff:.4f} >= 0.05"
    print(f"✓ OOB score close to validation R2: diff={oob_val_diff:.4f} < 0.05")
    
    # Check R² score is reasonable (> 0.5)
    assert val_metrics['r2'] > 0.5, f"Validation R2 should be > 0.5, got {val_metrics['r2']:.4f}"
    print(f"✓ Validation R2 > 0.5: {val_metrics['r2']:.4f}")
    
    # Check MSE is reasonable (< 2.0)
    assert val_metrics['mse'] < 2.0, f"Validation MSE should be < 2.0, got {val_metrics['mse']:.4f}"
    print(f"✓ Validation MSE < 2.0: {val_metrics['mse']:.4f}")
    
    # Check training R2 is not too much higher than validation (no severe overfitting - allow some)
    r2_diff = train_metrics['r2'] - val_metrics['r2']
    assert r2_diff < 0.1, f"Training R2 should not be much higher than validation R2, diff={r2_diff:.4f}"
    print(f"✓ No severe overfitting: train R2 - val R2 = {r2_diff:.4f} < 0.1")
