#!/usr/bin/env python3
"""Test script for trainer module."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    print("="*50)
    print("Testing Trainer Module")
    print("="*50)
    
    # Test imports
    print("\n1. Testing imports...")
    from trainer import ModelTrainer
    print("   OK - Imports work")
    
    # Create sample data
    print("\n2. Creating sample data...")
    import numpy as np
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(200)*0.1
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    
    # Test XGBoost trainer
    print("\n3. Testing XGBoost trainer (regression)...")
    trainer = ModelTrainer(
        task_type="regression",
        model_type="xgboost",
        n_trials=3,  # Quick test
        timeout=60,  # 1 minute max
        cv_folds=3,
        random_state=42,
    )
    
    model, params, score = trainer.train(X, y)
    print(f"   Best params: {params}")
    print(f"   Best score (MSE): {score:.4f}")
    assert model is not None
    assert params is not None
    print("   OK - XGBoost training works")
    
    # Test study results
    print("\n4. Testing study results...")
    results = trainer.get_study_results()
    print(f"   n_trials: {results['n_trials']}")
    assert results['n_trials'] >= 3
    print("   OK - Study results work")
    
    # Test LightGBM trainer
    print("\n5. Testing LightGBM trainer...")
    trainer_lgbm = ModelTrainer(
        task_type="regression",
        model_type="lightgbm",
        n_trials=3,
        timeout=60,
        cv_folds=3,
        random_state=42,
    )
    model_lgbm, params_lgbm, score_lgbm = trainer_lgbm.train(X, y)
    print(f"   LightGBM score (MSE): {score_lgbm:.4f}")
    print("   OK - LightGBM training works")
    
    # Test train_single (without search)
    print("\n6. Testing train_single...")
    fixed_params = {
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
    }
    model_single = trainer.train_single(X, y, fixed_params)
    print("   OK - Single training works")
    
    # Test classification task
    print("\n7. Testing classification...")
    y_cls = (y > 0).astype(int)
    trainer_cls = ModelTrainer(
        task_type="classification",
        model_type="xgboost",
        n_trials=3,
        timeout=60,
        cv_folds=3,
        random_state=42,
    )
    model_cls, params_cls, score_cls = trainer_cls.train(X, y_cls)
    print(f"   Classification score (accuracy): {score_cls:.4f}")
    print("   OK - Classification training works")
    
    print("\n" + "="*50)
    print("ALL TRAINER TESTS PASSED")
    print("="*50)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)