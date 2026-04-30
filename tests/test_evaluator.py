#!/usr/bin/env python3
"""Test script for evaluator module."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    print("="*50)
    print("Testing Evaluator Module")
    print("="*50)
    
    # Test imports
    print("\n1. Testing imports...")
    from evaluator import ModelEvaluator
    print("   OK - Imports work")
    
    # Test with mock model
    print("\n2. Testing ModelEvaluator initialization...")
    import tempfile
    import pandas as pd
    import numpy as np
    
    tmpdir = tempfile.mkdtemp()
    model_path = os.path.join(tmpdir, "model.joblib")
    baseline_path = os.path.join(tmpdir, "baseline.json")
    history_path = os.path.join(tmpdir, "history.json")
    
    evaluator = ModelEvaluator(
        model_path=model_path,
        baseline_path=baseline_path,
        metrics_history_path=history_path,
        task_type="regression",
        mse_threshold=0.05,
    )
    print("   OK - Evaluator initialized")
    
    # Test baseline loading (no baseline exists yet)
    print("\n3. Testing baseline loading (no baseline)...")
    assert evaluator.baseline_metrics == {}, "Should be empty dict"
    print("   OK - Empty baseline as expected")
    
    # Test metrics registration
    print("\n4. Testing metrics registration...")
    test_metrics = {'mse': 0.5, 'rmse': 0.7, 'mae': 0.3, 'r2': 0.8}
    evaluator.register_metrics(test_metrics, set_as_baseline=True)
    print(f"   Registered: {test_metrics}")
    
    # Verify saved
    evaluator2 = ModelEvaluator(
        model_path=model_path,
        baseline_path=baseline_path,
        metrics_history_path=history_path,
        task_type="regression",
    )
    assert evaluator2.baseline_metrics['mse'] == 0.5, "Should load saved metrics"
    print("   OK - Metrics saved and loaded")
    
    # Test should_retrain logic
    print("\n5. Testing should_retrain logic...")
    
    # Case 1: MSE below threshold (should NOT retrain)
    should_retrain, reason = evaluator.should_retrain({'mse': 0.51})
    print(f"   MSE 0.51 vs baseline 0.5: retrain={should_retrain}, reason='{reason}'")
    assert not should_retrain, "Should NOT retrain for 2% increase"
    print("   OK - No retrain for small increase")
    
    # Case 2: MSE above threshold (should retrain)
    should_retrain, reason = evaluator.should_retrain({'mse': 0.6})
    print(f"   MSE 0.6 vs baseline 0.5: retrain={should_retrain}, reason='{reason}'")
    assert should_retrain, "Should retrain for 20% increase"
    print("   OK - Retrain triggered for large increase")
    
    # Test get_comparison
    print("\n6. Testing metrics comparison...")
    comparison = evaluator.get_comparison({'mse': 0.4, 'rmse': 0.6})
    print(f"   Comparison: {comparison}")
    assert comparison['has_baseline'] == True
    assert comparison['mse']['old'] == 0.5
    assert comparison['mse']['new'] == 0.4
    assert abs(comparison['mse']['pct_change'] - (-20.0)) < 0.01  # ~20% improvement
    print("   OK - Comparison works")
    
    # Test classification task type
    print("\n7. Testing classification evaluation...")
    eval_class = ModelEvaluator(
        model_path=model_path,
        baseline_path=baseline_path.replace('.json', '_cls.json'),
        metrics_history_path=history_path.replace('.json', '_cls.json'),
        task_type="classification",
        accuracy_threshold=0.03,
    )
    test_metrics_cls = {'accuracy': 0.9, 'precision': 0.85, 'recall': 0.88, 'f1': 0.86}
    eval_class.register_metrics(test_metrics_cls, set_as_baseline=True)
    
    # Should NOT retrain for small decrease
    should_retrain, reason = eval_class.should_retrain({'accuracy': 0.88})
    print(f"   Accuracy 0.88 vs baseline 0.9: retrain={should_retrain}, reason='{reason}'")
    assert not should_retrain, "Should NOT retrain for ~2% decrease"
    print("   OK - No retrain for small accuracy drop")
    
    # Should retrain for large decrease
    should_retrain, reason = eval_class.should_retrain({'accuracy': 0.85})
    print(f"   Accuracy 0.85 vs baseline 0.9: retrain={should_retrain}, reason='{reason}'")
    assert should_retrain, "Should retrain for ~5% decrease"
    print("   OK - Retrain triggered for accuracy drop")
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)
    
    print("\n" + "="*50)
    print("ALL EVALUATOR TESTS PASSED")
    print("="*50)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)