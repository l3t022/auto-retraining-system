#!/usr/bin/env python3
"""Test script for model_selector module."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    print("="*50)
    print("Testing Model Selector Module")
    print("="*50)
    
    # Test imports
    print("\n1. Testing imports...")
    from model_selector import ModelSelector
    print("   OK - Imports work")
    
    # Test with regression
    print("\n2. Testing ModelSelector (regression)...")
    selector = ModelSelector(models_dir=".", task_type="regression")
    
    # Add candidates
    selector.add_candidate("model_a.joblib", {'mse': 0.5, 'rmse': 0.7, 'r2': 0.8})
    selector.add_candidate("model_b.joblib", {'mse': 0.3, 'rmse': 0.5, 'r2': 0.9})
    selector.add_candidate("model_c.joblib", {'mse': 0.4, 'rmse': 0.6, 'r2': 0.85})
    
    best = selector.select_best()
    print(f"   Best model: {best['model_path']}")
    print(f"   Best MSE: {best['metrics']['mse']}")
    assert best['model_path'] == 'model_b.joblib', "Should select lowest MSE"
    assert best['metrics']['mse'] == 0.3
    print("   OK - Best selection works")
    
    # Test compare_all
    print("\n3. Testing compare_all (ranking)...")
    ranked = selector.compare_all()
    print(f"   Rank 1: {ranked[0]['model_path']} (MSE {ranked[0]['metrics']['mse']})")
    print(f"   Rank 2: {ranked[1]['model_path']} (MSE {ranked[1]['metrics']['mse']})")
    print(f"   Rank 3: {ranked[2]['model_path']} (MSE {ranked[2]['metrics']['mse']})")
    assert ranked[0]['rank'] == 1
    assert ranked[1]['rank'] == 2
    assert ranked[2]['rank'] == 3
    print("   OK - Ranking works")
    
    # Test should_replace
    print("\n4. Testing should_replace (regression)...")
    should_replace, reason = selector.should_replace(
        current_model_path="model_a.joblib",
        current_metrics={'mse': 0.5},
        new_model_path="model_b.joblib",
        new_metrics={'mse': 0.3},
        improvement_threshold=0.01,  # 1% threshold
    )
    print(f"   Should replace: {should_replace}, reason='{reason}'")
    assert should_replace, "Should replace (20% improvement)"
    print("   OK - Replacement decision works")
    
    # Test should NOT replace (insufficient improvement)
    should_replace, reason = selector.should_replace(
        current_model_path="model_a.joblib",
        current_metrics={'mse': 0.5},
        new_model_path="model_c.joblib",
        new_metrics={'mse': 0.49},  # only 2% improvement
        improvement_threshold=0.05,  # requires 5%
    )
    print(f"   Should NOT replace (2%): {should_replace}, reason='{reason}'")
    assert not should_replace, "Should NOT replace (below threshold)"
    print("   OK - No replacement when below threshold")
    
    # Test improvement report
    print("\n5. Testing improvement report...")
    report = selector.get_improvement_report(
        current_metrics={'mse': 0.5, 'rmse': 0.7},
        new_metrics={'mse': 0.3, 'rmse': 0.5},
    )
    print(f"   MSE: old={report['mse']['old']}, new={report['mse']['new']}, change={report['mse']['pct_change']:.1f}%")
    print(f"   RMSE: old={report['rmse']['old']}, new={report['rmse']['new']}, change={report['rmse']['pct_change']:.1f}%")
    assert report['mse']['improved'] == True
    assert report['rmse']['improved'] == True
    print("   OK - Improvement report works")
    
    # Test classification
    print("\n6. Testing classification selection...")
    selector_cls = ModelSelector(models_dir=".", task_type="classification")
    
    selector_cls.add_candidate("clf_a.joblib", {'accuracy': 0.85, 'precision': 0.8, 'recall': 0.82})
    selector_cls.add_candidate("clf_b.joblib", {'accuracy': 0.92, 'precision': 0.9, 'recall': 0.88})
    
    best_cls = selector_cls.select_best()
    print(f"   Best classifier: {best_cls['model_path']}")
    print(f"   Best accuracy: {best_cls['metrics']['accuracy']}")
    assert best_cls['model_path'] == 'clf_b.joblib', "Should select highest accuracy"
    print("   OK - Classification selection works")
    
    # Test should_replace for classification
    should_replace, reason = selector_cls.should_replace(
        current_model_path="clf_a.joblib",
        current_metrics={'accuracy': 0.85},
        new_model_path="clf_b.joblib",
        new_metrics={'accuracy': 0.92},
        improvement_threshold=0.01,
    )
    print(f"   Should replace: {should_replace}, reason='{reason}'")
    assert should_replace, "Should replace for accuracy improvement"
    print("   OK - Classification replacement works")
    
    print("\n" + "="*50)
    print("ALL MODEL SELECTOR TESTS PASSED")
    print("="*50)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)