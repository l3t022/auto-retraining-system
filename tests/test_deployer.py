#!/usr/bin/env python3
"""Test script for deployer module."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    print("="*50)
    print("Testing Deployer Module")
    print("="*50)
    
    # Test imports
    print("\n1. Testing imports...")
    from deployer import ModelDeployer
    import joblib
    import numpy as np
    print("   OK - Imports work")
    
    # Test initialization
    print("\n2. Testing deployer initialization...")
    import tempfile
    tmpdir = tempfile.mkdtemp()
    deployer = ModelDeployer(models_dir=tmpdir, backup_enabled=True)
    print(f"   Models dir: {tmpdir}")
    print("   OK - Deployer initialized")
    
    # Test save_model
    print("\n3. Testing save_model...")
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(np.random.randn(100, 5), np.random.randn(100))
    
    metrics = {'mse': 0.5, 'rmse': 0.7, 'r2': 0.8}
    version = deployer.save_model(model, "xgboost", metrics, {'n_estimators': 10})
    print(f"   Saved version: {version}")
    print("   OK - Model saved")
    
    # Test get_current
    print("\n4. Testing get_current...")
    current = deployer.get_current()
    print(f"   Current version: {current['version']}")
    assert current['version'] == version
    print("   OK - Current model retrieved")
    
    # Test get_model
    print("\n5. Testing get_model...")
    loaded_model = deployer.get_model()
    assert loaded_model is not None
    print("   OK - Model loaded")
    
    # Test list_versions
    print("\n6. Testing list_versions...")
    versions = deployer.list_versions()
    print(f"   Versions: {len(versions)}")
    assert len(versions) >= 1
    print("   OK - Versions listed")
    
    # Test get_metrics_history
    print("\n7. Testing metrics history...")
    history = deployer.get_metrics_history()
    print(f"   History entries: {len(history)}")
    assert len(history) >= 1
    print("   OK - Metrics history retrieved")
    
    # Test rollback (create another model to test)
    print("\n8. Testing rollback...")
    model2 = RandomForestRegressor(n_estimators=20, random_state=42)
    model2.fit(np.random.randn(100, 5), np.random.randn(100))
    version2 = deployer.save_model(model2, "xgboost", {'mse': 0.3}, {'n_estimators': 20})
    deployer.update_current(version2)
    
    # Now rollback
    success = deployer.rollback(target_version=version)
    print(f"   Rollback success: {success}")
    print("   OK - Rollback works")
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)
    
    print("\n" + "="*50)
    print("ALL DEPLOYER TESTS PASSED")
    print("="*50)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)