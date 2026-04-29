#!/usr/bin/env python3
"""Test script for data_loader validation features."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    print("="*50)
    print("Testing Data Loader Validation Module")
    print("="*50)
    
    # Test imports
    print("\n1. Testing imports...")
    from data_loader import DataValidator, detect_pii, sanitize_pii, DataLoader, DataValidationError
    print("   OK - All classes imported")
    
    # Test DataValidator
    print("\n2. Testing DataValidator...")
    import pandas as pd
    validator = DataValidator()
    
    # Test with null values exceeding threshold
    df_with_nulls = pd.DataFrame({
        'a': [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
        'b': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    })
    is_valid, report = validator.validate(df_with_nulls, 'regression')
    print(f"   is_valid: {is_valid}")
    print(f"   errors: {report['errors']}")
    assert not is_valid, "Expected validation to fail with nulls"
    print("   OK - Null validation works")
    
    # Test with valid data
    df_valid = pd.DataFrame({
        'a': range(1, 101),
        'b': range(101, 201)
    })
    is_valid, report = validator.validate(df_valid, 'regression')
    print(f"   Valid data is_valid: {is_valid}")
    assert is_valid, "Expected validation to pass with valid data"
    print("   OK - Valid data passes")
    
    # Test PII detection
    print("\n3. Testing PII detection...")
    df_pii = pd.DataFrame({
        'email': ['test@example.com'],
        'phone': ['1234567890'],
        'edad': [25],
        'direccion': ['123 Main St']
    })
    pii_cols = detect_pii(df_pii)
    print(f"   Detected PII: {pii_cols}")
    assert len(pii_cols) >= 3, f"Expected to detect email, phone, direccion, got {pii_cols}"
    print("   OK - PII detection works")
    
    # Test PII sanitization
    print("\n4. Testing PII sanitization...")
    df_sanitized = sanitize_pii(df_pii, pii_cols)
    print(f"   Sanitized sample: {df_sanitized['email'].iloc[0][:12]}...")
    assert df_sanitized['email'].iloc[0] != 'test@example.com', "Email should be hashed"
    print("   OK - PII sanitization works")
    
    # Test DataLoader hash consistency
    print("\n5. Testing hash consistency...")
    loader = DataLoader('dummy.csv', 'logs/baseline.json')
    df1 = pd.DataFrame({'a': [1, 2, 3]})
    df2 = pd.DataFrame({'a': [1, 2, 3]})
    df3 = pd.DataFrame({'a': [4, 5, 6]})
    hash1 = loader.compute_hash(df1)
    hash2 = loader.compute_hash(df2)
    hash3 = loader.compute_hash(df3)
    print(f"   Same data: {hash1 == hash2}")
    print(f"   Different data: {hash1 != hash3}")
    assert hash1 == hash2, "Same data should have same hash"
    assert hash1 != hash3, "Different data should have different hash"
    print("   OK - Hash consistency works")
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED")
    print("="*50)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)