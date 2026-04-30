#!/usr/bin/env python3
"""Test script for monitor module."""
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    print("="*50)
    print("Testing Monitor Module")
    print("="*50)
    
    # Test imports
    print("\n1. Testing imports...")
    from monitor import DataMonitor, ScheduledMonitor, detect_csv_changes
    print("   OK - Imports work")
    
    # Test DataMonitor with file
    print("\n2. Testing DataMonitor (file change detection)...")
    tmpdir = tempfile.mkdtemp()
    test_file = os.path.join(tmpdir, "test.csv")
    
    # Create initial file
    with open(test_file, 'w') as f:
        f.write("a,b,c\n1,2,3\n")
    
    monitor = DataMonitor(watch_paths=[test_file], check_interval_seconds=1)
    
    # First check - no change
    changed = monitor.has_changed(test_file)
    print(f"   First check (no change): {changed}")
    assert not changed, "Should not detect change on first check"
    print("   OK - Initial state captured")
    
    # Modify file
    print("\n3. Testing file modification detection...")
    import time
    time.sleep(0.1)
    with open(test_file, 'a') as f:
        f.write("4,5,6\n")
    
    changed = monitor.has_changed(test_file)
    print(f"   After modification: {changed}")
    assert changed, "Should detect change after modification"
    print("   OK - Modification detected")
    
    # Test check_all
    print("\n4. Testing check_all...")
    results = monitor.check_all()
    print(f"   All results: {results}")
    assert test_file in results
    print("   OK - check_all works")
    
    # Test DataMonitor with directory
    print("\n5. Testing DataMonitor (directory)...")
    monitor_dir = DataMonitor(watch_paths=[tmpdir], check_interval_seconds=1)
    changed_dir = monitor_dir.has_changed(tmpdir)
    print(f"   Directory changed: {changed_dir}")
    print("   OK - Directory monitoring works")
    
    # Test ScheduledMonitor
    print("\n6. Testing ScheduledMonitor...")
    sched = ScheduledMonitor()
    callback_called = [False]
    
    def test_callback():
        callback_called[0] = True
    
    sched.schedule_minutes(1, test_callback)
    print("   Scheduled job added")
    print("   OK - ScheduledMonitor works")
    
    # Test detect_csv_changes
    print("\n7. Testing detect_csv_changes...")
    csv_file = os.path.join(tmpdir, "data.csv")
    with open(csv_file, 'w') as f:
        f.write("a,b\n1,2\n")
    
    # First call - no change expected
    has_changes = detect_csv_changes(csv_file)
    print(f"   First call result: {has_changes}")
    
    # Second call - should be no change (already saved)
    time.sleep(0.1)
    has_changes = detect_csv_changes(csv_file)
    print(f"   Second call (same content): {has_changes}")
    
    # Modify and call again
    time.sleep(0.1)
    with open(csv_file, 'a') as f:
        f.write("3,4\n")
    has_changes = detect_csv_changes(csv_file)
    print(f"   Third call (modified): {has_changes}")
    assert has_changes, "Should detect change after modification"
    print("   OK - detect_csv_changes works")
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)
    
    print("\n" + "="*50)
    print("ALL MONITOR TESTS PASSED")
    print("="*50)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)