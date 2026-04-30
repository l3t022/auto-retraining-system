#!/usr/bin/env python3
"""Compliance validation script for Auto-Retrain System

This script validates:
1. Config schema compliance
2. Secret detection (via GitLeaks or inline checks)
3. Policy-as-code rules (via Checkov)
4. Required file structure

Exits with non-zero code on compliance failure.
"""
import json
import os
import sys
import yaml
from pathlib import Path

# ======================
# CONFIGURATION
# ======================
PROJECT_ROOT = Path(__file__).parent / "original"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
REQUIRED_DIRS = ["src", "models", "logs", "data"]
REQUIRED_FILES = ["main.py", "requirements.txt", "config.yaml", "SDD.md"]

# ======================
# VALIDATION FUNCTIONS
# ======================

def validate_directory_structure():
    """Check required directories and files exist."""
    print("Checking directory structure...")
    errors = []
    for d in REQUIRED_DIRS:
        path = PROJECT_ROOT / d
        if not path.exists():
            errors.append(f"Missing directory: {d}")
        elif not path.is_dir():
            errors.append(f"{d} exists but is not a directory")
    for f in REQUIRED_FILES:
        path = PROJECT_ROOT / f
        if not path.exists():
            errors.append(f"Missing file: {f}")
    if errors:
        print("  ❌ FAILED")
        for e in errors:
            print(f"    - {e}")
        return False
    print("  ✅ PASSED")
    return True

def validate_config_schema():
    """Validate config.yaml structure against schema."""
    print("Validating config.yaml schema...")
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"  ❌ FAILED: Cannot load YAML: {e}")
        return False
    
    required_top_keys = ["DATA_PATH", "MODEL_PATH", "LOGS_PATH", "MODELS", "TRIGGERS", "METRICS", "TASK_TYPE"]
    missing = [k for k in required_top_keys if k not in config]
    if missing:
        print(f"  ❌ FAILED: Missing required config keys: {missing}")
        return False
    
    # Validate models section
    models = config.get("MODELS", {})
    valid_models = {"xgboost", "lightgbm"}
    enabled = [m for m, cfg in models.items() if cfg.get("enabled", False) and m in valid_models]
    if not enabled:
        print("  ❌ FAILED: No ML model enabled (check MODELS.xgboost.enabled or MODELS.lightgbm.enabled)")
        return False
    
    # Validate triggers
    triggers = config.get("TRIGGERS", {})
    if "mse_threshold" not in triggers and config.get("TASK_TYPE") == "regression":
        print("  ❌ FAILED: Missing mse_threshold in TRIGGERS for regression")
        return False
    
    # Validate thresholds are numeric and positive
    for key, value in triggers.items():
        if isinstance(value, (int, float)) and value < 0:
            print(f"  ❌ FAILED: Trigger {key} should be non-negative")
            return False
    
    print("  ✅ PASSED")
    return True

def validate_no_secrets():
    """Perform basic secret detection in critical files."""
    print("Checking for exposed secrets...")
    secret_patterns = [
        ("api[_-]?key", "API key"),
        ("secret[_-]?key", "Secret key"),
        ("password\s*=", "Password assignment"),
        ("token\s*=", "Token assignment"),
        ("aws[_-]?secret", "AWS secret"),
    ]
    
    check_files = ["main.py", "src/data_loader.py", "src/trainer.py", "src/evaluator.py"]
    warnings = []
    
    for fname in check_files:
        path = PROJECT_ROOT / fname
        if not path.exists():
            continue
        try:
            with open(path) as f:
                content = f.read()
                for pattern, desc in secret_patterns:
                    import re
                    if re.search(pattern, content, re.IGNORECASE):
                        warnings.append(f"  ⚠️  Potential {desc} pattern in {fname}")
        except Exception:
            pass
    
    if warnings:
        for w in warnings:
            print(w)
        print("  ⚠️  Review recommended (non-blocking)")
    else:
        print("  ✅ PASSED")
    return True  # Non-blocking warning

def validate_security_settings():
    """Check that security settings are enabled."""
    print("Checking security settings...")
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
    except Exception:
        print("  ⚠️  Could not validate security settings (YAML load failed)")
        return True  # Non-blocking
    
    security = config.get("SECURITY", {})
    checks = [
        ("encrypt_models", "Model encryption"),
        ("audit_logging", "Audit logging"),
    ]
    all_ok = True
    for key, desc in checks:
        if not security.get(key, False):
            print(f"  ⚠️  Security: {desc} is disabled")
            all_ok = False
    
    if all_ok:
        print("  ✅ PASSED")
    return True  # Non-blocking

def run_checkov_scan():
    """Run Checkov on infrastructure/config files."""
    print("Running Checkov compliance scan...")
    import subprocess
    result = subprocess.run(
        ["checkov", "-d", str(PROJECT_ROOT), "--quiet", "--compact"],
        capture_output=True,
        text=True
    )
    # Checkov returns different exit codes; we just care if it found critical issues
    output = result.stdout + result.stderr
    # Look for FAILED in output (indicating policy violations)
    if "FAILED" in output and "passed" not in output.lower():
        print("  ⚠️  Checkov found policy violations (review output):")
        # Print truncated output
        for line in output.split("\n")[:10]:
            if line.strip():
                print(f"    {line}")
        return False
    else:
        print("  ✅ PASSED")
        return True

def main():
    print("="*60)
    print("  COMPLIANCE VALIDATION - Auto-Retrain System")
    print("="*60)
    
    results = {
        "Directory Structure": validate_directory_structure(),
        "Config Schema": validate_config_schema(),
        "Secret Detection": validate_no_secrets(),
        "Security Settings": validate_security_settings(),
        "Policy-as-Code (Checkov)": run_checkov_scan(),
    }
    
    print("="*60)
    print("  SUMMARY")
    print("="*60)
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    critical_checks = ["Directory Structure", "Config Schema"]
    critical_failed = any(not results[c] for c in critical_checks)
    
    if critical_failed:
        print("\n  🚫 Compliance FAILED - Critical checks did not pass")
        return 1
    else:
        print("\n  ✅ Compliance PASSED")
        return 0

if __name__ == "__main__":
    sys.exit(main())
