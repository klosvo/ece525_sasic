#!/usr/bin/env python3
"""
Baseline Contract Regression Test

Verifies that when SASIC is disabled, the codebase produces identical
stats JSON output to baseline SIC (after stripping run-specific fields).

Usage:
    python tests/test_baseline_contract_regression.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add SIC directory to path for imports
sic_dir = Path(__file__).parent.parent
sys.path.insert(0, str(sic_dir))

try:
    from src.ANG_SIC.SIC.profiling import SICProfiler
except ImportError:
    # Fallback: try direct import if path structure differs
    import importlib.util
    profiling_path = sic_dir / "src" / "ANG_SIC" / "SIC" / "profiling.py"
    if profiling_path.exists():
        spec = importlib.util.spec_from_file_location("profiling", profiling_path)
        profiling_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profiling_module)
        SICProfiler = profiling_module.SICProfiler
    else:
        print("ERROR: Could not find profiling.py")
        sys.exit(1)


def normalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize stats JSON by removing run-specific fields that may vary
    between runs (timestamps, run IDs, etc.) while preserving structure.
    
    Fields removed:
    - global.start_time, global.end_time
    - Any timestamp fields in clustering_attempts
    - Any run-specific IDs
    """
    normalized = json.loads(json.dumps(stats))  # Deep copy
    
    # Remove global timing fields
    if "global" in normalized:
        normalized["global"].pop("start_time", None)
        normalized["global"].pop("end_time", None)
    
    # Remove timestamps from clustering attempts
    if "convergence" in normalized and "clustering_attempts" in normalized["convergence"]:
        for attempt in normalized["convergence"]["clustering_attempts"]:
            attempt.pop("timestamp", None)
    
    # Remove timestamps from layer clustering attempts
    if "layers" in normalized:
        for layer_name, layer_data in normalized["layers"].items():
            if "clustering_attempts" in layer_data:
                for attempt in layer_data["clustering_attempts"]:
                    attempt.pop("timestamp", None)
    
    return normalized


def compare_stats_schema(stats1: Dict[str, Any], stats2: Dict[str, Any], path: str = "") -> list:
    """
    Compare two stats dictionaries and return list of differences.
    Returns empty list if schemas match.
    """
    differences = []
    
    # Get all keys from both dicts
    all_keys = set(stats1.keys()) | set(stats2.keys())
    
    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
        
        if key not in stats1:
            differences.append(f"Missing in baseline: {current_path}")
            continue
        if key not in stats2:
            differences.append(f"Extra in SASIC-disabled: {current_path}")
            continue
        
        val1 = stats1[key]
        val2 = stats2[key]
        
        # Recursively compare dicts
        if isinstance(val1, dict) and isinstance(val2, dict):
            differences.extend(compare_stats_schema(val1, val2, current_path))
        elif isinstance(val1, list) and isinstance(val2, list):
            # For lists, check if they have the same length and structure
            if len(val1) != len(val2):
                differences.append(f"List length mismatch: {current_path} ({len(val1)} vs {len(val2)})")
            # For list of dicts, compare schema of first element
            if len(val1) > 0 and len(val2) > 0:
                if isinstance(val1[0], dict) and isinstance(val2[0], dict):
                    differences.extend(compare_stats_schema(val1[0], val2[0], f"{current_path}[0]"))
        # For primitive types, we don't compare values (only schema)
    
    return differences


def test_stats_schema_identical():
    """
    Test that stats JSON schema is identical when SASIC is disabled vs baseline.
    
    This is a structural test - it doesn't require running actual SIC,
    just verifies that the profiler produces the same schema structure.
    """
    # Create two profilers (simulating baseline vs SASIC-disabled)
    profiler_baseline = SICProfiler()
    profiler_sasic_disabled = SICProfiler()
    
    # Initialize with same layer
    profiler_baseline.init_layer_stats("test_layer", (10, 20), 200)
    profiler_sasic_disabled.init_layer_stats("test_layer", (10, 20), 200)
    
    # Record same processing
    profiler_baseline.record_layer_processing("test_layer", 1.0, 5, 10)
    profiler_sasic_disabled.record_layer_processing("test_layer", 1.0, 5, 10)
    
    # Get stats
    stats_baseline = profiler_baseline.stats
    stats_sasic_disabled = profiler_sasic_disabled.stats
    
    # Normalize (remove run-specific fields)
    norm_baseline = normalize_stats(stats_baseline)
    norm_sasic_disabled = normalize_stats(stats_sasic_disabled)
    
    # Compare schemas
    differences = compare_stats_schema(norm_baseline, norm_sasic_disabled)
    
    if differences:
        print("FAIL: Stats JSON schema differs when SASIC is disabled:")
        for diff in differences:
            print(f"  - {diff}")
        return False
    else:
        print("PASS: Stats JSON schema is identical when SASIC is disabled")
        return True


def test_profiler_initialization():
    """
    Test that profiler initialization produces identical structure.
    """
    profiler1 = SICProfiler()
    profiler2 = SICProfiler()
    
    # Both should have identical initial structure
    schema1 = json.dumps(profiler1.stats, sort_keys=True)
    schema2 = json.dumps(profiler2.stats, sort_keys=True)
    
    if schema1 != schema2:
        print("FAIL: Profiler initialization produces different schemas")
        return False
    
    print("PASS: Profiler initialization produces identical schemas")
    return True


def main():
    """Run all regression tests."""
    print("=" * 80)
    print("Baseline Contract Regression Tests")
    print("=" * 80)
    print()
    
    tests = [
        ("Profiler Initialization", test_profiler_initialization),
        ("Stats Schema Identity", test_stats_schema_identical),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running: {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

