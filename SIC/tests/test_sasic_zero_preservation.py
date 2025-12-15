#!/usr/bin/env python3
"""
Tests for SASIC Mode A zero preservation.

Verifies that Mode A never converts exact zeros (w == 0.0) into nonzero values,
preserving baseline SIC behavior for zero weights.

Pure numpy test - no torch dependencies required.
"""

import numpy as np
import sys
from pathlib import Path

# Add SIC directory to path for imports
sic_dir = Path(__file__).parent.parent
sys.path.insert(0, str(sic_dir))

# Import only the function we need, avoiding torch imports
import importlib.util
sasic_utils_path = sic_dir / "src" / "ANG_SIC" / "SIC" / "sasic_utils.py"
if not sasic_utils_path.exists():
    print("ERROR: Could not find sasic_utils.py")
    sys.exit(1)

# Load module without executing torch imports
spec = importlib.util.spec_from_file_location("sasic_utils", sasic_utils_path)
sasic_utils_module = importlib.util.module_from_spec(spec)

# Temporarily mock torch to avoid import errors
import types
mock_torch = types.ModuleType('torch')
mock_torch.nn = types.ModuleType('torch.nn')
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn

try:
    spec.loader.exec_module(sasic_utils_module)
    attach_quiet_inputs_to_clusters = sasic_utils_module.attach_quiet_inputs_to_clusters
finally:
    # Clean up mock
    if 'torch' in sys.modules and sys.modules['torch'] is mock_torch:
        del sys.modules['torch']
        del sys.modules['torch.nn']


def test_zero_preservation_basic():
    """Test B: Verify all zeros remain zero after attachment."""
    # Construct weight vector with zeros and nonzeros
    weights_full = np.array([0.0, 0.5, 0.0, 0.3, 0.0, 0.7, 0.0, 0.2], dtype=np.float32)
    # Active mask: some zeros and nonzeros are active, some are quiet
    active_mask = np.array([True, True, False, True, False, False, True, False], dtype=bool)
    
    # Cluster centers from Jenks on active nonzero [0.5, 0.3]
    cluster_centers = np.array([0.25, 0.5], dtype=np.float32)
    
    # Cluster assignments for active inputs (only nonzero ones matter)
    # active_indices = [0, 1, 3, 6] (indices 0, 1, 3, 6 are active)
    # active nonzero = [1, 3] (indices 1, 3 are active nonzero)
    # cluster_assignments[1] = cluster for index 1 (0.5) -> cluster 1
    # cluster_assignments[3] = cluster for index 3 (0.3) -> cluster 0
    cluster_assignments = np.array([0, 1, 0, 0], dtype=np.int32)  # For active indices [0, 1, 3, 6]
    
    result = attach_quiet_inputs_to_clusters(
        weights_full=weights_full,
        active_mask=active_mask,
        cluster_centers=cluster_centers,
        cluster_assignments=cluster_assignments,
        vectorized=True,
    )
    
    # Verify all original zeros remain zero
    original_zero_mask = weights_full == 0.0
    assert np.all(result[original_zero_mask] == 0.0), \
        f"Zeros were converted! Original zeros at {np.where(original_zero_mask)[0]}, " \
        f"result values: {result[original_zero_mask]}"
    
    # Verify active nonzero were assigned to cluster centers
    assert result[1] == cluster_centers[1], f"Active nonzero at index 1 should be {cluster_centers[1]}, got {result[1]}"
    assert result[3] == cluster_centers[0], f"Active nonzero at index 3 should be {cluster_centers[0]}, got {result[3]}"
    
    # Verify quiet nonzero were attached to nearest cluster
    # Index 5: 0.7 -> nearest to 0.5 (cluster 1)
    assert result[5] == cluster_centers[1], f"Quiet nonzero at index 5 should be {cluster_centers[1]}, got {result[5]}"
    # Index 7: 0.2 -> nearest to 0.25 (cluster 0)
    assert result[7] == cluster_centers[0], f"Quiet nonzero at index 7 should be {cluster_centers[0]}, got {result[7]}"
    
    print("PASS: test_zero_preservation_basic")


def test_zero_preservation_all_active():
    """Test A (equivalence sanity): If all inputs are active, zeros should still be preserved."""
    # All inputs active, but some are zero
    weights_full = np.array([0.0, 0.5, 0.0, 0.3, 0.7], dtype=np.float32)
    active_mask = np.array([True, True, True, True, True], dtype=bool)
    
    # Cluster centers from Jenks on active nonzero [0.5, 0.3, 0.7]
    cluster_centers = np.array([0.3, 0.5, 0.7], dtype=np.float32)
    
    # Cluster assignments for all active inputs
    # active_indices = [0, 1, 2, 3, 4]
    # active nonzero = [1, 3, 4]
    cluster_assignments = np.array([0, 1, 0, 0, 2], dtype=np.int32)
    
    result = attach_quiet_inputs_to_clusters(
        weights_full=weights_full,
        active_mask=active_mask,
        cluster_centers=cluster_centers,
        cluster_assignments=cluster_assignments,
        vectorized=True,
    )
    
    # Verify all zeros remain zero
    original_zero_mask = weights_full == 0.0
    assert np.all(result[original_zero_mask] == 0.0), \
        f"Zeros were converted! Original zeros at {np.where(original_zero_mask)[0]}, " \
        f"result values: {result[original_zero_mask]}"
    
    # Verify nonzero were assigned correctly
    assert result[1] == cluster_centers[1]
    assert result[3] == cluster_centers[0]
    assert result[4] == cluster_centers[2]
    
    print("PASS: test_zero_preservation_all_active")


def test_zero_preservation_all_quiet():
    """Test that quiet zeros remain zero."""
    # All inputs quiet, some are zero
    weights_full = np.array([0.0, 0.5, 0.0, 0.3], dtype=np.float32)
    active_mask = np.array([False, False, False, False], dtype=bool)
    
    # Cluster centers (from some other neuron's active set)
    cluster_centers = np.array([0.25, 0.5], dtype=np.float32)
    
    # No active inputs, so cluster_assignments is empty
    cluster_assignments = np.array([], dtype=np.int32)
    
    result = attach_quiet_inputs_to_clusters(
        weights_full=weights_full,
        active_mask=active_mask,
        cluster_centers=cluster_centers,
        cluster_assignments=cluster_assignments,
        vectorized=True,
    )
    
    # Verify all zeros remain zero
    original_zero_mask = weights_full == 0.0
    assert np.all(result[original_zero_mask] == 0.0), \
        f"Quiet zeros were converted! Original zeros at {np.where(original_zero_mask)[0]}, " \
        f"result values: {result[original_zero_mask]}"
    
    # Verify quiet nonzero were attached
    assert result[1] == cluster_centers[1]  # 0.5 -> nearest to 0.5
    assert result[3] == cluster_centers[0]  # 0.3 -> nearest to 0.25
    
    print("PASS: test_zero_preservation_all_quiet")


def test_zero_preservation_edge_cases():
    """Test edge cases: all zeros, all nonzero, etc."""
    # Case 1: All zeros
    weights_full = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    active_mask = np.array([True, False, True], dtype=bool)
    cluster_centers = np.array([0.5], dtype=np.float32)
    cluster_assignments = np.array([0, 0], dtype=np.int32)  # For 2 active inputs
    
    result = attach_quiet_inputs_to_clusters(
        weights_full=weights_full,
        active_mask=active_mask,
        cluster_centers=cluster_centers,
        cluster_assignments=cluster_assignments,
        vectorized=True,
    )
    
    assert np.all(result == 0.0), "All zeros should remain zero"
    print("PASS: test_zero_preservation_edge_cases (all zeros)")
    
    # Case 2: All nonzero
    weights_full = np.array([0.5, 0.3, 0.7], dtype=np.float32)
    active_mask = np.array([True, True, False], dtype=bool)
    cluster_centers = np.array([0.3, 0.5], dtype=np.float32)
    cluster_assignments = np.array([0, 1], dtype=np.int32)
    
    result = attach_quiet_inputs_to_clusters(
        weights_full=weights_full,
        active_mask=active_mask,
        cluster_centers=cluster_centers,
        cluster_assignments=cluster_assignments,
        vectorized=True,
    )
    
    # No zeros to preserve, but verify function works
    assert result[0] == cluster_centers[0]
    assert result[1] == cluster_centers[1]
    assert result[2] == cluster_centers[1]  # 0.7 -> nearest to 0.5
    print("PASS: test_zero_preservation_edge_cases (all nonzero)")


def main():
    """Run all zero preservation tests."""
    print("=" * 80)
    print("SASIC Mode A Zero Preservation Tests")
    print("=" * 80)
    print()
    
    tests = [
        ("Zero Preservation Basic", test_zero_preservation_basic),
        ("Zero Preservation All Active", test_zero_preservation_all_active),
        ("Zero Preservation All Quiet", test_zero_preservation_all_quiet),
        ("Zero Preservation Edge Cases", test_zero_preservation_edge_cases),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running: {test_name}...")
        try:
            test_func()
            results.append((test_name, True))
        except AssertionError as e:
            print(f"FAIL: {e}")
            results.append((test_name, False))
        except Exception as e:
            print(f"ERROR: {e}")
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

