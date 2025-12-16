"""
SASIC Mode A (Active-Subset) Utility Functions

Helper functions for implementing Active-Subset SASIC in the SIC clustering pipeline.
These functions support Mode A only (not Weighted or Hybrid modes).

See sasic_design.md §5.1 (Mode A — Active-Subset SASIC) for the specification.
"""

from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn


def get_layer_key_for_sasic(module_name: str, module: nn.Module) -> str:
    """
    Get the layer key used in activation_stats for a given module.
    
    This should match the key used when collecting activation statistics
    in collect_activation_stats(). The key is typically the module's
    qualified name as returned by model.named_modules().
    
    Args:
        module_name: The name/identifier of the module (e.g., from model.named_modules()).
        module: The module object (currently unused but kept for consistency).
    
    Returns:
        The layer key string used in activation_stats dict.
    """
    # The key in activation_stats is the module name as returned by named_modules()
    return module_name


def get_active_indices_for_neuron(
    activation_stats: Optional[Dict[str, Dict[int, Dict[int, float]]]],
    layer_key: str,
    neuron_idx: int,
    threshold: float,
    num_inputs: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Get active input indices for a specific neuron based on activation statistics.
    
    Implements Mode A active set definition: A = { i : a_i >= threshold }.
    See sasic_design.md §5.1 for specification.
    
    Args:
        activation_stats: Nested dict from collect_activation_stats:
            activation_stats[layer_key][neuron_idx][input_idx] = a_i
        layer_key: Layer identifier (from get_layer_key_for_sasic).
        neuron_idx: Neuron index (row index in flattened weight matrix).
        threshold: Activation threshold (tau_active in design doc; a_i >= threshold means active).
        num_inputs: Optional ground-truth number of inputs (from weight vector length).
            If provided, mask will be exactly this length. Missing indices in stats
            are treated as quiet (False). If None, infers from max(input_idx) + 1.
    
    Returns:
        Boolean mask array (1D numpy array) of length num_inputs (or inferred),
        where True indicates active inputs (a_i >= threshold).
        Returns None if:
        - activation_stats is None
        - layer_key not found in stats
        - neuron_idx not found for this layer
        - All a_i values are invalid/NaN
        - No inputs are active (all False) - caller should handle fallback
    """
    if activation_stats is None:
        return None
    
    if layer_key not in activation_stats:
        return None
    
    layer_stats = activation_stats[layer_key]
    if neuron_idx not in layer_stats:
        return None
    
    neuron_stats = layer_stats[neuron_idx]
    if not neuron_stats:
        return None
    
    # Determine mask length: use ground truth if provided, otherwise infer
    # (sasic_design.md §4.3: stats must align with flattened weight indices)
    # Always prefer ground-truth num_inputs to avoid mask length mismatches
    if num_inputs is not None:
        mask_len = int(num_inputs)
        # Validate: stats should not contain indices >= num_inputs
        max_idx_in_stats = max(neuron_stats.keys()) if neuron_stats else -1
        if max_idx_in_stats >= mask_len:
            # Inconsistent: stats have indices beyond expected length
            return None
    else:
        # Infer from stats (legacy behavior, less reliable)
        # Caller should always provide num_inputs for Mode A
        max_input_idx = max(neuron_stats.keys()) if neuron_stats else -1
        if max_input_idx < 0:
            return None
        mask_len = max_input_idx + 1
    
    # Build boolean mask: True where a_i >= threshold
    # Missing indices in stats are treated as quiet (False)
    # (sasic_design.md §5.1: A = { i : a_i >= threshold })
    active_mask = np.zeros(mask_len, dtype=bool)
    for input_idx, a_i in neuron_stats.items():
        input_idx_int = int(input_idx)
        if input_idx_int >= mask_len:
            # Out of bounds - skip this entry
            continue
        if isinstance(a_i, (int, float)) and not np.isnan(a_i):
            if a_i >= threshold:
                active_mask[input_idx_int] = True
    
    # Check if all are invalid/NaN
    if not np.any(active_mask) and not np.any(~active_mask):
        # All NaN/invalid - fallback
        return None
    
    return active_mask


def attach_quiet_inputs_to_clusters(
    weights_full: np.ndarray,
    active_mask: np.ndarray,
    cluster_centers: np.ndarray,
    cluster_assignments: np.ndarray,
    vectorized: bool = True,
) -> np.ndarray:
    """
    Attach quiet inputs to nearest cluster centers by weight distance.
    
    Implements Mode A quiet input attachment (sasic_design.md §5.1, step 4):
    - Active NONZERO inputs are assigned to clusters (via cluster_assignments).
    - Quiet NONZERO inputs are attached to nearest cluster center by weight distance.
    - ALL zeros (active and quiet) are preserved as 0.0 (never converted to nonzero).
    
    CRITICAL: This function preserves exact zeros (w == 0.0) like baseline SIC.
    Only nonzero weights are modified.
    
    Args:
        weights_full: 1D numpy array of original input weights for this neuron
            (full vector, same length as active_mask).
        active_mask: Boolean mask (1D numpy array), True = active, False = quiet.
            Must have same length as weights_full.
        cluster_centers: 1D numpy array of cluster centers (one per cluster),
            computed from Jenks clustering on active nonzero inputs.
        cluster_assignments: 1D numpy array of cluster indices for active inputs.
            Length should equal np.sum(active_mask).
            cluster_assignments[i] is the cluster index for the i-th active input.
            Only entries for active nonzero inputs are meaningful.
        vectorized: If True (default), use vectorized distance computation for quiet inputs.
            If False, use the original loop-based implementation.
    
    Returns:
        1D numpy array of same length as weights_full, containing consolidated values:
        - Active nonzero inputs: assigned to their cluster center (from cluster_assignments).
        - Quiet nonzero inputs: assigned to nearest cluster center by |w - center|.
        - ALL zeros (active and quiet): preserved as 0.0 (unchanged).
    """
    if len(weights_full) != len(active_mask):
        raise ValueError(
            f"weights_full length ({len(weights_full)}) != active_mask length ({len(active_mask)})"
        )
    
    if np.sum(active_mask) != len(cluster_assignments):
        raise ValueError(
            f"Number of active inputs ({np.sum(active_mask)}) != "
            f"cluster_assignments length ({len(cluster_assignments)})"
        )
    
    consolidated = weights_full.copy()
    
    # CRITICAL FIX: Only process nonzero indices to preserve zeros like baseline SIC
    # Mode A must never convert exact zeros (w == 0.0) into nonzero values
    
    # Map active NONZERO inputs to their cluster centers (preserve active zeros)
    # cluster_assignments[i] corresponds to the i-th active input in active_indices
    # We only process entries where the active input is nonzero
    active_indices = np.where(active_mask)[0]
    nz_mask = weights_full != 0.0
    
    # Map active nonzero to cluster centers
    # cluster_assignments[pos] is the cluster for active_indices[pos]
    # We only process entries where active_indices[pos] is nonzero
    for pos_in_active, input_idx in enumerate(active_indices):
        if pos_in_active < len(cluster_assignments):
            # Only assign cluster center if this active input is nonzero
            # Active zeros remain zero (preserved)
            if nz_mask[input_idx]:
                cluster_idx = int(cluster_assignments[pos_in_active])
                if 0 <= cluster_idx < len(cluster_centers):
                    consolidated[input_idx] = cluster_centers[cluster_idx]
            # If input_idx is zero (active zero), we skip it (preserve as 0.0)
    
    # Attach quiet NONZERO inputs to nearest cluster center (preserve quiet zeros)
    quiet_indices = np.where(~active_mask)[0]
    quiet_nz_mask = nz_mask[quiet_indices]
    quiet_nz_indices = quiet_indices[quiet_nz_mask]
    
    if len(quiet_nz_indices) == 0:
        return consolidated
    
    if vectorized:
        # Vectorized: compute all distances at once
        # Shape: (num_quiet_nz, num_clusters) - distance from each quiet nonzero weight to each cluster center
        quiet_nz_weights = weights_full[quiet_nz_indices]
        # Broadcasting: quiet_nz_weights[:, None] - cluster_centers[None, :]
        # Results in (num_quiet_nz, num_clusters) matrix of distances
        distances = np.abs(quiet_nz_weights[:, None] - cluster_centers[None, :])
        # Find nearest cluster for each quiet nonzero input (argmin along cluster axis)
        nearest_clusters = np.argmin(distances, axis=1)
        # Assign cluster centers to quiet nonzero inputs only
        consolidated[quiet_nz_indices] = cluster_centers[nearest_clusters]
    else:
        # Original loop-based implementation (for compatibility/testing)
        for input_idx in quiet_nz_indices:
            w_quiet = weights_full[input_idx]
            distances = np.abs(cluster_centers - w_quiet)
            nearest_cluster = int(np.argmin(distances))
            consolidated[input_idx] = cluster_centers[nearest_cluster]
    
    return consolidated


def choose_k_candidates_heuristic(
    max_k: int,
    mode: str,
    max_trials_per_neuron: Optional[int] = None,
) -> list[int]:
    """
    Choose k candidates for clustering based on heuristic mode.
    
    This is a SASIC-only optimization that reduces the number of k values
    evaluated per neuron, reducing runtime while preserving acceptance semantics.
    
    Args:
        max_k: Maximum k value that would be tried in baseline (from uniq_nz - 1, clamped by max_k_per_neuron).
        mode: Heuristic mode. Currently supports:
            - "budgeted_sweep": Evenly spaced k values up to max_trials_per_neuron
        max_trials_per_neuron: Maximum number of k values to try (budget).
            If None or >= max_k, returns full range [1, max_k] (no reduction).
    
    Returns:
        Sorted list of k values to try (integers from 1 to max_k, inclusive).
        Always includes k=1 (minimum) and k=max_k (maximum) if max_k >= 1.
        When heuristic is disabled or budget >= max_k, returns [1, 2, ..., max_k].
    
    Note:
        This function does NOT change acceptance semantics. It only reduces
        the candidate k values considered. The acceptance logic remains unchanged.
    """
    if max_k < 1:
        return []
    
    # If no budget or budget >= max_k, return full range (no reduction)
    if max_trials_per_neuron is None or max_trials_per_neuron >= max_k:
        return list(range(1, max_k + 1))
    
    if mode == "budgeted_sweep":
        # Evenly spaced k values, always including k=1 and k=max_k
        if max_trials_per_neuron <= 2:
            # Minimum: try k=1 and k=max_k
            if max_k == 1:
                return [1]
            return [1, max_k]
        
        # Generate evenly spaced k values
        # Use numpy linspace to get evenly spaced indices, then round to integers
        k_indices = np.linspace(0, max_k - 1, num=max_trials_per_neuron, dtype=int)
        k_values = [int(k_idx + 1) for k_idx in k_indices]  # Convert to 1-indexed k values
        
        # Ensure k=1 and k=max_k are included
        k_set = set(k_values)
        k_set.add(1)
        k_set.add(max_k)
        k_values = sorted(list(k_set))
        
        # Clamp to valid range [1, max_k]
        k_values = [k for k in k_values if 1 <= k <= max_k]
        
        return k_values
    else:
        # Unknown mode: fallback to full range
        return list(range(1, max_k + 1))
