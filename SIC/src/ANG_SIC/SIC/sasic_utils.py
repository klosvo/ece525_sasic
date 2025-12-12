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
) -> np.ndarray:
    """
    Attach quiet inputs to nearest cluster centers by weight distance.
    
    Implements Mode A quiet input attachment (sasic_design.md §5.1, step 4):
    - Active inputs are already assigned to clusters (via cluster_assignments).
    - Quiet inputs (j not in A) are attached to nearest cluster center by weight distance:
      j -> argmin_c |w_j - mu_c|
    
    Args:
        weights_full: 1D numpy array of original input weights for this neuron
            (full vector, same length as active_mask).
        active_mask: Boolean mask (1D numpy array), True = active, False = quiet.
            Must have same length as weights_full.
        cluster_centers: 1D numpy array of cluster centers (one per cluster),
            computed from Jenks clustering on active inputs.
        cluster_assignments: 1D numpy array of cluster indices for active inputs only.
            Length should equal np.sum(active_mask).
            cluster_assignments[i] is the cluster index for the i-th active input.
    
    Returns:
        1D numpy array of same length as weights_full, containing consolidated values:
        - Active inputs: assigned to their cluster center (from cluster_assignments).
        - Quiet inputs: assigned to nearest cluster center by |w - center|.
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
    
    # Map active inputs to their cluster centers
    active_indices = np.where(active_mask)[0]
    for i, input_idx in enumerate(active_indices):
        cluster_idx = int(cluster_assignments[i])
        if 0 <= cluster_idx < len(cluster_centers):
            consolidated[input_idx] = cluster_centers[cluster_idx]
    
    # Attach quiet inputs to nearest cluster center
    quiet_indices = np.where(~active_mask)[0]
    for input_idx in quiet_indices:
        w_quiet = weights_full[input_idx]
        # Find nearest cluster center by absolute distance
        distances = np.abs(cluster_centers - w_quiet)
        nearest_cluster = int(np.argmin(distances))
        consolidated[input_idx] = cluster_centers[nearest_cluster]
    
    return consolidated
