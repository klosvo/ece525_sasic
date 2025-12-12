"""
SASIC Mode A (Active-Subset) Utility Functions

Helper functions for implementing Active-Subset SASIC in the SIC clustering pipeline.
These functions support Mode A only (not Weighted or Hybrid modes).
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
) -> Optional[np.ndarray]:
    """
    Get active input indices for a specific neuron based on activation statistics.
    
    Args:
        activation_stats: Nested dict from collect_activation_stats:
            activation_stats[layer_key][neuron_idx][input_idx] = a_i
        layer_key: Layer identifier (from get_layer_key_for_sasic).
        neuron_idx: Neuron index (row index in flattened weight matrix).
        threshold: Activation threshold (a_i >= threshold means active).
    
    Returns:
        Boolean mask array (1D numpy array) of same length as number of inputs,
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
    
    # Get number of inputs (max input_idx + 1, or infer from stats)
    max_input_idx = max(neuron_stats.keys()) if neuron_stats else -1
    if max_input_idx < 0:
        return None
    
    num_inputs = max_input_idx + 1
    
    # Build boolean mask: True where a_i >= threshold
    active_mask = np.zeros(num_inputs, dtype=bool)
    for input_idx, a_i in neuron_stats.items():
        if isinstance(a_i, (int, float)) and not np.isnan(a_i):
            if a_i >= threshold:
                active_mask[int(input_idx)] = True
    
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
    
    This implements Mode A's quiet input attachment step:
    - Active inputs are already assigned to clusters (via cluster_assignments).
    - Quiet inputs are assigned to the nearest cluster center by absolute weight distance.
    
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
