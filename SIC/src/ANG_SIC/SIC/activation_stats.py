"""
SASIC Activation Statistics Collection Module

This module collects activation statistics from a calibration slice to support
Sparsity-Aware Synaptic Input Consolidation (SASIC) Mode A (Active-Subset).

The module performs training-free forward passes on a small subset of data to
compute per-input activation indicators, which are then used to identify
active vs. quiet inputs during clustering.

See sasic_design.md §4 (Activation Statistics) and §8.3 (Activation Pass) for
the specification. Current implementation uses nested dict storage; design doc
§4.3 describes a future tensor-based interface.

TODO: Enforce strict class-balance in calibration slice (design doc §4.1).
      Currently uses random subset via calibration_fraction.
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings


def collect_activation_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sasic_cfg: Dict[str, Any],
    sic_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[int, Dict[int, float]]]:
    """
    Collect activation statistics for SASIC from a calibration slice.

    This function performs forward passes on a calibration subset to compute
    per-input activation indicators for each layer, neuron, and input index.

    Args:
        model: PyTorch model (will be set to eval mode, no gradients).
        loader: DataLoader for the calibration slice (should be class-balanced
            subset of training/validation data).
        device: Device to run forward passes on.
        sasic_cfg: SASIC configuration dict containing:
            - `activation_stat`: str, one of "p_above" or "mean_abs"
                - "p_above": probability that |x_i| > activation_threshold
                - "mean_abs": mean absolute activation E[|x_i|]
            - `activation_threshold`: float, threshold for "p_above" mode
            - `calibration_fraction`: float, fraction of data used (for logging)
        sic_cfg: Optional SIC configuration dict (for future use, e.g., layer
            exclusion rules). Currently unused.

    Returns:
        Nested dict structure:
            activation_stats[layer_name][neuron_idx][input_idx] = a_i
        where:
            - `layer_name`: str, name of the layer/module
            - `neuron_idx`: int, index of the neuron (row index in weight matrix)
            - `input_idx`: int, index of the input (column index in weight matrix)
            - `a_i`: float, activation indicator in [0, 1] or [0, inf) depending on stat type

    Returns:
        Nested dict structure:
            activation_stats[layer_name][neuron_idx][input_idx] = a_i
        where:
            - `layer_name`: str, name of the layer/module
            - `neuron_idx`: int, index of the neuron (row index in weight matrix)
            - `input_idx`: int, index of the input (column index in weight matrix)
            - `a_i`: float, activation indicator in [0, 1] or [0, inf) depending on stat type

    Example output structure:
        {
            "fc1": {
                0: {0: 0.85, 1: 0.12, 2: 0.91, ...},  # neuron 0, inputs 0,1,2,...
                1: {0: 0.23, 1: 0.67, 2: 0.45, ...},  # neuron 1, inputs 0,1,2,...
                ...
            },
            "fc2": {
                ...
            },
        }

    Notes:
        - Only processes layers that have a weight tensor (Linear, Conv2d, etc.)
        - Skips BatchNorm and other layers without weights
        - Statistics are computed over all batches in the calibration loader
        - Model is set to eval() mode and gradients are disabled
    """
    # Check for empty loader or invalid calibration fraction
    calibration_fraction = float(sasic_cfg.get("calibration_fraction", 0.1))
    if calibration_fraction <= 0:
        warnings.warn("calibration_fraction <= 0, returning empty activation stats")
        return {}
    
    try:
        _ = iter(loader)
        if len(loader) == 0:
            warnings.warn("Calibration loader is empty, returning empty activation stats")
            return {}
    except (TypeError, AttributeError):
        warnings.warn("Invalid loader, returning empty activation stats")
        return {}

    # Get configuration
    # activation_stat: "p_above" ≈ nzrate, "mean_abs" = mean absolute activation
    # activation_threshold: maps to tau_active in design doc (§5.1, §7.2), threshold for "p_above" mode
    activation_stat = str(sasic_cfg.get("activation_stat", "p_above"))
    activation_threshold = float(sasic_cfg.get("activation_threshold", 0.01))
    # weight_exponent: reserved for future weighted modes (gamma in design doc §4.2, §7.2), not used in Mode A
    _weight_exponent = float(sasic_cfg.get("weight_exponent", 1.0))  # Not used in Mode A
    
    if activation_stat not in {"p_above", "mean_abs"}:
        raise ValueError(
            f"Invalid activation_stat: '{activation_stat}'. Must be one of: 'p_above', 'mean_abs'"
        )

    # Get target layers
    target_layers = _get_target_layers(model, sic_cfg)
    if not target_layers:
        return {}

    # Storage for activations: layer_name -> list of (batch, flattened_input) tensors
    activation_buffers: Dict[str, List[torch.Tensor]] = {name: [] for name in target_layers}
    
    # Register hooks
    hook_handles = _register_forward_hooks(model, target_layers, activation_buffers)
    
    try:
        # Set model to eval mode and disable gradients
        model.eval()
        with torch.no_grad():
            # Run forward passes on all batches
            for batch in loader:
                # Extract input from batch
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                elif isinstance(batch, dict):
                    x = batch.get("pixel_values") or batch.get("images") or batch.get("input") or batch.get("x")
                else:
                    x = batch
                
                if x is None:
                    continue
                
                # Move to device
                if hasattr(x, "to"):
                    x = x.to(device, non_blocking=(device.type == "cuda"))
                
                # Forward pass (hooks will capture inputs)
                _ = model(x)
        
        # Aggregate statistics
        activation_stats: Dict[str, Dict[int, Dict[int, float]]] = {}
        
        for layer_name in target_layers:
            if not activation_buffers[layer_name]:
                continue
            
            # Get layer module to determine weight shape
            module = dict(model.named_modules())[layer_name]
            if not hasattr(module, "weight") or not torch.is_tensor(module.weight):
                continue
            
            w = module.weight
            w_shape = w.shape
            
            # Concatenate all batches for this layer
            all_inputs = torch.cat(activation_buffers[layer_name], dim=0)  # (total_samples, flattened_input_size)
            
            # Determine number of neurons and inputs based on layer type
            if isinstance(module, nn.Linear):
                # Linear: weight shape (out_features, in_features)
                num_neurons = w_shape[0]  # out_features
                num_inputs = w_shape[1]   # in_features
                # Inputs are already flattened: (batch, in_features)
                if all_inputs.shape[1] != num_inputs:
                    continue
                
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Conv: weight shape (out_channels, in_channels, ...)
                # SIC flattens conv weights to (out_channels, in_channels * kH * kW)
                num_neurons = w_shape[0]  # out_channels
                # Flattened weight: (out_channels, in_channels * kernel_size)
                num_inputs = w.numel() // num_neurons  # in_channels * kernel_size
                
                # Input was flattened from (batch, in_channels, H, W, ...) to (batch, in_channels * H * W * ...)
                # We need to map this to the kernel structure
                # For simplicity, we'll use the flattened input size and match it to weight structure
                if all_inputs.shape[1] < num_inputs:
                    continue
                # If input is larger (spatial dims), we'll use the first num_inputs elements
                # This approximates the receptive field
                if all_inputs.shape[1] > num_inputs:
                    all_inputs = all_inputs[:, :num_inputs]
            else:
                # Other layer types: use flattened weight shape
                num_neurons = w_shape[0]
                num_inputs = w.numel() // num_neurons
                if all_inputs.shape[1] < num_inputs:
                    continue
                if all_inputs.shape[1] > num_inputs:
                    all_inputs = all_inputs[:, :num_inputs]
            
            # Initialize stats for this layer
            layer_stats: Dict[int, Dict[int, float]] = {}
            
            # For each neuron, compute activity rate for each input
            # Note: For Linear layers, all neurons see the same input
            # For Conv layers, we need to handle spatial dimensions properly
            for neuron_idx in range(num_neurons):
                # For Linear: all neurons see same input, so we can reuse
                # For Conv: each output channel sees the same input (flattened)
                # So we compute stats once per input index
                input_stats: Dict[int, float] = {}
                
                for input_idx in range(num_inputs):
                    # Extract activations for this input across all samples
                    input_activations = all_inputs[:, input_idx]  # (num_samples,)
                    
                    # Compute activity rate
                    a_i = _compute_activity_rate(
                        input_activations.unsqueeze(1),  # (num_samples, 1) for compatibility
                        activation_stat,
                        activation_threshold,
                    )
                    
                    # Convert to Python float
                    input_stats[input_idx] = float(a_i.item())
                
                layer_stats[neuron_idx] = input_stats
            
            activation_stats[layer_name] = layer_stats
    
    finally:
        # Always remove hooks
        _remove_forward_hooks(hook_handles)
    
    return activation_stats


def _register_forward_hooks(
    model: nn.Module,
    target_layers: List[str],
    activation_buffers: Dict[str, List[torch.Tensor]],
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register forward hooks to capture input activations for target layers.

    Args:
        model: PyTorch model to attach hooks to.
        target_layers: List of layer names to hook.
        activation_buffers: Dict mapping layer names to lists where activations
            will be stored. Modified in-place by hooks.

    Returns:
        List of hook handles that can be used to remove hooks later.
    """
    hook_handles = []
    named_modules = dict(model.named_modules())
    
    for layer_name in target_layers:
        if layer_name not in named_modules:
            continue
        
        module = named_modules[layer_name]
        
        def make_hook(name: str, buffer_list: List[torch.Tensor]):
            def hook_fn(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
                # Capture the input (first element of input tuple)
                if input and len(input) > 0:
                    inp = input[0]
                    # Flatten spatial dimensions for conv layers, keep as-is for linear
                    if inp.ndim > 2:
                        # Conv layer: flatten (batch, channels, H, W, ...) to (batch, channels * H * W * ...)
                        batch_size = inp.shape[0]
                        inp_flat = inp.view(batch_size, -1)
                    else:
                        # Linear layer: already (batch, features)
                        inp_flat = inp
                    
                    # Detach and move to CPU to save memory
                    buffer_list.append(inp_flat.detach().cpu())
            
            return hook_fn
        
        handle = module.register_forward_hook(make_hook(layer_name, activation_buffers[layer_name]))
        hook_handles.append(handle)
    
    return hook_handles


def _remove_forward_hooks(hook_handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    """
    Remove all registered forward hooks.

    Args:
        hook_handles: List of hook handles returned by _register_forward_hooks().
    """
    for handle in hook_handles:
        handle.remove()


def _compute_activity_rate(
    activations: torch.Tensor,
    stat_type: str,
    threshold: float,
) -> torch.Tensor:
    """
    Compute activity rate indicator for a set of activations.

    Args:
        activations: Tensor of shape (num_samples, 1) containing
            activation values for a single input across multiple samples.
        stat_type: str, one of "p_above" or "mean_abs"
            - "p_above": probability that |x_i| > threshold
            - "mean_abs": mean absolute activation E[|x_i|]
        threshold: float, threshold for "p_above" mode (ignored for "mean_abs").

    Returns:
        Scalar tensor containing activity rate a_i for this input.

    Raises:
        ValueError: If stat_type is not "p_above" or "mean_abs".
    """
    if stat_type == "p_above":
        # Probability that |x_i| > threshold
        abs_activations = torch.abs(activations)
        above_threshold = (abs_activations > threshold).float()
        a_i = above_threshold.mean()
        return a_i
    elif stat_type == "mean_abs":
        # Mean absolute activation E[|x_i|]
        abs_activations = torch.abs(activations)
        a_i = abs_activations.mean()
        return a_i
    else:
        raise ValueError(f"Invalid stat_type: '{stat_type}'. Must be one of: 'p_above', 'mean_abs'")


def _get_target_layers(model: nn.Module, sic_cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Get list of layer names that should have activation statistics collected.

    Args:
        model: PyTorch model to inspect.
        sic_cfg: Optional SIC configuration dict (for layer exclusion rules).
            Currently checks for `sic.exclude_modules_by_name` list.

    Returns:
        List of layer names (matching model.named_modules() keys) that have
        weight tensors and should be processed.
    """
    from .pos_common import is_pos_linear, is_pos_conv2d
    
    target_layers = []
    exclude_names = []
    
    if sic_cfg:
        sic_section = sic_cfg.get("sic", {}) or {}
        exclude_names = sic_section.get("exclude_modules_by_name", [])
    
    for name, module in model.named_modules():
        # Skip BatchNorm layers
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            continue
        
        # Skip excluded by name
        if exclude_names:
            if any(excluded in name for excluded in exclude_names):
                continue
        
        # Skip PoS layers (no weight)
        if is_pos_linear(module) or is_pos_conv2d(module):
            continue
        
        # Only include layers with weight tensors
        if hasattr(module, "weight") and torch.is_tensor(getattr(module, "weight")):
            target_layers.append(name)
    
    return target_layers
