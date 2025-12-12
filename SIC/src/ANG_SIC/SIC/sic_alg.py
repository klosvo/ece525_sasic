from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from time import perf_counter
import jenkspy

from .profiling import SICProfiler, MemoryTracker
from ..core.config import LogBuffer
from .sic_utils import all_samples_correct
from ..core.numeric import clamp_decimals, round_tensor
from ..core.acceptance import build_acceptance_subset_from_loaders
from .pos_common import is_pos_linear, is_pos_conv2d

def _jenks_breaks(arr: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(jenkspy.jenks_breaks(np.asarray(arr, dtype=float), int(k)), dtype=float)

def _flatten_weight(w: torch.Tensor) -> torch.Tensor:
    return w.view(w.size(0), -1) if w.ndim > 2 else w

def _uwc(model: nn.Module, decimals: int, exclude_zero: bool = True) -> int:
    s = set()
    scale = float(10 ** max(0, int(decimals)))
    for m in model.modules():
        if hasattr(m, "weight") and torch.is_tensor(getattr(m, "weight")):
            v = m.weight.detach().view(-1).cpu().numpy()
            v = np.round(v * scale) / scale
            if exclude_zero:
                v = v[v != 0.0]
            s.update(v.tolist())
        if _is_indexed_sparse_linear(m):
            v = m.theta.detach().view(-1).cpu().numpy()
            v = np.round(v * scale) / scale
            if exclude_zero:
                v = v[v != 0.0]
            s.update(v.tolist())
    return len(s)

def _should_gpu_rowlen(row_len: int, sic_cfg: dict) -> bool:
    return int(row_len) >= int(sic_cfg.get("gpu_row_len_min", 2048))

def _is_indexed_sparse_linear(m: nn.Module) -> bool:
    return (
        hasattr(m, "indices") and torch.is_tensor(getattr(m, "indices"))
        and hasattr(m, "theta") and torch.is_tensor(getattr(m, "theta"))
        and hasattr(m, "bias") and torch.is_tensor(getattr(m, "bias"))
        and hasattr(m, "in_features") and hasattr(m, "out_features")
        and not hasattr(m, "weight")
    )

def _is_excluded_by_name(name: str, sic_cfg: dict) -> bool:
    skip = set(sic_cfg.get("exclude_modules_by_name", []))
    return any(name == pat or name.endswith(pat) for pat in skip)

def _row_slice_indexed_sparse(m: nn.Module, r: int) -> Tuple[torch.Tensor, torch.Tensor, Union[int, torch.Tensor], Optional[int]]:
    if hasattr(m, "row_ptr") and torch.is_tensor(getattr(m, "row_ptr")) and m.row_ptr.numel() >= 2:
        s = int(m.row_ptr[r].item())
        e = int(m.row_ptr[r + 1].item())
        cols = m.indices[1, s:e]
        vals = m.theta[s:e]
        return cols, vals, s, e
    row_ids = m.indices[0]
    sel = torch.nonzero(row_ids == r, as_tuple=False).view(-1)
    cols = m.indices[1, sel]
    vals = m.theta.index_select(0, sel)
    return cols, vals, sel, None

def _write_back_indexed_sparse_vals(m: nn.Module, s: Union[int, torch.Tensor], e: Optional[int], vals: torch.Tensor) -> None:
    if e is None:
        m.theta.index_copy_(0, s.to(m.theta.device), vals.to(m.theta.device))
    else:
        m.theta[s:e] = vals.to(m.theta.device)

@torch.inference_mode()
def _sic_core(model: nn.Module, train_loader, device: torch.device, visualize: bool, profiler: Optional[SICProfiler], cfg: Optional[Dict[str, Any]], val_loader=None, mode: str = "classic", calibration_loader=None):
    cfg = cfg or {}
    sic_cfg = cfg.get("sic", {}) or {}
    io_cfg = cfg.get("io", {}) or {}
    sasic_cfg = cfg.get("sasic", {}) or {}

    rounding_decimals = int(sic_cfg.get("rounding_decimals", 6))
    neuron_log_path = io_cfg.get("neuron_log_path", "SIC_per_neuron_times.jsonl")
    log_flush_every = int(io_cfg.get("log_flush_every", 1000))
    enable_neuron_log = bool(io_cfg.get("enable_neuron_log", True))
    enable_autosave = bool(io_cfg.get("enable_autosave_progress", True))
    autosave_path = io_cfg.get("autosave_progress_path", "SIC_Progress_Autosave.json")
    autosave_each = bool(sic_cfg.get("autosave_json_each_layer", True)) and enable_autosave
    eval_max = sic_cfg.get("eval_max", None)
    eval_max = None if eval_max in (None, "null", 0, "0") else int(eval_max)
    max_passes = int(sic_cfg.get("max_passes", 1))
    merge_after = bool(sic_cfg.get("merge_after_clustering", True))
    max_k_per_neuron = int(sic_cfg.get("max_k_per_neuron", 1_000_000))
    progress = bool(sic_cfg.get("progress", True))
    debug_skip = bool(sic_cfg.get("debug_skip", False))
    uwc_exclude_zero = bool(sic_cfg.get("uwc_exclude_zero", True))

    if bool(sic_cfg.get("compile_model", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    uwc_patience = int(sic_cfg.get("uwc_patience", 1))
    uwc_min_rel_delta = float(sic_cfg.get("uwc_min_rel_delta", 0.0))
    stale_uwc = 0
    prev_uwc = _uwc(model, rounding_decimals, exclude_zero=uwc_exclude_zero)

    min_pass_weight_delta = float(sic_cfg.get("min_pass_weight_delta", 0.0))

    def _asc_gate(eval_cap: Optional[int]):
        ok, _ = all_samples_correct(model, filtered_dataset, device, eval_max=eval_cap, profiler=profiler)
        return ok

    tiered = sic_cfg.get("tiered_asc", {}) or {}
    tier_enable = bool(tiered.get("enable", False))
    tier_fast = int(tiered.get("fast", 256))
    tier_med = int(tiered.get("medium", 1024))

    def _asc_progressive():
        if not tier_enable:
            return _asc_gate(eval_max)
        total = len(filtered_dataset)
        if not _asc_gate(min(tier_fast, total)):
            return False
        if not _asc_gate(min(tier_med, total)):
            return False
        return _asc_gate(eval_max)

    profiler = profiler or SICProfiler()
    profiler.start_profiling(memory_tracker=MemoryTracker())
    torch.set_grad_enabled(False)
    model.eval()

    profiler.stats["global"]["original_params"] = sum(p.numel() for p in model.parameters())
    profiler.start_phase("initialization")
    profiler.end_phase("initialization")

    profiler.start_phase("filtering")
    filtered_dataset = build_acceptance_subset_from_loaders(
        model=model,
        train_loader=train_loader,
        device=device,
        val_loader=val_loader,
        from_spec=(cfg.get("sic", {}).get("acceptance_from", "train+val")),
        max_examples=eval_max,
    )
    profiler.stats["phases"]["filtering"]["samples_before"] = None
    profiler.stats["phases"]["filtering"]["samples_after"] = int(len(filtered_dataset))
    profiler.end_phase("filtering")

    # SASIC: Collect activation statistics if enabled
    activation_stats = None
    if calibration_loader is not None and sasic_cfg.get("enabled", False) and sasic_cfg.get("mode") == "active":
        from .activation_stats import collect_activation_stats
        calib_start = perf_counter()
        print("\n[SASIC] Collecting activation statistics from calibration slice...")
        activation_stats = collect_activation_stats(
            model=model,
            loader=calibration_loader,
            device=device,
            sasic_cfg=sasic_cfg,
            sic_cfg=cfg,
        )
        calib_time = perf_counter() - calib_start
        num_batches = len(calibration_loader)
        print(f"[SASIC] Calibration complete: {num_batches} batches, {calib_time:.2f}s")
        if profiler:
            profiler.stats.setdefault("sasic", {})["calibration_time_sec"] = calib_time
            profiler.stats["sasic"]["calibration_batches"] = num_batches

    profiler.start_phase("clustering")
    neuron_log = LogBuffer(neuron_log_path, flush_every=log_flush_every) if enable_neuron_log else None
    total_neurons = 0
    total_success = 0
    initial_layer_flats: Dict[str, np.ndarray] = {}

    for pass_idx in range(max_passes):
        changed_any = False
        pass_t0 = perf_counter()

        w_before = {
            n: m.weight.detach().clone()
            for n, m in model.named_modules()
            if hasattr(m, "weight") and torch.is_tensor(getattr(m, "weight"))
        }
        theta_before = {
            n: m.theta.detach().clone()
            for n, m in model.named_modules()
            if _is_indexed_sparse_linear(m)
        }

        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue
            if _is_excluded_by_name(name, sic_cfg):
                if debug_skip:
                    print(f"[SIC] Skipping by name: {name}")
                continue
            if is_pos_linear(module) or is_pos_conv2d(module):
                if debug_skip:
                    print(f"[SIC] Skipping PoS/fast-PoS (no weight): {name}")
                continue

            if hasattr(module, "weight") and torch.is_tensor(getattr(module, "weight")):
                layer_t0 = perf_counter()
                w = module.weight
                shp = tuple(w.shape)
                flat = _flatten_weight(w)

                if pass_idx == 0:
                    before_flat_cpu = flat.detach().cpu().numpy()
                    initial_layer_flats[name] = before_flat_cpu.copy()
                    total_neurons += flat.size(0)
                    profiler.init_layer_stats(name, shp, int(w.numel()))
                    profiler.record_weight_distribution(name, before_flat_cpu, before_flat_cpu)

                best_flat = flat.clone()
                success_count_this_layer = 0
                num_neurons = flat.size(0)
                
                # SASIC: Initialize per-layer active input stats accumulation (only on first pass)
                sasic_layer_active = (
                    pass_idx == 0
                    and sasic_cfg.get("enabled", False)
                    and sasic_cfg.get("mode") == "active"
                    and activation_stats is not None
                )
                sasic_layer_stats = None
                if sasic_layer_active:
                    sasic_layer_stats = {
                        "num_inputs_list": [],
                        "num_active_list": [],
                        "active_frac_list": [],
                    }
                pos_clusters_for_layer = getattr(module, "_sic_pos_clusters", [[] for _ in range(num_neurons)])
                if len(pos_clusters_for_layer) != num_neurons:
                    pos_clusters_for_layer = [[] for _ in range(num_neurons)]

                freeze_patience = int(sic_cfg.get("neuron_patience", 0))
                no_change = getattr(module, "_sic_no_change", [0] * num_neurons)
                if len(no_change) != num_neurons:
                    no_change = [0] * num_neurons

                use_gpu = (w.device.type == "cuda") and _should_gpu_rowlen(flat.size(1), sic_cfg) if mode == "hybrid" else False

                for idx in range(num_neurons):
                    if freeze_patience > 0 and no_change[idx] >= freeze_patience:
                        profiler.record_neuron_attempt(name, idx, 0, False, "frozen_by_patience")
                        continue

                    orig_t = flat[idx].clone()
                    orig_np = orig_t.detach().cpu().numpy()
                    nz = orig_np != 0
                    vals = orig_np[nz]
                    if vals.size == 0:
                        profiler.record_neuron_attempt(name, idx, 0, True, "all_zero_weights")
                        if neuron_log:
                            neuron_log.add({"layer": name, "neuron_idx": idx})
                        no_change[idx] = 0
                        continue

                    converged = False
                    uniq_nz = int(np.unique(vals).size)
                    max_k = max(1, min(uniq_nz - 1, max_k_per_neuron))

                    flat[idx] = torch.zeros_like(orig_t)
                    ok = _asc_progressive()
                    if ok:
                        best_flat[idx] = flat[idx]
                        success_count_this_layer += 1
                        total_success += 1
                        profiler.record_neuron_attempt(name, idx, 0, True, "zeroed_perceptron")
                        pos_clusters_for_layer[idx] = []
                        changed_any = True
                        converged = True
                        no_change[idx] = 0
                    else:
                        flat[idx] = orig_t
                        profiler.record_neuron_attempt(name, idx, 0, False, "zero_rejected")

                    # SASIC Mode A — Active-Subset SASIC (sasic_design.md §5.1)
                    # Mode A is active for this neuron if and only if ALL of the following are true:
                    # 1. SASIC is enabled and mode is "active"
                    # 2. Global activation_stats exist and contain data for this layer/neuron
                    # 3. We can build an active_mask_full with length exactly equal to len(orig_np)
                    # 4. The active set A is "non-trivial": at least 2 active inputs AND
                    #    active fraction is neither ~0% nor ~100% (optimization: skip if >95% active)
                    # If any condition fails, Mode A is disabled for this neuron and we use pure baseline.
                    # (sasic_design.md §10: must preserve baseline behavior when disabled)
                    sasic_active = (
                        sasic_cfg.get("enabled", False)
                        and sasic_cfg.get("mode") == "active"
                        and activation_stats is not None
                    )
                    
                    active_mask_full = None
                    if sasic_active:
                        from .sasic_utils import get_layer_key_for_sasic, get_active_indices_for_neuron
                        layer_key = get_layer_key_for_sasic(name, module)
                        # activation_threshold maps to tau_active in design doc (§5.1, §7.2)
                        activation_threshold = float(sasic_cfg.get("activation_threshold", 0.01))
                        # Pass ground-truth num_inputs to ensure mask length matches weight vector
                        # (sasic_design.md §4.3: stats must align with flattened weight indices)
                        active_mask_full = get_active_indices_for_neuron(
                            activation_stats, layer_key, idx, activation_threshold, num_inputs=len(orig_np)
                        )
                        
                        # Conservative fallback conditions (sasic_design.md §10: must preserve baseline when disabled)
                        # All checks must pass; if any fails, abort Mode A entirely for this neuron.
                        if active_mask_full is None:
                            # Stats missing or inconsistent
                            sasic_active = False
                            active_mask_full = None
                        elif active_mask_full.shape[0] != len(orig_np):
                            # Hard requirement: mask length must match weight vector length
                            sasic_active = False
                            active_mask_full = None
                        else:
                            num_active = int(np.sum(active_mask_full))
                            active_frac = float(num_active / len(orig_np)) if len(orig_np) > 0 else 0.0
                            
                            # Fallback if active set is too small (< 2 active inputs)
                            # (Design doc §5.1 requires clustering on active inputs; need at least 2 for Jenks)
                            if num_active < 2:
                                sasic_active = False
                                active_mask_full = None
                            # Optimization: skip SASIC if almost all inputs are active (>95%)
                            # This avoids overhead when Mode A provides little benefit
                            elif active_frac > 0.95:
                                sasic_active = False
                                active_mask_full = None
                            # Fallback if no active inputs (should be caught above, but double-check)
                            elif num_active == 0:
                                sasic_active = False
                                active_mask_full = None
                    
                    # SASIC: Accumulate active input stats for this neuron (only on first pass)
                    if sasic_layer_active and active_mask_full is not None and active_mask_full.shape[0] == len(orig_np):
                        try:
                            num_inputs = len(orig_np)
                            num_active = int(np.sum(active_mask_full))
                            active_frac = float(num_active / num_inputs) if num_inputs > 0 else 0.0
                            sasic_layer_stats["num_inputs_list"].append(num_inputs)
                            sasic_layer_stats["num_active_list"].append(num_active)
                            sasic_layer_stats["active_frac_list"].append(active_frac)
                        except Exception:
                            # Skip logging for this neuron if anything goes wrong
                            pass

                    # Track k-trials for this neuron (for profiler stats)
                    # Count every k evaluation regardless of accept/reject, baseline vs SASIC
                    k_trials_this_neuron = 0
                    
                    for k in range(1, max_k + 1):
                        if converged:
                            break
                        if progress:
                            print(
                                f"[SIC-{mode}] pass {pass_idx+1}/{max_passes} | {name} n={idx+1}/{num_neurons} k={k}    ",
                                end="\r",
                                flush=True,
                            )

                        # Count this k evaluation (regardless of accept/reject, baseline vs SASIC)
                        # (sasic_design.md §9: track number of k values attempted per neuron)
                        k_trials_this_neuron += 1
                        
                        # SASIC Mode A clustering (sasic_design.md §5.1, step 3):
                        # Run Jenks clustering on {w_i : i in A} (active inputs only)
                        # Implementation note: we cluster active nonzero inputs, then handle
                        # active zeros separately by assigning to nearest cluster center.
                        # (Design doc §5.1 is flexible on active zero handling; this is our implementation choice)
                        if sasic_active and active_mask_full is not None:
                            # Get active mask for nonzero entries
                            active_mask_nz = active_mask_full[nz]
                            num_active_nz = int(np.sum(active_mask_nz))
                            
                            # Conservative check: need at least 2 active nonzero for clustering
                            # If this fails, abort Mode A entirely for this neuron (pure baseline)
                            # (sasic_design.md §10: must preserve baseline behavior when disabled)
                            if num_active_nz < 2:
                                # Fallback: abort Mode A, use pure baseline for this neuron
                                # This ensures no "half SASIC, half baseline" behavior
                                sasic_active = False
                                active_mask_full = None
                                breaks = _jenks_breaks(vals, k)
                            else:
                                # Extract active nonzero values and run Jenks on active subset
                                # (sasic_design.md §5.1, step 3: cluster weights of active inputs)
                                vals_active = vals[active_mask_nz]
                                breaks = _jenks_breaks(vals_active, k)
                        else:
                            # Baseline: run Jenks on all nonzero values (sasic_design.md §2)
                            breaks = _jenks_breaks(vals, k)

                        if use_gpu:
                            # GPU path
                            # SASIC Mode A consolidation (sasic_design.md §5.1, steps 3-4):
                            # - Active inputs assigned to cluster centers from Jenks
                            # - Quiet inputs attached to nearest cluster center by weight distance
                            if sasic_active and active_mask_full is not None:
                                # Re-check active nonzero count (may have changed in k-loop)
                                active_mask_nz = active_mask_full[nz]
                                num_active_nz = int(np.sum(active_mask_nz))
                                
                                if num_active_nz >= 2:
                                    # SASIC Mode A: Convert to numpy for SASIC logic, then convert back
                                    from .sasic_utils import attach_quiet_inputs_to_clusters
                                    # Extract active nonzero values
                                    vals_active = vals[active_mask_nz]
                                    
                                    # Get indices of active nonzero inputs in the full vector
                                    active_nz_indices_full = np.where(active_mask_full & nz)[0]
                                    
                                    # Compute cluster centers and assignments for active inputs
                                    cluster_centers = np.zeros(len(breaks) - 1)
                                    cluster_assignments_active = np.zeros(len(vals_active), dtype=np.int32)
                                    
                                    for i_c in range(len(breaks) - 1):
                                        lo, hi = breaks[i_c], breaks[i_c + 1]
                                        if i_c == 0:
                                            sel_active = (vals_active >= lo) & (vals_active <= hi)
                                        else:
                                            sel_active = (vals_active > lo) & (vals_active <= hi)
                                        
                                        if np.any(sel_active):
                                            group = vals_active[sel_active]
                                            avg = float(np.mean(group)) if group.size else float((lo + hi) / 2)
                                            cluster_centers[i_c] = avg
                                            cluster_assignments_active[sel_active] = i_c
                                    
                                    # Build cluster assignments for all active inputs
                                    active_indices_full = np.where(active_mask_full)[0]
                                    cluster_assignments_full = np.zeros(len(active_indices_full), dtype=np.int32)
                                    
                                    # Map assignments from active nonzero to active full
                                    for i, nz_idx in enumerate(active_nz_indices_full):
                                        pos = np.where(active_indices_full == nz_idx)[0]
                                        if len(pos) > 0:
                                            cluster_assignments_full[pos[0]] = cluster_assignments_active[i]
                                    
                                    # For active zero inputs, assign to nearest cluster center by distance from 0.0
                                    # (Design doc §5.1 is flexible on active zero handling; this is our implementation choice)
                                    active_zero_mask = active_mask_full & ~nz
                                    active_zero_indices = np.where(active_zero_mask)[0]
                                    for zero_idx in active_zero_indices:
                                        pos = np.where(active_indices_full == zero_idx)[0]
                                        if len(pos) > 0:
                                            distances = np.abs(cluster_centers - 0.0)
                                            nearest_cluster = int(np.argmin(distances))
                                            cluster_assignments_full[pos[0]] = nearest_cluster
                                    
                                    # Build full consolidated vector using SASIC attachment
                                    # (sasic_design.md §5.1, step 4: attach quiet inputs to nearest cluster center)
                                    new_flat_np = attach_quiet_inputs_to_clusters(
                                        weights_full=orig_np,
                                        active_mask=active_mask_full,
                                        cluster_centers=cluster_centers,
                                        cluster_assignments=cluster_assignments_full,
                                    )
                                    
                                    # Convert back to torch tensor
                                    trial_t = torch.from_numpy(new_flat_np).to(orig_t.device, dtype=module.weight.dtype)
                                    flat[idx] = trial_t
                                    
                                    # Build row_pos for PoS
                                    row_pos: List[Tuple[torch.Tensor, float]] = []
                                    for i_c in range(len(breaks) - 1):
                                        sel_mask = (cluster_assignments_active == i_c)
                                        if np.any(sel_mask):
                                            idxs = active_nz_indices_full[sel_mask]
                                            row_pos.append((torch.tensor(idxs, dtype=torch.long), float(cluster_centers[i_c])))
                                else:
                                    # Fallback: not enough active inputs, abort Mode A and use pure baseline GPU path
                                    # (sasic_design.md §10: must preserve baseline behavior when disabled)
                                    # This ensures no "half SASIC, half baseline" behavior
                                    sasic_active = False
                                    active_mask_full = None
                                    v = orig_t.clone()
                                    mask = v != 0
                                    vv = v[mask]
                                    b = torch.tensor(breaks, device=v.device, dtype=v.dtype)
                                    bin_ids = torch.clamp(torch.bucketize(vv, b) - 1, 0, b.numel() - 2)
                                    G = b.numel() - 1
                                    means = torch.zeros(G, device=v.device, dtype=v.dtype)
                                    counts = torch.zeros_like(means)
                                    means.index_add_(0, bin_ids, vv)
                                    counts.index_add_(0, bin_ids, torch.ones_like(vv))
                                    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
                                    means = means / counts
                                    vv_repl = means[bin_ids]
                                    trial_t = orig_t.clone()
                                    trial_t[mask] = vv_repl
                                    flat[idx] = trial_t
                                    row_pos: List[Tuple[torch.Tensor, float]] = []
                                    sel_idx = torch.nonzero(mask, as_tuple=False).view(-1).cpu()
                                    bin_ids_cpu = bin_ids.detach().cpu()
                                    means_cpu = means.detach().cpu()
                                    for g in range(G):
                                        g_mask = (bin_ids_cpu == g)
                                        if int(g_mask.sum().item()) == 0:
                                            continue
                                        idxs_cpu = sel_idx[g_mask]
                                        row_pos.append((idxs_cpu, float(means_cpu[g].item())))
                            else:
                                # Baseline GPU path
                                v = orig_t.clone()
                                mask = v != 0
                                vv = v[mask]
                                b = torch.tensor(breaks, device=v.device, dtype=v.dtype)
                                bin_ids = torch.clamp(torch.bucketize(vv, b) - 1, 0, b.numel() - 2)
                                G = b.numel() - 1
                                means = torch.zeros(G, device=v.device, dtype=v.dtype)
                                counts = torch.zeros_like(means)
                                means.index_add_(0, bin_ids, vv)
                                counts.index_add_(0, bin_ids, torch.ones_like(vv))
                                counts = torch.where(counts == 0, torch.ones_like(counts), counts)
                                means = means / counts
                                vv_repl = means[bin_ids]
                                trial_t = orig_t.clone()
                                trial_t[mask] = vv_repl
                                flat[idx] = trial_t
                                row_pos: List[Tuple[torch.Tensor, float]] = []
                                sel_idx = torch.nonzero(mask, as_tuple=False).view(-1).cpu()
                                bin_ids_cpu = bin_ids.detach().cpu()
                                means_cpu = means.detach().cpu()
                                for g in range(G):
                                    g_mask = (bin_ids_cpu == g)
                                    if int(g_mask.sum().item()) == 0:
                                        continue
                                    idxs_cpu = sel_idx[g_mask]
                                    row_pos.append((idxs_cpu, float(means_cpu[g].item())))
                        else:
                            # CPU path
                            # SASIC Mode A consolidation (sasic_design.md §5.1, steps 3-4)
                            if sasic_active and active_mask_full is not None:
                                # Re-check active nonzero count (may have changed in k-loop)
                                active_mask_nz = active_mask_full[nz]
                                num_active_nz = int(np.sum(active_mask_nz))
                                
                                if num_active_nz >= 2:
                                    # SASIC Mode A: Cluster active inputs, attach quiet inputs
                                    from .sasic_utils import attach_quiet_inputs_to_clusters
                                    # Extract active nonzero values
                                    vals_active = vals[active_mask_nz]
                                    
                                    # Get indices of active nonzero inputs in the full vector
                                    active_nz_indices_full = np.where(active_mask_full & nz)[0]
                                    
                                    # Compute cluster centers and assignments for active inputs
                                    cluster_centers = np.zeros(len(breaks) - 1)
                                    cluster_assignments_active = np.zeros(len(vals_active), dtype=np.int32)
                                    
                                    for i_c in range(len(breaks) - 1):
                                        lo, hi = breaks[i_c], breaks[i_c + 1]
                                        if i_c == 0:
                                            sel_active = (vals_active >= lo) & (vals_active <= hi)
                                        else:
                                            sel_active = (vals_active > lo) & (vals_active <= hi)
                                        
                                        if np.any(sel_active):
                                            group = vals_active[sel_active]
                                            avg = float(np.mean(group)) if group.size else float((lo + hi) / 2)
                                            cluster_centers[i_c] = avg
                                            cluster_assignments_active[sel_active] = i_c
                                    
                                    # Build cluster assignments for all active inputs (nonzero only, since we clustered nonzero)
                                    # active_nz_indices_full contains the full vector indices of active nonzero inputs
                                    # cluster_assignments_active[i] is the cluster for the i-th active nonzero value
                                    # We need cluster_assignments_full where cluster_assignments_full[j] is the cluster
                                    # for the j-th active input in the full vector
                                    
                                    active_indices_full = np.where(active_mask_full)[0]
                                    cluster_assignments_full = np.zeros(len(active_indices_full), dtype=np.int32)
                                    
                                    # Create mapping: for each active nonzero input, find its position in active_indices_full
                                    for i, nz_idx in enumerate(active_nz_indices_full):
                                        pos = np.where(active_indices_full == nz_idx)[0]
                                        if len(pos) > 0:
                                            cluster_assignments_full[pos[0]] = cluster_assignments_active[i]
                                    
                                    # For active zero inputs, assign to nearest cluster center by distance from 0.0
                                    # (Design doc §5.1 is flexible on active zero handling; this is our implementation choice)
                                    active_zero_mask = active_mask_full & ~nz
                                    active_zero_indices = np.where(active_zero_mask)[0]
                                    for zero_idx in active_zero_indices:
                                        pos = np.where(active_indices_full == zero_idx)[0]
                                        if len(pos) > 0:
                                            distances = np.abs(cluster_centers - 0.0)
                                            nearest_cluster = int(np.argmin(distances))
                                            cluster_assignments_full[pos[0]] = nearest_cluster
                                    
                                    # Build full consolidated vector using SASIC attachment
                                    # (sasic_design.md §5.1, step 4: attach quiet inputs to nearest cluster center)
                                    new_flat = attach_quiet_inputs_to_clusters(
                                        weights_full=orig_np,
                                        active_mask=active_mask_full,
                                        cluster_centers=cluster_centers,
                                        cluster_assignments=cluster_assignments_full,
                                    )
                                    
                                    # Build row_pos for PoS (only active inputs that were clustered)
                                    row_pos = []
                                    for i_c in range(len(breaks) - 1):
                                        sel_mask = (cluster_assignments_active == i_c)
                                        if np.any(sel_mask):
                                            idxs = active_nz_indices_full[sel_mask]
                                            row_pos.append((torch.tensor(idxs, dtype=torch.long), float(cluster_centers[i_c])))
                                else:
                                    # Fallback: not enough active inputs, abort Mode A and use pure baseline
                                    # (sasic_design.md §10: must preserve baseline behavior when disabled)
                                    # This ensures no "half SASIC, half baseline" behavior
                                    sasic_active = False
                                    active_mask_full = None
                                    new_flat = orig_np.copy()
                                    row_pos = []
                                    for i_c in range(len(breaks) - 1):
                                        lo, hi = breaks[i_c], breaks[i_c + 1]
                                        if i_c == 0:
                                            sel = (orig_np >= lo) & (orig_np <= hi) & nz
                                            group = vals[(vals >= lo) & (vals <= hi)]
                                        else:
                                            sel = (orig_np > lo) & (orig_np <= hi) & nz
                                            group = vals[(vals > lo) & (vals <= hi)]
                                        if np.any(sel):
                                            avg = float(np.mean(group)) if group.size else float((lo + hi) / 2)
                                            new_flat[sel] = avg
                                            idxs = np.where(sel)[0]
                                            row_pos.append((torch.tensor(idxs, dtype=torch.long), float(avg)))
                            else:
                                # Baseline: standard consolidation
                                new_flat = orig_np.copy()
                                row_pos = []
                                for i_c in range(len(breaks) - 1):
                                    lo, hi = breaks[i_c], breaks[i_c + 1]
                                    if i_c == 0:
                                        sel = (orig_np >= lo) & (orig_np <= hi) & nz
                                        group = vals[(vals >= lo) & (vals <= hi)]
                                    else:
                                        sel = (orig_np > lo) & (orig_np <= hi) & nz
                                        group = vals[(vals > lo) & (vals <= hi)]
                                    if np.any(sel):
                                        avg = float(np.mean(group)) if group.size else float((lo + hi) / 2)
                                        new_flat[sel] = avg
                                        idxs = np.where(sel)[0]
                                        row_pos.append((torch.tensor(idxs, dtype=torch.long), float(avg)))
                            
                            trial_t = torch.from_numpy(new_flat).to(orig_t.device, dtype=module.weight.dtype)
                            flat[idx] = trial_t

                        ok = _asc_progressive()
                        if ok:
                            best_flat[idx] = flat[idx]
                            success_count_this_layer += 1
                            total_success += 1
                            if row_pos is not None:
                                pos_clusters_for_layer[idx] = row_pos
                            profiler.record_neuron_attempt(name, idx, k, True, "converged")
                            changed_any = True
                            converged = True
                            no_change[idx] = 0
                            break
                        else:
                            profiler.record_neuron_attempt(name, idx, k, False, "accuracy_loss")
                            flat[idx] = orig_t

                    if not converged:
                        profiler.record_neuron_attempt(name, idx, max_k, False, "max_clusters_exceeded")
                        no_change[idx] = no_change[idx] + 1
                    
                    # Record k-trials for this neuron (for profiler stats)
                    # Track both per-layer and global totals (sasic_design.md §9: metrics and logging)
                    if name not in profiler.stats["layers"]:
                        profiler.stats["layers"][name] = {}
                    layer_stats = profiler.stats["layers"][name]
                    if "k_trials_list" not in layer_stats:
                        layer_stats["k_trials_list"] = []
                    layer_stats["k_trials_list"].append(k_trials_this_neuron)
                    
                    # Initialize global k-trials counters if needed
                    if "sic" not in profiler.stats:
                        profiler.stats["sic"] = {}
                    if "total_k_trials" not in profiler.stats["sic"]:
                        profiler.stats["sic"]["total_k_trials"] = 0
                    if "num_neurons_evaluated" not in profiler.stats["sic"]:
                        profiler.stats["sic"]["num_neurons_evaluated"] = 0
                    
                    profiler.stats["sic"]["total_k_trials"] += k_trials_this_neuron
                    profiler.stats["sic"]["num_neurons_evaluated"] += 1

                with torch.no_grad():
                    module.weight.copy_(best_flat.view_as(w) if w.ndim > 2 else best_flat)
                setattr(module, "_sic_pos_clusters", pos_clusters_for_layer)
                setattr(module, "_sic_no_change", no_change)
                
                # SASIC: Store per-layer active input stats in profiler (only on first pass)
                if sasic_layer_active and sasic_layer_stats is not None:
                    try:
                        if sasic_layer_stats["num_inputs_list"]:
                            from .sasic_utils import get_layer_key_for_sasic
                            layer_key = get_layer_key_for_sasic(name, module)
                            num_inputs = sasic_layer_stats["num_inputs_list"][0]  # Should be same for all neurons
                            avg_active = float(np.mean(sasic_layer_stats["num_active_list"]))
                            avg_active_frac = float(np.mean(sasic_layer_stats["active_frac_list"]))
                            
                            profiler.stats.setdefault("sasic", {}).setdefault("layers", {})[layer_key] = {
                                "num_inputs": int(num_inputs),
                                "avg_active": avg_active,
                                "avg_active_frac": avg_active_frac,
                            }
                    except Exception:
                        # Skip logging for this layer if anything goes wrong
                        pass
                
                # Compute per-layer k-trials statistics (sasic_design.md §9: per-layer metrics)
                if name in profiler.stats["layers"]:
                    layer_stats = profiler.stats["layers"][name]
                    if "k_trials_list" in layer_stats and layer_stats["k_trials_list"]:
                        k_trials_list = layer_stats["k_trials_list"]
                        layer_stats["avg_k_trials_per_neuron"] = float(np.mean(k_trials_list))
                        layer_stats["total_k_trials"] = int(sum(k_trials_list))
                        # Keep list for detailed analysis, but also store aggregate
                
                if autosave_each and autosave_path:
                    profiler.save_detailed_stats(autosave_path)
                profiler.record_layer_processing(name, perf_counter() - layer_t0, success_count_this_layer, num_neurons)
                if progress:
                    print(" " * 120, end="\r")
                continue

            if _is_indexed_sparse_linear(module):
                layer_t0 = perf_counter()
                K = int(module.out_features)
                D = int(module.in_features)

                if pass_idx == 0:
                    before_cpu = module.theta.detach().view(-1).cpu().numpy()
                    initial_layer_flats[name] = before_cpu.copy()
                    total_neurons += K
                    profiler.init_layer_stats(name, (K, D), int(module.theta.numel()))
                    profiler.record_weight_distribution(name, before_cpu, before_cpu)

                success_count_this_layer = 0
                pos_clusters_for_layer = getattr(module, "_sic_pos_clusters", [[] for _ in range(K)])
                if len(pos_clusters_for_layer) != K:
                    pos_clusters_for_layer = [[] for _ in range(K)]

                freeze_patience = int(sic_cfg.get("neuron_patience", 0))
                no_change = getattr(module, "_sic_no_change", [0] * K)
                if len(no_change) != K:
                    no_change = [0] * K

                avg_row_len = (module.theta.numel() // max(1, K))
                use_gpu = (module.theta.device.type == "cuda") and _should_gpu_rowlen(max(D, avg_row_len), sic_cfg) if mode == "hybrid" else False

                for r in range(K):
                    if freeze_patience > 0 and no_change[r] >= freeze_patience:
                        profiler.record_neuron_attempt(name, r, 0, False, "frozen_by_patience")
                        continue

                    cols, vals_view, s_or_idx, e = _row_slice_indexed_sparse(module, r)
                    if vals_view.numel() == 0:
                        profiler.record_neuron_attempt(name, r, 0, True, "all_zero_edges")
                        if neuron_log:
                            neuron_log.add({"layer": name, "neuron_idx": r})
                        no_change[r] = 0
                        continue

                    orig_vals = vals_view.detach().clone()
                    orig_np = orig_vals.detach().cpu().numpy()
                    nz_vals = orig_np
                    uniq_nz = int(np.unique(nz_vals).size)
                    max_k = max(1, min(uniq_nz - 1, max_k_per_neuron))
                    converged = False

                    _write_back_indexed_sparse_vals(module, s_or_idx, e, torch.zeros_like(orig_vals))
                    ok = _asc_progressive()
                    if ok:
                        success_count_this_layer += 1
                        total_success += 1
                        profiler.record_neuron_attempt(name, r, 0, True, "zeroed_perceptron")
                        pos_clusters_for_layer[r] = []
                        changed_any = True
                        converged = True
                        no_change[r] = 0
                    else:
                        _write_back_indexed_sparse_vals(module, s_or_idx, e, orig_vals)
                        profiler.record_neuron_attempt(name, r, 0, False, "zero_rejected")

                    for k in range(1, max_k + 1):
                        if converged:
                            break
                        if progress:
                            print(
                                f"[SIC-{mode}] pass {pass_idx+1}/{max_passes} | {name} row={r+1}/{K} k={k}    ",
                                end="\r",
                                flush=True,
                            )

                        breaks = _jenks_breaks(nz_vals, k)

                        if use_gpu:
                            v = orig_vals.clone().to(module.theta.device)
                            b = torch.tensor(breaks, device=v.device, dtype=v.dtype)
                            bin_ids = torch.clamp(torch.bucketize(v, b) - 1, 0, b.numel() - 2)
                            G = b.numel() - 1
                            means = torch.zeros(G, device=v.device, dtype=v.dtype)
                            counts = torch.zeros_like(means)
                            means.index_add_(0, bin_ids, v)
                            counts.index_add_(0, bin_ids, torch.ones_like(v))
                            counts = torch.where(counts == 0, torch.ones_like(counts), counts)
                            means = means / counts
                            trial_vals = means[bin_ids]
                            _write_back_indexed_sparse_vals(module, s_or_idx, e, trial_vals)
                            row_pos = []
                            bin_ids_cpu = bin_ids.detach().cpu()
                            means_cpu = means.detach().cpu()
                            cols_cpu = cols.detach().cpu()
                            for g in range(G):
                                g_mask = (bin_ids_cpu == g)
                                if int(g_mask.sum().item()) == 0:
                                    continue
                                idxs_cpu = cols_cpu[g_mask]
                                row_pos.append((idxs_cpu, float(means_cpu[g].item())))
                        else:
                            new_vals = orig_np.copy()
                            row_pos = []
                            for i_c in range(len(breaks) - 1):
                                lo, hi = breaks[i_c], breaks[i_c + 1]
                                if i_c == 0:
                                    sel = (orig_np >= lo) & (orig_np <= hi)
                                    group = orig_np[(orig_np >= lo) & (orig_np <= hi)]
                                else:
                                    sel = (orig_np > lo) & (orig_np <= hi)
                                    group = orig_np[(orig_np > lo) & (orig_np <= hi)]
                                if np.any(sel):
                                    avg = float(np.mean(group)) if group.size else float((lo + hi) / 2)
                                    new_vals[sel] = avg
                                    idxs = cols.detach().cpu().numpy()[sel]
                                    row_pos.append((torch.tensor(idxs, dtype=torch.long), float(avg)))
                            trial_vals = torch.from_numpy(new_vals).to(module.theta.device, dtype=module.theta.dtype)
                            _write_back_indexed_sparse_vals(module, s_or_idx, e, trial_vals)

                        ok = _asc_progressive()
                        if ok:
                            success_count_this_layer += 1
                            total_success += 1
                            if row_pos is not None:
                                pos_clusters_for_layer[r] = row_pos
                            profiler.record_neuron_attempt(name, r, k, True, "converged")
                            changed_any = True
                            converged = True
                            no_change[r] = 0
                            break
                        else:
                            profiler.record_neuron_attempt(name, r, k, False, "accuracy_loss")
                            _write_back_indexed_sparse_vals(module, s_or_idx, e, orig_vals)

                    if not converged:
                        profiler.record_neuron_attempt(name, r, max_k, False, "max_clusters_exceeded")
                        no_change[r] = no_change[r] + 1
                        _write_back_indexed_sparse_vals(module, s_or_idx, e, orig_vals)

                after_cpu = module.theta.detach().view(-1).cpu().numpy()
                profiler.record_weight_distribution(name, initial_layer_flats[name], after_cpu)
                setattr(module, "_sic_pos_clusters", pos_clusters_for_layer)
                setattr(module, "_sic_no_change", no_change)
                if autosave_each and autosave_path:
                    profiler.save_detailed_stats(autosave_path)
                profiler.record_layer_processing(name, perf_counter() - layer_t0, success_count_this_layer, K)
                if progress:
                    print(" " * 120, end="\r")
                continue

            if debug_skip:
                print(f"[SIC] Skipping unsupported module: {name}")

        max_abs = 0.0
        for n, m in model.named_modules():
            if hasattr(m, "weight") and torch.is_tensor(getattr(m, "weight")):
                d = (m.weight - w_before[n]).abs().max().item()
                max_abs = max(max_abs, d)
        for n, m in model.named_modules():
            if _is_indexed_sparse_linear(m):
                d = (m.theta - theta_before[n]).abs().max().item()
                max_abs = max(max_abs, d)

        stop_by_small_delta = max_abs < min_pass_weight_delta

        curr_uwc = _uwc(model, rounding_decimals, exclude_zero=uwc_exclude_zero)
        denom = max(1.0, float(prev_uwc))
        uwc_rel_drop = max(0.0, (float(prev_uwc) - float(curr_uwc))) / denom

        rec = {
            "pass_index": pass_idx + 1,
            "changed_any": bool(changed_any),
            "duration_sec": round(perf_counter() - pass_t0, 6),
            "unique_weight_count": int(curr_uwc),
            "uwc_rel_drop": float(uwc_rel_drop),
            "max_abs_weight_delta": float(max_abs),
        }

        if (curr_uwc == prev_uwc) or (uwc_rel_drop < uwc_min_rel_delta):
            stale_uwc += 1
        else:
            stale_uwc = 0
        stop_by_uwc = stale_uwc >= uwc_patience

        prev_uwc = curr_uwc
        profiler.stats.setdefault("passes", []).append(rec)

        if (not changed_any) or stop_by_uwc or stop_by_small_delta:
            break

    for name, module in model.named_modules():
        if _is_indexed_sparse_linear(module) and (name in initial_layer_flats):
            after_cpu = module.theta.detach().view(-1).cpu().numpy()
            profiler.record_weight_distribution(name, initial_layer_flats[name], after_cpu)

    if neuron_log:
        neuron_log.flush()

    profiler.stats["phases"]["clustering"]["total_neurons"] = total_neurons
    profiler.stats["phases"]["clustering"]["successful_neurons"] = total_success
    profiler.end_phase("clustering")

    if merge_after:
        profiler.start_phase("merging")
        merges = 0
        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, "weight") and torch.is_tensor(getattr(module, "weight")):
                    w = module.weight
                    flat_t = _flatten_weight(w).clone()
                    rd = clamp_decimals(rounding_decimals)
                    rnd = round_tensor(flat_t, rd)
                    nz = flat_t != 0
                    for i in range(flat_t.size(0)):
                        row = flat_t[i]
                        rr = rnd[i]
                        mask = nz[i]
                        if not mask.any():
                            continue
                        for u in torch.unique(rr[mask]):
                            idxs = (rr == u) & mask
                            if int(idxs.sum().item()) > 1:
                                row[idxs] = row[idxs].mean()
                                merges += 1
                    module.weight.copy_(flat_t.view_as(w) if w.ndim > 2 else flat_t)
            for name, module in model.named_modules():
                if not _is_indexed_sparse_linear(module):
                    continue
                K = int(module.out_features)
                rd = clamp_decimals(rounding_decimals)
                for r in range(K):
                    cols, vals_view, s_or_idx, e = _row_slice_indexed_sparse(module, r)
                    if vals_view.numel() == 0:
                        continue
                    v = vals_view.clone()
                    vr = round_tensor(v, rd)
                    uniq = torch.unique(vr)
                    new_vals = v.clone()
                    for u in uniq:
                        mask = (vr == u)
                        if int(mask.sum().item()) > 1:
                            new_vals[mask] = v[mask].mean()
                            merges += 1
                    _write_back_indexed_sparse_vals(module, s_or_idx, e, new_vals)
        profiler.stats["phases"]["merging"]["merges_performed"] = merges
        profiler.end_phase("merging")

    profiler.start_phase("verification")
    profiler.end_phase("verification")
    profiler.end_profiling()
    torch.set_grad_enabled(True)
    return model, profiler.stats

@torch.inference_mode()
def SIC_hybrid(model: nn.Module, train_loader, device: torch.device, visualize: bool = False, profiler: Optional[SICProfiler] = None, cfg: Optional[Dict[str, Any]] = None, val_loader=None, calibration_loader=None):
    return _sic_core(model, train_loader, device, visualize, profiler, cfg, val_loader, mode="hybrid", calibration_loader=calibration_loader)

@torch.inference_mode()
def SIC(model: nn.Module, train_loader, device: torch.device, visualize: bool = False, profiler: Optional[SICProfiler] = None, cfg: Optional[Dict[str, Any]] = None, val_loader=None, calibration_loader=None):
    return _sic_core(model, train_loader, device, visualize, profiler, cfg, val_loader, mode="classic", calibration_loader=calibration_loader)
