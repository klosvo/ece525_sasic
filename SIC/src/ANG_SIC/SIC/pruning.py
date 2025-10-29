from __future__ import annotations
from time import perf_counter
from typing import Optional, Tuple, List, Dict, Any
import copy
import hashlib
import torch
import torch.nn as nn
from .profiling import SICProfiler
from .sic_utils import all_samples_correct
from .pos_common import rows_container_name, is_pos_linear, is_pos_conv2d, dense_from_pos
from src.ANG_SIC.core.acceptance import build_acceptance_subset_from_loaders

def _dataset_is_empty(ds) -> bool:
    if ds is None:
        return True
    if hasattr(ds, "__len__"):
        return len(ds) == 0
    return not bool(ds)

@torch.inference_mode()
def magnitude_prune_per_neuron_sequential(
    model: nn.Module,
    filtered_dataset,
    device: torch.device,
    eval_max: Optional[int] = None,
    include_bias: bool = False,
    profiler: Optional[SICProfiler] = None,
) -> bool:
    if _dataset_is_empty(filtered_dataset):
        print("[PRUNE] Skipping: acceptance set empty.")
        return False
    changed_any = False
    if profiler is not None:
        profiler.start_phase("pruning")
    t0_all = perf_counter()
    model.eval()
    for name, module in list(model.named_modules()):
        if is_pos_linear(module) or is_pos_conv2d(module):
            c = magnitude_prune_pos_sequential(
                module, model, filtered_dataset, device, eval_max=eval_max, include_bias=include_bias, profiler=profiler
            )
            changed_any = changed_any or c
            continue
        W = getattr(module, "weight", None)
        if W is None:
            continue
        with torch.no_grad():
            w = W.detach().clone()
        flat = w.view(w.size(0), -1) if w.ndim > 2 else w.clone()
        out_dim = flat.size(0)
        b = getattr(module, "bias", None)
        if include_bias and (b is not None):
            bval = b.detach().clone()
        for out_idx in range(out_dim):
            row = flat[out_idx]
            for j in torch.argsort(torch.abs(row), descending=False).tolist():
                if row[j].item() == 0.0:
                    continue
                prev = row[j].item()
                row[j] = 0.0
                with torch.no_grad():
                    (W.view(out_dim, -1) if w.ndim > 2 else W)[out_idx, j] = 0.0
                ok, _ = all_samples_correct(
                    model, filtered_dataset, device, eval_max=eval_max, profiler=profiler, move_first_offender=True
                )
                if ok:
                    changed_any = True
                else:
                    row[j] = prev
                    with torch.no_grad():
                        (W.view(out_dim, -1) if w.ndim > 2 else W)[out_idx, j] = prev
                    break
        if include_bias and (b is not None):
            for j in torch.argsort(torch.abs(bval), descending=False).tolist():
                if bval[j].item() == 0.0:
                    continue
                prev = bval[j].item()
                with torch.no_grad():
                    b[j] = 0.0
                ok, _ = all_samples_correct(model, filtered_dataset, device, eval_max=eval_max, profiler=profiler)
                if ok:
                    changed_any = True
                    bval[j] = 0.0
                else:
                    with torch.no_grad():
                        b[j] = prev
                    break
        if profiler is not None:
            profiler.record_weight_distribution(name, w.detach().cpu().numpy(), W.detach().cpu().numpy())
    if profiler is not None:
        profiler.end_phase("pruning")
        profiler._add_time("pruning_total", perf_counter() - t0_all)
    return changed_any

@torch.inference_mode()
def magnitude_prune_binary_search(
    model: nn.Module,
    filtered_dataset,
    device: torch.device,
    eval_max: Optional[int] = None,
    include_bias: bool = False,
    check_batch_size: int = 1024,
    profiler: Optional[SICProfiler] = None,
) -> bool:
    if _dataset_is_empty(filtered_dataset):
        print("[PRUNE] Skipping: acceptance set empty.")
        return False
    changed_any = False
    if profiler is not None:
        profiler.start_phase("pruning")
    baseline = copy.deepcopy(model.state_dict())
    for name, module in list(model.named_modules()):
        W = getattr(module, "weight", None)
        if W is None:
            continue
        w = W.detach().clone()
        w_flat = w.reshape(-1)
        if w_flat.numel() == 0:
            continue
        mags_w = w_flat.abs()
        has_bias = include_bias and hasattr(module, "bias") and (module.bias is not None)
        if has_bias:
            bpar = module.bias.detach().clone()
            mags_b = bpar.reshape(-1).abs()
            comb = torch.cat([mags_w, mags_b], dim=0)
            split = mags_w.numel()
        else:
            comb = mags_w
            split = mags_w.numel()
        _, perm = torch.sort(comb, descending=False)
        total = int(comb.numel())
        if total == 0:
            continue
        lo, hi, best_k = 0, total, 0
        def apply_zero(k: int):
            model.load_state_dict(baseline, strict=True)
            with torch.no_grad():
                zero_idx = perm[:k]
                zm_w = torch.ones(split, dtype=torch.bool, device=w.device)
                if has_bias:
                    zm_b = torch.ones(bpar.numel(), dtype=torch.bool, device=w.device)
                if k > 0:
                    w_sel = zero_idx[zero_idx < split]
                    if w_sel.numel() > 0:
                        zm_w[w_sel] = False
                    if has_bias:
                        b_sel = zero_idx[zero_idx >= split] - split
                        if b_sel.numel() > 0:
                            zm_b[b_sel] = False
                W.copy_((w_flat * zm_w).view_as(W))
                if has_bias:
                    module.bias.copy_(bpar.view(-1) * zm_b)
        while lo <= hi:
            mid = (lo + hi) // 2
            apply_zero(mid)
            ok, _ = all_samples_correct(
                model, filtered_dataset, device, eval_max=eval_max, profiler=profiler, batch_size=check_batch_size
            )
            if ok:
                best_k = mid
                lo = mid + 1
            else:
                hi = mid - 1
        apply_zero(best_k)
        changed_any = changed_any or (best_k > 0)
        if profiler is not None:
            profiler.record_weight_distribution(name, w.detach().cpu().numpy(), W.detach().cpu().numpy())
        baseline = copy.deepcopy(model.state_dict())
    if profiler is not None:
        profiler.end_phase("pruning")
    return changed_any

@torch.inference_mode()
def magnitude_prune_pos_sequential(
    module: nn.Module,
    model: nn.Module,
    filtered_dataset,
    device: torch.device,
    eval_max: Optional[int] = None,
    include_bias: bool = False,
    profiler: Optional[SICProfiler] = None,
) -> bool:
    if _dataset_is_empty(filtered_dataset):
        return False
    changed_any = False
    before = dense_from_pos(module).detach().cpu().numpy()
    rows_name = rows_container_name(module, conv=False) if is_pos_linear(module) else rows_container_name(module, conv=True)
    row_iter = enumerate(getattr(module, rows_name))
    b = getattr(module, "bias", None)
    if include_bias and (b is not None):
        bval = b.detach().clone()
    for _, holder in row_iter:
        k = int(getattr(holder, "k", 0))
        if k <= 0:
            continue
        means = holder.means
        for idx in torch.argsort(torch.abs(means)).tolist():
            prev = float(means[idx].item())
            if prev == 0.0:
                continue
            means[idx] = 0.0
            ok, _ = all_samples_correct(
                model, filtered_dataset, device, eval_max=eval_max, profiler=profiler, move_first_offender=True
            )
            if ok:
                changed_any = True
            else:
                means[idx] = prev
                break
    if include_bias and (b is not None):
        for j in torch.argsort(torch.abs(bval), descending=False).tolist():
            prev = float(bval[j].item())
            if prev == 0.0:
                continue
            b[j] = 0.0
            ok, _ = all_samples_correct(model, filtered_dataset, device, eval_max=eval_max, profiler=profiler)
            if ok:
                changed_any = True
                bval[j] = 0.0
            else:
                b[j] = prev
                break
    if profiler is not None:
        after = dense_from_pos(module).detach().cpu().numpy()
        profiler.record_weight_distribution(module.__class__.__name__ + "(PoS)", before, after)
    return changed_any

def run_pruning_if_needed(
    model: nn.Module,
    cfg: dict,
    train_loader,
    val_loader,
    device: torch.device,
    profiler: Optional[SICProfiler] = None,
    reason: str = "",
) -> Tuple[nn.Module, bool]:
    pr_cfg = cfg.get("prune", {}) or {}
    if not bool(pr_cfg.get("enabled", False)):
        return model, False
    filtered_dataset = build_acceptance_subset_from_loaders(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        from_spec=str(pr_cfg.get("acceptance_from", "train+val")),
        max_examples=(None if not pr_cfg.get("acceptance_max") else int(pr_cfg.get("acceptance_max"))),
    )
    eval_max = cfg.get("sic", {}).get("eval_max", None)
    strategy = str(pr_cfg.get("strategy", "per_neuron_sequential")).strip().lower()
    include_bias = bool(pr_cfg.get("include_bias", False))
    check_bs = int(pr_cfg.get("check_batch_size", 1024) or 1024)
    print(f"[PRUNE] Triggered ({reason}) strategy={strategy}, acceptance={len(filtered_dataset)} samples.")
    if _dataset_is_empty(filtered_dataset):
        print("[PRUNE] Skipping: acceptance set empty.")
        return model, False
    if strategy in {"per_layer_binary_search", "layer_binary", "layer_binary_search", "per_neuron_binary", "neuron_binary"}:
        changed = magnitude_prune_binary_search(
            model, filtered_dataset, device, eval_max=eval_max, include_bias=include_bias, check_batch_size=check_bs, profiler=profiler
        )
        return model, changed
    if strategy in {"per_neuron_binary_pos", "neuron_binary_pos", "pos_binary"}:
        changed_any = False
        for _, module in list(model.named_modules()):
            if is_pos_linear(module) or is_pos_conv2d(module):
                c = magnitude_prune_pos_binary_search(
                    module, model, filtered_dataset, device, eval_max=eval_max, include_bias=include_bias, check_batch_size=check_bs, profiler=profiler
                )
                changed_any = changed_any or c
        return model, changed_any
    max_passes = int(cfg.get("sic", {}).get("max_passes", 3) or 3)
    repeat = bool(cfg.get("sic", {}).get("repeat_until_no_change", True))
    pass_idx, changed_any, prev_fp = 0, False, None
    def fingerprint(m: nn.Module) -> str:
        h = hashlib.sha256()
        for t in m.state_dict().values():
            h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    while True:
        pass_idx += 1
        c = magnitude_prune_per_neuron_sequential(
            model, filtered_dataset, device, eval_max=eval_max, include_bias=include_bias, profiler=profiler
        )
        changed_any = changed_any or c
        if not repeat or pass_idx >= max_passes:
            break
        fp = fingerprint(model)
        if prev_fp is not None and fp == prev_fp:
            break
        prev_fp = fp
    if changed_any:
        print("[PRUNE] Pruning committed changes.")
    else:
        print("[PRUNE] No acceptable pruning found (kept weights unchanged).")
    return model, changed_any

@torch.inference_mode()
def magnitude_prune_pos_binary_search(
    module: nn.Module,
    model: nn.Module,
    filtered_dataset,
    device: torch.device,
    eval_max: Optional[int] = None,
    include_bias: bool = False,
    check_batch_size: int = 1024,
    profiler: Optional[SICProfiler] = None,
) -> bool:
    if _dataset_is_empty(filtered_dataset):
        return False
    if is_pos_linear(module):
        holders = list(getattr(module, rows_container_name(module)))
    else:
        holders = list(getattr(module, rows_container_name(module, conv=True)))
    before = dense_from_pos(module).detach().cpu().numpy()
    entries: List[Tuple[bool, Optional[int], int, float]] = []
    for hi, h in enumerate(holders):
        k = int(getattr(h, "k", 0))
        for mi in range(k):
            v = float(h.means[mi].item())
            if v != 0.0:
                entries.append((False, hi, mi, abs(v)))
    b = getattr(module, "bias", None)
    has_bias = include_bias and (b is not None) and (b.numel() > 0)
    if has_bias:
        for j in range(b.numel()):
            v = float(b[j].item())
            if v != 0.0:
                entries.append((True, None, j, abs(v)))
    if not entries:
        return False
    mags = torch.tensor([e[3] for e in entries], dtype=torch.float32)
    _, perm = torch.sort(mags, descending=False)
    total = int(mags.numel())
    baseline = copy.deepcopy(model.state_dict())
    lo, hi, best_k = 0, total, 0
    def apply_zero(k: int):
        model.load_state_dict(baseline, strict=True)
        with torch.no_grad():
            for t in range(k):
                is_bias, hi_, mi_or_j, _ = entries[perm[t].item()]
                if is_bias:
                    b[mi_or_j] = 0.0
                else:
                    holders[hi_].means[mi_or_j] = 0.0
    while lo <= hi:
        mid = (lo + hi) // 2
        apply_zero(mid)
        ok, _ = all_samples_correct(
            model, filtered_dataset, device, eval_max=eval_max, profiler=profiler, batch_size=check_batch_size
        )
        if ok:
            best_k = mid
            lo = mid + 1
        else:
            hi = mid - 1
    apply_zero(best_k)
    if profiler is not None:
        after = dense_from_pos(module).detach().cpu().numpy()
        profiler.record_weight_distribution(module.__class__.__name__ + "(PoS)", before, after)
    return best_k > 0
