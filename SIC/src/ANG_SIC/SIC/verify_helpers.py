from __future__ import annotations

import math
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch import nn

def _bias_dtype(m: nn.Module) -> torch.dtype:
    b = getattr(m, "bias", None)
    return b.dtype if isinstance(b, torch.Tensor) else torch.float32

def _bias_count(m: nn.Module) -> int:
    b = getattr(m, "bias", None)
    return int(b.numel()) if isinstance(b, torch.Tensor) else 0

def _has_weight(m: nn.Module) -> bool:
    w = getattr(m, "weight", None)
    return isinstance(w, torch.Tensor) and w.ndim >= 2 and w.numel() > 0

def _is_masked_linear_like(m: nn.Module) -> bool:
    return (
        _has_weight(m)
        and hasattr(m, "mask")
        and isinstance(getattr(m, "mask"), torch.Tensor)
        and not isinstance(m, nn.Linear)
        and getattr(m, "weight").ndim == 2
    )

def is_sic_linear(mod: nn.Module) -> bool:
    return (
        mod.__class__.__name__ == "SICLinear"
        or (
            hasattr(mod, "rows")
            and hasattr(mod, "in_features")
            and hasattr(mod, "out_features")
            and not hasattr(mod, "weight")
        )
    )

def is_sic_conv2d(mod: nn.Module) -> bool:
    return (
        mod.__class__.__name__ == "SICConv2d"
        or (
            hasattr(mod, "out_rows")
            and hasattr(mod, "in_channels")
            and hasattr(mod, "kernel_size")
            and not hasattr(mod, "weight")
        )
    )

@torch.inference_mode()
def _dense_from_siclinear(m: nn.Module) -> Tuple[torch.Tensor, int]:
    in_features = int(m.in_features)
    dtype = _bias_dtype(m)
    rows = []
    for holder in m.rows:
        row = torch.zeros(in_features, dtype=dtype)
        k = int(getattr(holder, "k", 0))
        for g in range(k):
            idxs = holder.idx[holder.cid == g]
            if idxs.numel():
                row[idxs.long()] = holder.means[g].to(row.dtype)
        rows.append(row)
    W = torch.stack(rows, dim=0) if rows else torch.zeros((0, in_features), dtype=dtype)
    return W, _bias_count(m)

@torch.inference_mode()
def _dense_from_sicconv2d(m: nn.Module) -> Tuple[torch.Tensor, int]:
    in_channels = int(m.in_channels)
    if isinstance(m.kernel_size, (tuple, list)):
        kH, kW = int(m.kernel_size[0]), int(m.kernel_size[1])
    else:
        kH = kW = int(m.kernel_size)
    flat_len = in_channels * kH * kW
    dtype = _bias_dtype(m)
    rows = []
    for holder in m.out_rows:
        flat = torch.zeros(flat_len, dtype=dtype)
        k = int(getattr(holder, "k", 0))
        for g in range(k):
            idxs = holder.idx[holder.cid == g]
            if idxs.numel():
                flat[idxs.long()] = holder.means[g].to(flat.dtype)
        rows.append(flat.view(in_channels, kH, kW))
    W = torch.stack(rows, dim=0) if rows else torch.zeros((0, in_channels, kH, kW), dtype=dtype)
    return W, _bias_count(m)

@torch.inference_mode()
def _dense_from_fast_pos_linear(m: nn.Module) -> Tuple[Optional[torch.Tensor], int]:
    of = int(getattr(m, "out_features", 0))
    inf = int(getattr(m, "in_features", 0))
    dest = getattr(m, "dest", None)
    col = getattr(m, "col_idx", None)
    means = getattr(m, "means", None)
    if of <= 0 or inf <= 0 or dest is None or col is None or means is None:
        return None, 0
    dest = dest.detach().cpu().long().view(-1)
    col = col.detach().cpu().long().view(-1)
    means = means.detach().cpu()
    G = int(means.size(1)) if means.ndim == 2 else 0
    W = torch.zeros(of, inf, dtype=means.dtype)
    if G > 0 and col.numel() > 0 and dest.numel() > 0:
        row = torch.div(dest, G, rounding_mode="floor")
        grp = torch.remainder(dest, G)
        W[row, col] = means[row, grp]
    return W, _bias_count(m)

@torch.inference_mode()
def _dense_from_fast_pos_conv2d(m: nn.Module) -> Tuple[Optional[torch.Tensor], int]:
    outC = int(getattr(m, "out_channels", 0))
    inC = int(getattr(m, "in_channels", 0))
    k = getattr(m, "kernel_size", (1, 1))
    kH = int(k[0]) if isinstance(k, (tuple, list)) else int(k)
    kW = int(k[1]) if isinstance(k, (tuple, list)) else int(k)
    dest = getattr(m, "dest", None)
    col = getattr(m, "col_idx", None)
    means = getattr(m, "means", None)
    if outC <= 0 or inC <= 0 or kH <= 0 or kW <= 0 or dest is None or col is None or means is None:
        return None, 0
    L = inC * kH * kW
    dest = dest.detach().cpu().long().view(-1)
    col = col.detach().cpu().long().view(-1)
    means = means.detach().cpu()
    G = int(means.size(1)) if means.ndim == 2 else 0
    Wflat = torch.zeros(outC, L, dtype=means.dtype)
    if G > 0 and col.numel() > 0 and dest.numel() > 0:
        row = torch.div(dest, G, rounding_mode="floor")
        grp = torch.remainder(dest, G)
        Wflat[row, col] = means[row, grp]
    W = Wflat.view(outC, inC, kH, kW)
    return W, _bias_count(m)

def _looks_like_fast_pos(m: nn.Module) -> bool:
    return hasattr(m, "dest") and hasattr(m, "col_idx") and hasattr(m, "means") and not hasattr(m, "weight")

def _recurse_inner(mod: nn.Module) -> Tuple[Optional[torch.Tensor], int]:
    for attr in ("inner", "base", "pos", "module", "_mod", "_fastpos"):
        inner = getattr(mod, attr, None)
        if isinstance(inner, nn.Module):
            W, b = to_dense_and_bias(inner)
            if W is not None:
                return W, b
    return None, 0

@torch.inference_mode()
def to_dense_and_bias(module: nn.Module) -> Tuple[Optional[torch.Tensor], int]:
    if _is_masked_linear_like(module):
        W = module.weight.detach().cpu()
        M = module.mask.detach().cpu().to(W.dtype)
        return W * M, _bias_count(module)
    if _has_weight(module):
        return module.weight.detach().cpu(), _bias_count(module)
    if is_sic_linear(module):
        return _dense_from_siclinear(module)
    if is_sic_conv2d(module):
        return _dense_from_sicconv2d(module)
    if _looks_like_fast_pos(module):
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            W, b = _dense_from_fast_pos_linear(module)
            if W is not None:
                return W, b
        if hasattr(module, "in_channels") and hasattr(module, "out_channels") and hasattr(module, "kernel_size"):
            W, b = _dense_from_fast_pos_conv2d(module)
            if W is not None:
                return W, b
    W, b = _recurse_inner(module)
    if W is not None:
        return W, b
    for pname, p in module.named_parameters(recurse=False):
        if pname.lower() == "bias":
            continue
        if isinstance(p, torch.Tensor) and p.ndim in (2, 4) and p.numel() > 0:
            return p.detach().cpu(), _bias_count(module)
    return None, 0

def _is_masked_conv_like(m: nn.Module) -> bool:
    return (
        hasattr(m, "weight") and isinstance(getattr(m, "weight"), torch.Tensor) and getattr(m, "weight").ndim == 4
        and hasattr(m, "mask") and isinstance(getattr(m, "mask"), torch.Tensor)
    )

def _is_fast_pos_linear(m: nn.Module) -> bool:
    return (
        hasattr(m, "dest") and hasattr(m, "col_idx") and hasattr(m, "means")
        and hasattr(m, "in_features") and hasattr(m, "out_features")
        and not hasattr(m, "weight")
    )

def _is_fast_pos_conv2d(m: nn.Module) -> bool:
    return (
        hasattr(m, "dest") and hasattr(m, "col_idx") and hasattr(m, "means")
        and hasattr(m, "in_channels") and hasattr(m, "out_channels") and hasattr(m, "kernel_size")
        and not hasattr(m, "weight")
    )

def _is_indexed_sparse_linear(m: nn.Module) -> bool:
    return (
        hasattr(m, "indices") and isinstance(getattr(m, "indices"), torch.Tensor)
        and hasattr(m, "theta") and isinstance(getattr(m, "theta"), torch.Tensor)
        and hasattr(m, "bias") and isinstance(getattr(m, "bias"), torch.Tensor)
        and (hasattr(m, "in_features") or hasattr(m, "D"))
        and (hasattr(m, "out_features") or hasattr(m, "K"))
        and not hasattr(m, "weight")
    )

def _group_counts_1d(vec: torch.Tensor, rounding_decimals: int | None = None) -> dict[float, int]:
    v = vec.detach().cpu().view(-1).numpy()
    if rounding_decimals is not None:
        v = np.round(v, decimals=int(rounding_decimals))
    counts: dict[float, int] = {}
    for val in v:
        if val == 0.0:
            continue
        counts[val] = counts.get(val, 0) + 1
    return counts

@torch.inference_mode()
def _fast_pos_group_stats_linear(m: nn.Module) -> tuple[list[int], list[int]]:
    if not _is_fast_pos_linear(m):
        return [], []
    dest = m.dest.detach().cpu().long().view(-1)
    col = m.col_idx.detach().cpu().long().view(-1)
    if dest.numel() == 0 or col.numel() == 0:
        return [0] * int(m.out_features), [0] * int(m.out_features)
    Gmax = int(m.means.size(1)) if m.means.ndim == 2 else 0
    if Gmax <= 0:
        return [0] * int(m.out_features), [0] * int(m.out_features)
    rows = dest.div(Gmax, rounding_mode="floor")
    grps = dest.remainder(Gmax)
    outF = int(m.out_features)
    G_list = [0] * outF
    adds_list = [0] * outF
    key = rows * (Gmax if Gmax > 0 else 1) + grps
    uniq, cnts = torch.unique(key, return_counts=True)
    u_rows = uniq.div(Gmax if Gmax > 0 else 1, rounding_mode="floor")
    from collections import defaultdict as _dd
    per_row_counts = _dd(list)
    for r, c in zip(u_rows.tolist(), cnts.tolist()):
        per_row_counts[int(r)].append(int(c))
    for r in range(outF):
        counts = per_row_counts.get(r, [])
        G = len(counts)
        G_list[r] = G
        adds_list[r] = sum(max(0, c - 1) for c in counts)
    return G_list, adds_list

@torch.inference_mode()
def _fast_pos_group_stats_conv(m: nn.Module) -> tuple[list[int], list[int]]:
    if not _is_fast_pos_conv2d(m):
        return [], []
    dest = m.dest.detach().cpu().long().view(-1)
    col = m.col_idx.detach().cpu().long().view(-1)
    if dest.numel() == 0 or col.numel() == 0:
        return [0] * int(m.out_channels), [0] * int(m.out_channels)
    Gmax = int(m.means.size(1)) if m.means.ndim == 2 else 0
    if Gmax <= 0:
        return [0] * int(m.out_channels), [0] * int(m.out_channels)
    rows = dest.div(Gmax, rounding_mode="floor")
    grps = dest.remainder(Gmax)
    outC = int(m.out_channels)
    G_list = [0] * outC
    adds_list = [0] * outC
    key = rows * (Gmax if Gmax > 0 else 1) + grps
    uniq, cnts = torch.unique(key, return_counts=True)
    u_rows = uniq.div(Gmax if Gmax > 0 else 1, rounding_mode="floor")
    from collections import defaultdict as _dd
    per_row_counts = _dd(list)
    for r, c in zip(u_rows.tolist(), cnts.tolist()):
        per_row_counts[int(r)].append(int(c))
    for r in range(outC):
        counts = per_row_counts.get(r, [])
        G = len(counts)
        G_list[r] = G
        adds_list[r] = sum(max(0, c - 1) for c in counts)
    return G_list, adds_list

@torch.inference_mode()
def _indexed_sparse_nnz_per_out(m: nn.Module) -> torch.Tensor:
    if hasattr(m, "row_ptr") and isinstance(getattr(m, "row_ptr"), torch.Tensor) and m.row_ptr.numel() >= 2:
        rp = m.row_ptr.detach().cpu().long().view(-1)
        return (rp[1:] - rp[:-1]).clamp_min(0)
    idx = m.indices.detach().cpu().long()
    if idx.numel() == 0:
        K = int(getattr(m, "out_features", getattr(m, "K", 0)))
        return torch.zeros(int(K), dtype=torch.long)
    rows = idx[0]
    K = int(getattr(m, "out_features", getattr(m, "K", rows.max().item() + 1)))
    return torch.bincount(rows, minlength=K)[:K]

@torch.inference_mode()
def _fast_pos_groups_and_nz_means(m: nn.Module) -> tuple[int, int]:
    if _is_fast_pos_linear(m):
        G_list, _ = _fast_pos_group_stats_linear(m)
        groups_total = int(sum(G_list))
        means = m.means.detach().cpu()
        nz_means = 0
        for r, G in enumerate(G_list):
            if G <= 0:
                continue
            nz_means += int(torch.count_nonzero(means[r, :G]).item())
        return groups_total, nz_means
    if _is_fast_pos_conv2d(m):
        G_list, _ = _fast_pos_group_stats_conv(m)
        groups_total = int(sum(G_list))
        means = m.means.detach().cpu()
        nz_means = 0
        for r, G in enumerate(G_list):
            if G <= 0:
                continue
            nz_means += int(torch.count_nonzero(means[r, :G]).item())
        return groups_total, nz_means
    return 0, 0

def _mk_printer(printer, verbose):
    def _p(*a, **k):
        if verbose:
            printer(*a, **k)
    return _p

def _collect_interesting_modules(model: nn.Module) -> Dict[str, nn.Module]:
    interesting: dict[str, nn.Module] = {}
    for name, m in model.named_modules():
        w, _ = to_dense_and_bias(m)
        is_weight_like = w is not None and isinstance(w, torch.Tensor) and w.numel() > 0
        is_fastpos = _is_fast_pos_linear(m) or _is_fast_pos_conv2d(m)
        is_indexed_sparse = _is_indexed_sparse_linear(m)
        if (
            isinstance(
                m,
                (
                    nn.Flatten,
                    nn.MaxPool2d,
                    nn.AvgPool2d,
                    nn.AdaptiveAvgPool2d,
                    nn.BatchNorm2d,
                    nn.ReLU,
                    nn.Identity,
                ),
            )
            or is_weight_like
            or is_fastpos
            or is_indexed_sparse
        ):
            interesting[name] = m
    return interesting

def _normalize_candidates(try_shapes: list | None, model: nn.Module) -> list[tuple[str, tuple | int]]:
    out: list[tuple[str, tuple | int]] = []
    if try_shapes:
        for item in try_shapes:
            if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
                out.append((item[0], item[1]))
                continue
            if isinstance(item, (tuple, list)):
                if len(item) == 4:
                    _, C, H, W = item
                    out.append(("conv", (int(C), int(H), int(W))))
                elif len(item) == 3:
                    C, H, W = item
                    out.append(("conv", (int(C), int(H), int(W))))
                elif len(item) == 2:
                    out.append(("linear", int(item[1])))
                elif len(item) == 1:
                    out.append(("linear", int(item[0])))
            elif isinstance(item, (int, float)):
                out.append(("linear", int(item)))
    first_lin_in = None
    for _, m in model.named_modules():
        w, _ = to_dense_and_bias(m)
        if w is not None and w.ndim == 2:
            first_lin_in = int(w.shape[1])
            break
    if first_lin_in is not None:
        out += [("linear", int(first_lin_in))]
    out += [
        ("conv", (1, 28, 28)),
        ("conv", (3, 32, 32)),
        ("conv", (1, 32, 32)),
        ("conv", (3, 28, 28)),
    ]
    return out

def _register_hooks(interesting: Dict[str, nn.Module]) -> Tuple[List, dict]:
    layer_shapes: dict[str, dict] = {}
    def _mk_hook(name: str, module: nn.Module):
        def _hook(_mod, inp, out):
            in0 = inp[0] if isinstance(inp, (tuple, list)) else inp
            out0 = out[0] if (isinstance(out, (tuple, list)) and hasattr(out[0], "shape")) else out
            layer_shapes[name] = {
                "in": tuple(in0.shape) if hasattr(in0, "shape") else None,
                "out": tuple(out0.shape) if hasattr(out0, "shape") else None,
                "module": module,
            }
        return _hook
    handles = [md.register_forward_hook(_mk_hook(nm, md)) for nm, md in interesting.items()]
    return handles, layer_shapes

def _try_infer_shapes(
    model: nn.Module,
    interesting: Dict[str, nn.Module],
    sample: torch.Tensor | None,
    input_spec: tuple[str, int | tuple[int, int, int]] | None,
    try_shapes: list | None,
    dev: torch.device,
    _p,
) -> tuple[bool, dict, list]:
    handles, layer_shapes = _register_hooks(interesting)
    def _try_forward(x: torch.Tensor) -> bool:
        try:
            _ = model(x.to(dev, non_blocking=(dev.type == "cuda")))
            return True
        except Exception:
            return False
    tried: list[tuple[int, ...]] = []
    ok = False
    with torch.no_grad():
        if sample is not None:
            tried.append(tuple(sample.shape))
            ok = _try_forward(sample.cpu())
        if not ok and input_spec is not None:
            kind, spec = input_spec
            if kind == "linear":
                x = torch.randn(1, int(spec))
                tried.append(tuple(x.shape))
                ok = _try_forward(x)
            elif kind == "conv":
                C, H, W = spec
                x = torch.randn(1, int(C), int(H), int(W))
                tried.append(tuple(x.shape))
                ok = _try_forward(x)
        if not ok:
            candidates = _normalize_candidates(try_shapes, model)
            for kind, spec in candidates:
                x = torch.randn(1, int(spec)) if kind == "linear" else torch.randn(1, int(spec[0]), int(spec[1]), int(spec[2]))
                tried.append(tuple(x.shape))
                if _try_forward(x):
                    ok = True
                    break
    for h in handles:
        h.remove()
    return ok, layer_shapes, tried

def _ops_and_firsts(model: nn.Module) -> tuple[list[str], list[tuple[str, nn.Module, str]], nn.Module | None]:
    conv_order, ops, first_linear = [], [], None
    for name, m in model.named_modules():
        w, _ = to_dense_and_bias(m)
        if w is not None and w.ndim == 4:
            conv_order.append(name)
            ops.append(("conv", m, name))
        elif _is_fast_pos_conv2d(m):
            conv_order.append(name)
            ops.append(("conv", m, name))
        elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            ops.append(("pool", m, name))
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            ops.append(("adap", m, name))
        elif isinstance(m, nn.Flatten):
            ops.append(("flat", m, name))
        elif (w is not None and w.ndim == 2) or _is_fast_pos_linear(m) or _is_indexed_sparse_linear(m):
            if first_linear is None:
                first_linear = m
        elif isinstance(m, (nn.ReLU, nn.BatchNorm2d, nn.Identity)):
            ops.append(("pass", m, name))
    return conv_order, ops, first_linear

def _need_any_hw(conv_order: List[str], layer_shapes: Dict[str, dict]) -> bool:
    for nm in conv_order:
        shp = layer_shapes.get(nm)
        if not (shp and shp.get("out") and len(shp["out"]) == 4):
            return True
    return False

def _infer_conv_hw_static(
    model: nn.Module,
    interesting: Dict[str, nn.Module],
    conv_order: List[str],
    ops: List[tuple],
    first_linear: nn.Module | None,
    layer_shapes: Dict[str, dict],
    ok: bool,
    force_hook_hw: bool,
    _p,
) -> Dict[str, tuple[int, int]]:
    inferred_conv_hw: dict[str, tuple[int, int]] = {}
    if force_hook_hw or ok or first_linear is None or len(conv_order) == 0:
        return inferred_conv_hw
    last_conv_name = conv_order[-1]
    last_conv = None
    for n2, m2 in interesting.items():
        if n2 == last_conv_name:
            last_conv = m2
            break
    Co_last = None
    if last_conv is not None:
        w_last, _ = to_dense_and_bias(last_conv)
        if w_last is not None and w_last.ndim == 4:
            Co_last = int(w_last.shape[0])
        elif _is_fast_pos_conv2d(last_conv):
            Co_last = int(getattr(last_conv, "out_channels", 0))
    if not Co_last:
        return inferred_conv_hw
    target_area = None
    w_fc, _ = to_dense_and_bias(first_linear)
    fc_in = None
    if w_fc is not None and w_fc.ndim == 2:
        fc_in = int(w_fc.shape[1])
    elif _is_fast_pos_linear(first_linear) or _is_indexed_sparse_linear(first_linear):
        fc_in = int(getattr(first_linear, "in_features", getattr(first_linear, "D", 0)))
    if fc_in and fc_in % Co_last == 0:
        target_area = fc_in // Co_last
    found = False
    for H0 in range(8, 257):
        H = W = H0
        conv_out_hw: dict[str, tuple[int, int]] = {}
        for kind, mod, nm in ops:
            if kind == "conv":
                if hasattr(mod, "kernel_size"):
                    k = mod.kernel_size
                    kH = int(k[0]) if isinstance(k, (tuple, list)) else int(k)
                    kW = int(k[1]) if isinstance(k, (tuple, list)) else int(kH)
                else:
                    wt, _ = to_dense_and_bias(mod)
                    kH = int(wt.shape[-2]) if (wt is not None and wt.ndim == 4) else 1
                    kW = int(wt.shape[-1]) if (wt is not None and wt.ndim == 4) else 1
                s = getattr(mod, "stride", (1, 1))
                p = getattr(mod, "padding", (0, 0))
                d = getattr(mod, "dilation", (1, 1))
                sH = s[0] if isinstance(s, tuple) else int(s)
                sW = s[1] if isinstance(s, tuple) else int(sH)
                pH = p[0] if isinstance(p, tuple) else int(p)
                pW = p[1] if isinstance(p, tuple) else int(pH)
                dH = d[0] if isinstance(d, tuple) else int(d)
                dW = d[1] if isinstance(d, tuple) else int(dH)
                H = math.floor((H + 2 * pH - dH * (kH - 1) - 1) / sH + 1)
                W = math.floor((W + 2 * pW - dW * (kW - 1) - 1) / sW + 1)
                if H <= 0 or W <= 0:
                    conv_out_hw = {}
                    break
                conv_out_hw[nm] = (H, W)
            elif kind == "pool":
                k = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
                s = mod.stride if (mod.stride is not None) else k
                p = getattr(mod, "padding", (0, 0))
                kH, kW = k
                sH = s[0] if isinstance(s, tuple) else int(s)
                sW = s[1] if isinstance(s, tuple) else int(sH)
                pH = p[0] if isinstance(p, tuple) else int(p)
                pW = p[1] if isinstance(p, tuple) else int(pH)
                H = math.floor((H + 2 * pH - (kH - 1) - 1) / sH + 1)
                W = math.floor((W + 2 * pW - (kW - 1) - 1) / sW + 1)
                if H <= 0 or W <= 0:
                    conv_out_hw = {}
                    break
            elif kind == "adap":
                outsz = mod.output_size
                if isinstance(outsz, tuple):
                    H, W = int(outsz[0]), int(outsz[1])
                else:
                    H = W = int(outsz)
                conv_out_hw[nm] = (H, W)
        if conv_out_hw:
            if target_area is None:
                inferred_conv_hw = conv_out_hw
                found = True
                break
            Hl, Wl = conv_out_hw.get(last_conv_name, (None, None))
            if Hl is not None and (Hl * Wl == target_area):
                inferred_conv_hw = conv_out_hw
                found = True
                break
    if (not found) and target_area is not None:
        _p("[verify] Static solver did not find H=W in [8..256] matching the first Linear; conv counts may be partial.")
    return inferred_conv_hw

def _format_layer_label(name: str, is_conv_like: bool, is_linear_like: bool) -> str:
    n = name.lower()
    if n.startswith("conv") and len(name) > 4 and name[4:].isdigit():
        return f"Conv {name[4:]}"
    if n.startswith("fc") and len(name) > 2 and name[2:].isdigit():
        return f"FC {name[2:]}"
    if n.startswith("linear") and len(name) > 6 and name[6:].isdigit():
        return f"Linear {name[6:]}"
    return ("Conv " + name) if is_conv_like else (("Linear " + name) if is_linear_like else name)
