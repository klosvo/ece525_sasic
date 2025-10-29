from __future__ import annotations
import sys
from typing import Optional, Tuple, List
import torch
import torch.nn as nn

ROW_ALIASES = ("rows", "unit_rows", "grown_rows", "neurons", "units")
CONV_ROW_ALIASES = ("out_rows",) + ROW_ALIASES

def can_torch_compile() -> bool:
    return hasattr(torch, "compile") and sys.version_info < (3, 12)

def _has_pos_holder_shape(h) -> bool:
    return all(hasattr(h, t) for t in ("idx", "cid", "means", "k"))

def rows_container_name(mod: nn.Module, *, conv: bool = False) -> Optional[str]:
    names = CONV_ROW_ALIASES if conv else ROW_ALIASES
    for name in names:
        rows = getattr(mod, name, None)
        if isinstance(rows, nn.ModuleList) and len(rows) > 0 and _has_pos_holder_shape(rows[0]):
            return name
    return None

def is_pos_linear(mod: nn.Module) -> bool:
    return (rows_container_name(mod, conv=False) is not None) and \
           (not hasattr(mod, "weight")) and \
           hasattr(mod, "in_features") and hasattr(mod, "out_features")

def is_pos_conv2d(mod: nn.Module) -> bool:
    return (rows_container_name(mod, conv=True) is not None) and \
           (not hasattr(mod, "weight")) and \
           hasattr(mod, "in_channels") and hasattr(mod, "out_channels") and hasattr(mod, "kernel_size")

def is_masked_linear_like(m: nn.Module) -> bool:
    return hasattr(m, "weight") and isinstance(getattr(m, "weight"), torch.Tensor) and \
           getattr(m, "weight").ndim == 2 and hasattr(m, "mask") and isinstance(getattr(m, "mask"), torch.Tensor) and \
           not isinstance(m, nn.Linear)

@torch.inference_mode()
def dense_from_pos(mod: nn.Module) -> torch.Tensor:
    if is_pos_linear(mod):
        name = rows_container_name(mod, conv=False); rows = getattr(mod, name)
        L = int(mod.in_features)
        dtype = mod.bias.dtype if isinstance(getattr(mod, "bias", None), torch.Tensor) else torch.float32
        out = []
        for h in rows:
            row = torch.zeros(L, dtype=dtype)
            k = int(getattr(h, "k", 0))
            for g in range(k):
                idxs = h.idx[h.cid == g]
                if idxs.numel():
                    row[idxs.long()] = h.means[g].to(row.dtype)
            out.append(row)
        return torch.stack(out, dim=0) if out else torch.zeros((int(mod.out_features), L), dtype=dtype)

    if is_pos_conv2d(mod):
        name = rows_container_name(mod, conv=True); rows = getattr(mod, name)
        inC = int(mod.in_channels)
        kH, kW = map(int, (mod.kernel_size if isinstance(mod.kernel_size, (tuple, list)) else (mod.kernel_size, mod.kernel_size)))
        L = inC * kH * kW
        dtype = mod.bias.dtype if isinstance(getattr(mod, "bias", None), torch.Tensor) else torch.float32
        out = []
        for h in rows:
            flat = torch.zeros(L, dtype=dtype)
            k = int(getattr(h, "k", 0))
            for g in range(k):
                idxs = h.idx[h.cid == g]
                if idxs.numel():
                    flat[idxs.long()] = h.means[g].to(flat.dtype)
            out.append(flat.view(inC, kH, kW))
        return torch.stack(out, dim=0) if out else torch.zeros((int(mod.out_channels), inC, kH, kW), dtype=dtype)

    raise TypeError("dense_from_pos: unsupported module")

def conv2d_out_hw(H: int, W: int, kernel: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int], dilation: Tuple[int, int]) -> Tuple[int, int]:
    kH, kW = map(int, kernel); sH, sW = map(int, stride); pH, pW = map(int, padding); dH, dW = map(int, dilation)
    Ho = ((H + 2 * pH - dH * (kH - 1) - 1) // sH) + 1
    Wo = ((W + 2 * pW - dW * (kW - 1) - 1) // sW) + 1
    return Ho, Wo

@torch.inference_mode()
def pack_rows_to_indices_linear(rows: nn.ModuleList, out_features: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    Gs = [int(getattr(h, "k", 0)) for h in rows]
    Gmax = max(Gs) if Gs else 0
    row_ids, col_idx, group_ids = [], [], []
    means = torch.zeros(out_features, Gmax, dtype=torch.float32)
    for i, h in enumerate(rows):
        k = int(getattr(h, "k", 0))
        if k <= 0: continue
        means[i, :k] = h.means.detach().to(means.dtype)
        for g in range(k):
            idxs = h.idx[h.cid == g].long().to("cpu")
            if idxs.numel() == 0: continue
            row_ids.append(torch.full((idxs.numel(),), i, dtype=torch.long))
            col_idx.append(idxs)
            group_ids.append(torch.full((idxs.numel(),), g, dtype=torch.long))
    row_ids = torch.cat(row_ids) if row_ids else torch.empty(0, dtype=torch.long)
    col_idx = torch.cat(col_idx) if col_idx else torch.empty(0, dtype=torch.long)
    group_ids = torch.cat(group_ids) if group_ids else torch.empty(0, dtype=torch.long)
    dest = row_ids * (Gmax if Gmax > 0 else 1) + group_ids
    return dest, col_idx, means, Gmax

@torch.inference_mode()
def pack_rows_to_indices_conv(rows: nn.ModuleList, out_channels: int, in_channels: int, kernel: Tuple[int, int], groups: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    kH, kW = map(int, kernel)
    seg = (in_channels // max(1, groups)) * kH * kW
    Gs = [int(getattr(h, "k", 0)) for h in rows]
    Gmax = max(Gs) if Gs else 0
    row_ids, col_idx, group_ids = [], [], []
    means = torch.zeros(out_channels, Gmax, dtype=torch.float32)
    for oc, h in enumerate(rows):
        k = int(getattr(h, "k", 0))
        if k <= 0: continue
        means[oc, :k] = h.means.detach().to(means.dtype)
        idx_offset = ((oc * groups) // max(1, out_channels)) * seg if groups > 1 else 0
        for g in range(k):
            idxs = h.idx[h.cid == g].long().to("cpu")
            if idxs.numel() == 0: continue
            row_ids.append(torch.full((idxs.numel(),), oc, dtype=torch.long))
            col_idx.append(idxs + int(idx_offset))
            group_ids.append(torch.full((idxs.numel(),), g, dtype=torch.long))
    row_ids = torch.cat(row_ids) if row_ids else torch.empty(0, dtype=torch.long)
    col_idx = torch.cat(col_idx) if col_idx else torch.empty(0, dtype=torch.long)
    group_ids = torch.cat(group_ids) if group_ids else torch.empty(0, dtype=torch.long)
    dest = row_ids * (Gmax if Gmax > 0 else 1) + group_ids
    return dest, col_idx, means, Gmax
