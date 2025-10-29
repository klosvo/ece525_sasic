from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ANG_SIC.core.module_path import find_parent
from .pos_common import rows_container_name, can_torch_compile


def build_fast_pos_linear(m: nn.Module) -> nn.Module:
    holders = getattr(m, "rows", None)
    if holders is None:
        cname = rows_container_name(m)
        if cname is None:
            raise ValueError("build_fast_pos_linear: no rows-like container found")
        holders = getattr(m, cname)

    out_features = len(holders)
    in_features = getattr(m, "in_features", None)
    if in_features is None:
        max_idx = 0
        for h in holders:
            if getattr(h, "idx", None) is not None and h.idx.numel():
                max_idx = max(max_idx, int(h.idx.max().item()))
        in_features = max_idx + 1

    Gs = [int(getattr(h, "k", 0)) for h in holders]
    Gmax = max(Gs) if Gs else 0

    row_ids, col_idx, group_ids = [], [], []
    means = torch.zeros(out_features, Gmax, dtype=torch.float32)

    for i, h in enumerate(holders):
        k = int(getattr(h, "k", 0))
        if k <= 0:
            continue
        means[i, :k] = h.means.detach().to(means.dtype)
        for g in range(k):
            idxs = h.idx[h.cid == g].long().to("cpu")
            if idxs.numel() == 0:
                continue
            row_ids.append(torch.full((idxs.numel(),), i, dtype=torch.long))
            col_idx.append(idxs)
            group_ids.append(torch.full((idxs.numel(),), g, dtype=torch.long))

    row_ids = torch.cat(row_ids) if row_ids else torch.empty(0, dtype=torch.long)
    col_idx = torch.cat(col_idx) if col_idx else torch.empty(0, dtype=torch.long)
    group_ids = torch.cat(group_ids) if group_ids else torch.empty(0, dtype=torch.long)
    dest = row_ids * (Gmax if Gmax > 0 else 1) + group_ids

    bias = None
    if getattr(m, "bias", None) is not None:
        b = m.bias.detach() if hasattr(m.bias, "detach") else m.bias
        bias = b.clone().to(torch.float32)

    class FastPoSLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.Gmax = int(Gmax)
            self._cols = int(self.out_features * max(1, self.Gmax))
            self.register_buffer("dest", dest)
            self.register_buffer("col_idx", col_idx)
            self.register_buffer("means", means)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B = x.size(0)
            device, dtype = x.device, x.dtype
            dest = self.dest.to(device)
            col_idx = self.col_idx.to(device)
            means = self.means.to(device=device, dtype=dtype)
            bias = self.bias.to(device=device, dtype=dtype) if getattr(self, "bias", None) is not None else None
            if self.Gmax == 0 or col_idx.numel() == 0:
                y = x.new_zeros(B, self.out_features)
                if bias is not None:
                    y = y + bias
                return y
            src = x.index_select(1, col_idx)
            out2d = x.new_zeros(B, self._cols)
            out2d.scatter_add_(1, dest.expand(B, -1), src)
            sums = out2d.view(B, self.out_features, self.Gmax)
            y = (sums * means).sum(dim=2)
            if bias is not None:
                y = y + bias
            return y

    return FastPoSLinear()


def build_fast_pos_conv2d(m: nn.Module) -> nn.Module:
    inC = int(m.in_channels)
    kH, kW = int(m.kernel_size[0]), int(m.kernel_size[1])

    holders = m.out_rows
    out_channels = len(holders)
    Gs = [int(getattr(h, "k", 0)) for h in holders]
    Gmax = max(Gs) if Gs else 0

    row_ids, col_idx, group_ids = [], [], []
    means = torch.zeros(out_channels, Gmax, dtype=torch.float32)

    for oc, h in enumerate(holders):
        k = int(getattr(h, "k", 0))
        if k <= 0:
            continue
        means[oc, :k] = h.means.detach().to(means.dtype)
        for g in range(k):
            idxs = h.idx[h.cid == g].long().to("cpu")
            if idxs.numel() == 0:
                continue
            row_ids.append(torch.full((idxs.numel(),), oc, dtype=torch.long))
            col_idx.append(idxs)
            group_ids.append(torch.full((idxs.numel(),), g, dtype=torch.long))

    row_ids = torch.cat(row_ids) if row_ids else torch.empty(0, dtype=torch.long)
    col_idx = torch.cat(col_idx) if col_idx else torch.empty(0, dtype=torch.long)
    group_ids = torch.cat(group_ids) if group_ids else torch.empty(0, dtype=torch.long)
    dest = row_ids * (Gmax if Gmax > 0 else 1) + group_ids

    bias = None
    if getattr(m, "bias", None) is not None:
        b = m.bias.detach() if hasattr(m.bias, "detach") else m.bias
        bias = b.clone().to(torch.float32)

    class FastPoSConv2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = inC
            self.out_channels = out_channels
            self.kernel_size = (kH, kW)
            self.stride = m.stride
            self.padding = m.padding
            self.dilation = m.dilation
            self.groups = 1
            self.register_buffer("dest", dest)
            self.register_buffer("col_idx", col_idx)
            self.register_buffer("means", means)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None
            self._cached_hw: Tuple[int, int, int, int] | None = None
            self._G = int(max(1, means.size(1)))

        def _out_hw(self, H: int, W: int) -> Tuple[int, int]:
            if self._cached_hw is not None and self._cached_hw[0] == H and self._cached_hw[1] == W:
                return self._cached_hw[2], self._cached_hw[3]
            Ho = ((H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1)
            Wo = ((W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1)
            self._cached_hw = (H, W, Ho, Wo)
            return Ho, Wo

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, _, H, W = x.shape
            device, dtype = x.device, x.dtype
            dest = self.dest.to(device)
            col_idx = self.col_idx.to(device)
            means = self.means.to(device=device, dtype=dtype)
            bias = self.bias.to(device=device, dtype=dtype) if getattr(self, "bias", None) is not None else None
            if means.numel() == 0 or self.out_channels == 0 or col_idx.numel() == 0:
                Ho, Wo = self._out_hw(H, W)
                y = x.new_zeros(B, self.out_channels, Ho, Wo)
                if bias is not None:
                    y = y + bias.view(1, -1, 1, 1)
                return y
            patches = F.unfold(
                x,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
            )
            HW = patches.shape[-1]
            src = patches.index_select(1, col_idx)
            out2d = patches.new_zeros(B, self.out_channels * self._G, HW)
            scatter_dest = dest.view(1, -1, 1).expand(B, -1, HW)
            out2d.scatter_add_(1, scatter_dest, src)
            sums = out2d.view(B, self.out_channels, self._G, HW)
            y_cols = (sums * means.view(1, self.out_channels, self._G, 1)).sum(dim=2)
            Ho, Wo = self._out_hw(H, W)
            y = y_cols.view(B, self.out_channels, Ho, Wo)
            if bias is not None:
                y = y + bias.view(1, -1, 1, 1)
            return y

    return FastPoSConv2d()


def accelerate_pos_layers(model: nn.Module, use_torch_compile: bool = True) -> Tuple[nn.Module, int]:
    def _model_device_dtype(m: nn.Module):
        for p in m.parameters(recurse=True):
            return p.device, p.dtype
        for b in m.buffers(recurse=True):
            return b.device, b.dtype
        return torch.device("cpu"), torch.float32

    replaced = 0
    compile_ok = use_torch_compile and can_torch_compile()
    model_device, model_dtype = _model_device_dtype(model)

    for mod in list(model.modules()):
        fast = None
        if (rows_container_name(mod) is not None) and not hasattr(mod, "weight"):
            fast = build_fast_pos_linear(mod)
        elif hasattr(mod, "out_rows") and hasattr(mod, "in_channels") and not hasattr(mod, "weight"):
            fast = build_fast_pos_conv2d(mod)
        if fast is None:
            continue
        fast = fast.to(device=model_device, dtype=model_dtype)
        if compile_ok:
            fast = torch.compile(fast, mode="max-autotune", fullgraph=True, dynamic=False)
        parent, name = find_parent(model, mod)
        setattr(parent, name, fast)
        replaced += 1

    return model, replaced
