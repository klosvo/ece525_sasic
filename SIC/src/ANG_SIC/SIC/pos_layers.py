from typing import List, Tuple, Optional

import torch

from src.ANG_SIC.core.numeric import clamp_decimals, round_tensor, to_2tuple


@torch.no_grad()
def is_masked_linear_like(child) -> bool:
    return (
        hasattr(child, "weight")
        and child.weight is not None
        and child.weight.ndim == 2
        and not isinstance(child, torch.nn.Linear)
    )

@torch.no_grad()
def _standardize_for_masked_linear_like(child, rounding_decimals: int, membership):
    W = child.weight.detach()
    if hasattr(child, "mask") and child.mask is not None:
        W = W * child.mask.detach().to(W.dtype)
    return _standardize_membership_or_build(W, rounding_decimals, membership)

def _row_clusters_from_weights(row: torch.Tensor, decimals: int) -> List[Tuple[torch.Tensor, float]]:
    nz = row.ne(0)
    if not bool(nz.any()):
        return []
    rounded = round_tensor(row[nz], decimals=decimals)
    uniq_vals, inv = torch.unique(rounded, sorted=True, return_inverse=True)
    source_idx = torch.nonzero(nz).view(-1)
    clusters: List[Tuple[torch.Tensor, float]] = []
    for k in range(uniq_vals.numel()):
        idxs = source_idx[inv == k].to(dtype=torch.long)
        if idxs.numel() == 0:
            continue
        mean_val = row.index_select(0, idxs).mean().item()
        clusters.append((idxs, float(mean_val)))
    return clusters

@torch.no_grad()
def build_linear_clusters_from_weights(
    W: torch.Tensor, rounding_decimals: int = 6
) -> List[List[Tuple[torch.Tensor, float]]]:
    rows, _ = W.shape
    out: List[List[Tuple[torch.Tensor, float]]] = []
    for r in range(rows):
        out.append(_row_clusters_from_weights(W[r], rounding_decimals))
    return out

@torch.no_grad()
def build_conv_clusters_from_weights(
    W: torch.Tensor, rounding_decimals: int = 6
) -> List[List[Tuple[torch.Tensor, float]]]:
    outC, inC, kH, kW = W.shape
    L = inC * kH * kW
    Wf = W.reshape(outC, L)
    out: List[List[Tuple[torch.Tensor, float]]] = []
    for oc in range(outC):
        out.append(_row_clusters_from_weights(Wf[oc], rounding_decimals))
    return out

def _standardize_membership_or_build(
    W: torch.Tensor,
    rounding_decimals: int,
    membership: Optional[List[List[Tuple[torch.Tensor, float]]]],
) -> List[List[Tuple[torch.Tensor, float]]]:
    if not isinstance(membership, list) or len(membership) == 0:
        return (
            build_linear_clusters_from_weights(W, rounding_decimals)
            if W.dim() == 2
            else build_conv_clusters_from_weights(W, rounding_decimals)
        )
    std: List[List[Tuple[torch.Tensor, float]]] = []
    rows = W.size(0)
    for r in range(rows):
        row_m = membership[r] if r < len(membership) else None
        if not row_m:
            std.append(_row_clusters_from_weights(W[r].reshape(-1), rounding_decimals))
            continue
        if isinstance(row_m[0], tuple) and len(row_m[0]) == 2:
            fixed_row = []
            for idxs, mean_val in row_m:
                idxs = idxs.detach().clone().long().to("cpu")
                fixed_row.append((idxs, float(mean_val)))
            std.append(fixed_row)
            continue
        if torch.is_tensor(row_m[0]) or isinstance(row_m, list):
            base_row = W[r].reshape(-1)
            fixed_row = []
            for idxs in row_m:
                if not torch.is_tensor(idxs) or idxs.numel() == 0:
                    continue
                idxs = idxs.detach().clone().long().to("cpu")
                mean_val = base_row.index_select(0, idxs.to(base_row.device)).mean().item()
                fixed_row.append((idxs, float(mean_val)))
            if not fixed_row:
                fixed_row = _row_clusters_from_weights(base_row, rounding_decimals)
            std.append(fixed_row)
            continue
        std.append(_row_clusters_from_weights(W[r].reshape(-1), rounding_decimals))
    return std

class _PackedRow(torch.nn.Module):
    def __init__(self, idx_list: List[torch.Tensor], means_list: List[float]):
        super().__init__()
        if len(idx_list) == 0:
            self.register_buffer("idx", torch.zeros(0, dtype=torch.long))
            self.register_buffer("cid", torch.zeros(0, dtype=torch.long))
            self.register_buffer("means", torch.zeros(0, dtype=torch.float32))
            self.k = 0
            return
        idxs = torch.cat([t.view(-1).long().to("cpu") for t in idx_list], dim=0)
        counts = torch.tensor([int(t.numel()) for t in idx_list], dtype=torch.long)
        cid = torch.repeat_interleave(torch.arange(len(idx_list), dtype=torch.long), counts)
        self.register_buffer("idx", idxs)
        self.register_buffer("cid", cid)
        self.register_buffer("means", torch.tensor(means_list, dtype=torch.float32))
        self.k = len(idx_list)

class SICLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[torch.Tensor],
        clusters_per_row: List[List[Tuple[torch.Tensor, float]]],
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bias = None if bias is None else torch.nn.Parameter(bias.detach().clone())
        self.rows = torch.nn.ModuleList()
        Gs = []
        for row in clusters_per_row:
            idx_list, mean_list = [], []
            for idxs, m in row:
                idx_list.append(idxs.detach().clone().long().to("cpu"))
                mean_list.append(float(m))
            holder = _PackedRow(idx_list, mean_list)
            self.rows.append(holder)
            Gs.append(int(holder.k))
        self.Gmax = max(Gs) if Gs else 0
        row_ids, col_idx, group_ids = [], [], []
        means = torch.zeros(self.out_features, self.Gmax, dtype=torch.float32)
        for i, h in enumerate(self.rows):
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
        dest = row_ids * (self.Gmax if self.Gmax > 0 else 1) + group_ids
        self.register_buffer("dest", dest)
        self.register_buffer("col_idx", col_idx)
        self.register_buffer("means", means)
        self._cols = int(self.out_features * max(1, self.Gmax))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        if self.Gmax == 0 or self.col_idx.numel() == 0:
            y = x.new_zeros(B, self.out_features)
            if self.bias is not None:
                y = y + self.bias.to(dtype=x.dtype, device=x.device)
            return y
        means = self.means if self.means.dtype == x.dtype else self.means.to(x.dtype)
        src = x.index_select(1, self.col_idx)
        out2d = x.new_zeros(B, self._cols)
        out2d.scatter_add_(1, self.dest.expand(B, -1), src)
        sums = out2d.view(B, self.out_features, self.Gmax)
        y = (sums * means).sum(dim=2)
        if self.bias is not None:
            y = y + self.bias.to(dtype=x.dtype, device=x.device)
        return y

class SICConv2d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d, clusters_per_out: List[List[Tuple[torch.Tensor, float]]]):
        super().__init__()
        self.in_channels = int(conv.in_channels)
        self.out_channels = int(conv.out_channels)
        self.kernel_size = to_2tuple(conv.kernel_size)
        self.stride = to_2tuple(conv.stride)
        self.padding = to_2tuple(conv.padding)
        self.dilation = to_2tuple(conv.dilation)
        self.groups = int(conv.groups)
        self.bias = None if conv.bias is None else torch.nn.Parameter(conv.bias.detach().clone())
        self.out_rows = torch.nn.ModuleList()
        for row in clusters_per_out:
            idx_list, mean_list = [], []
            for idxs, m in row:
                idx_list.append(idxs.detach().clone().long().to("cpu"))
                mean_list.append(float(m))
            self.out_rows.append(_PackedRow(idx_list, mean_list))
        kH, kW = self.kernel_size
        inC_g = self.in_channels // max(1, self.groups)
        seg = inC_g * kH * kW
        for oc, holder in enumerate(self.out_rows):
            if holder.k == 0:
                continue
            if holder.idx.numel() > 0:
                max_idx = int(holder.idx.max().item())
                if max_idx >= seg:
                    raise ValueError(f"SICConv2d cluster index {max_idx} out of range (seg={seg})")
        Gs = [int(getattr(h, "k", 0)) for h in self.out_rows]
        self.Gmax = max(Gs) if Gs else 0
        row_ids, col_idx, group_ids = [], [], []
        means = torch.zeros(self.out_channels, self.Gmax, dtype=torch.float32)
        for oc, h in enumerate(self.out_rows):
            k = int(getattr(h, "k", 0))
            if k <= 0:
                continue
            means[oc, :k] = h.means.detach().to(means.dtype)
            g_idx = (oc * self.groups) // max(1, self.out_channels) if self.groups > 1 else 0
            idx_offset = g_idx * seg
            for g in range(k):
                idxs = h.idx[h.cid == g].long().to("cpu")
                if idxs.numel() == 0:
                    continue
                row_ids.append(torch.full((idxs.numel(),), oc, dtype=torch.long))
                col_idx.append(idxs + int(idx_offset))
                group_ids.append(torch.full((idxs.numel(),), g, dtype=torch.long))
        row_ids = torch.cat(row_ids) if row_ids else torch.empty(0, dtype=torch.long)
        col_idx = torch.cat(col_idx) if col_idx else torch.empty(0, dtype=torch.long)
        group_ids = torch.cat(group_ids) if group_ids else torch.empty(0, dtype=torch.long)
        dest = row_ids * (self.Gmax if self.Gmax > 0 else 1) + group_ids
        self.register_buffer("dest", dest)
        self.register_buffer("col_idx", col_idx)
        self.register_buffer("means", means)
        self._G = int(max(1, self.Gmax))
        self._cached_hw: Tuple[int, int, int, int] | None = None

    def _out_hw(self, H: int, W: int) -> Tuple[int, int]:
        if self._cached_hw is not None and self._cached_hw[0] == H and self._cached_hw[1] == W:
            return self._cached_hw[2], self._cached_hw[3]
        Ho = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        Wo = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        self._cached_hw = (H, W, Ho, Wo)
        return Ho, Wo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.means.numel() == 0 or self.out_channels == 0 or self.col_idx.numel() == 0:
            Ho, Wo = self._out_hw(H, W)
            y = x.new_zeros(B, self.out_channels, Ho, Wo)
            if self.bias is not None:
                y = y + self.bias.to(dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
            return y
        means = self.means if self.means.dtype == x.dtype else self.means.to(x.dtype)
        patches = torch.nn.functional.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        HW = patches.size(-1)
        src = patches.index_select(1, self.col_idx)
        out2d = patches.new_zeros(B, self.out_channels * self._G, HW)
        scatter_dest = self.dest.view(1, -1, 1).expand(B, -1, HW)
        out2d.scatter_add_(1, scatter_dest, src)
        sums = out2d.view(B, self.out_channels, self._G, HW)
        y_cols = (sums * means.view(1, self.out_channels, self._G, 1)).sum(dim=2)
        Ho, Wo = self._out_hw(H, W)
        y = y_cols.view(B, self.out_channels, Ho, Wo)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype, device=y.device).view(1, -1, 1, 1)
        return y

@torch.no_grad()
def _standardize_for_linear(
    child: torch.nn.Linear, rounding_decimals: int, membership
) -> List[List[Tuple[torch.Tensor, float]]]:
    W = child.weight.detach()
    return _standardize_membership_or_build(W, rounding_decimals, membership)

@torch.no_grad()
def _standardize_for_conv(
    child: torch.nn.Conv2d, rounding_decimals: int, membership
) -> List[List[Tuple[torch.Tensor, float]]]:
    W = child.weight.detach()
    return _standardize_membership_or_build(W, rounding_decimals, membership)

@torch.no_grad()
def replace_linear_and_conv_with_sic(module: torch.nn.Module, rounding_decimals: int = 6) -> None:
    rd = clamp_decimals(rounding_decimals)
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            clusters = getattr(child, "_sic_pos_clusters", None)
            clusters = _standardize_for_linear(child, rd, clusters)
            new_layer = SICLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias,
                clusters_per_row=clusters,
            ).to(child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, new_layer)
        elif isinstance(child, torch.nn.Conv2d):
            clusters = getattr(child, "_sic_pos_clusters", None)
            clusters = _standardize_for_conv(child, rd, clusters)
            new_layer = SICConv2d(child, clusters).to(child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, new_layer)
        elif is_masked_linear_like(child):
            clusters = getattr(child, "_sic_pos_clusters", None)
            clusters = _standardize_for_masked_linear_like(child, rd, clusters)
            out_features, in_features = child.weight.shape
            bias = getattr(child, "bias", None)
            new_layer = SICLinear(
                in_features=int(in_features),
                out_features=int(out_features),
                bias=bias,
                clusters_per_row=clusters,
            ).to(child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, new_layer)
        else:
            replace_linear_and_conv_with_sic(child, rd)

@torch.no_grad()
def replace_single_module_with_sic(
    root_module: torch.nn.Module, dotted_name: str, rounding_decimals: int = 6
) -> None:
    parts = dotted_name.split(".")
    parent = root_module
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return
        parent = getattr(parent, p)
    child_name = parts[-1]
    if not hasattr(parent, child_name):
        return
    child = getattr(parent, child_name)
    rd = clamp_decimals(rounding_decimals)
    if isinstance(child, torch.nn.Linear):
        clusters = getattr(child, "_sic_pos_clusters", None)
        clusters = _standardize_for_linear(child, rd, clusters)
        new_layer = SICLinear(
            in_features=child.in_features,
            out_features=child.out_features,
            bias=child.bias,
            clusters_per_row=clusters,
        ).to(child.weight.device, dtype=child.weight.dtype)
        setattr(parent, child_name, new_layer)
    elif isinstance(child, torch.nn.Conv2d):
        clusters = getattr(child, "_sic_pos_clusters", None)
        clusters = _standardize_for_conv(child, rd, clusters)
        new_layer = SICConv2d(child, clusters).to(child.weight.device, dtype=child.weight.dtype)
        setattr(parent, child_name, new_layer)
