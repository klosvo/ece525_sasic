from __future__ import annotations
from collections import defaultdict
from typing import Optional
import numpy as np
import torch
from torch import nn

from .verify_helpers import to_dense_and_bias, _mk_printer, _collect_interesting_modules, _try_infer_shapes, _ops_and_firsts, _infer_conv_hw_static, _is_fast_pos_linear, _is_fast_pos_conv2d, _is_indexed_sparse_linear, _is_masked_linear_like, _is_masked_conv_like, _indexed_sparse_nnz_per_out, _fast_pos_groups_and_nz_means, _group_counts_1d, _fast_pos_group_stats_linear, _fast_pos_group_stats_conv, _format_layer_label
from src.ANG_SIC.core.feature import model_device

def verify_model(
    path: str,
    model: nn.Module,
    profiler: Optional[object],
    sample: torch.Tensor | None = None,
    try_shapes: list | None = None,
    rounding_decimals: int | None = None,
    input_spec: tuple[str, int | tuple[int, int, int]] | None = None,
    verbose: bool = True,
    printer=print,
    *,
    force_hook_hw: bool = False,
    print_conv_shapes: bool = False,
    print_linear_shapes: bool = False,
    print_table: bool = False,
) -> dict:
    _p = _mk_printer(printer, verbose)

    def _model_has_pos(m: nn.Module) -> bool:
        for mod in m.modules():
            if _is_fast_pos_linear(mod) or _is_fast_pos_conv2d(mod):
                return True
        return False

    def _is_broadcastable(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
        la, lb = len(a), len(b)
        for i in range(1, max(la, lb) + 1):
            da = a[-i] if i <= la else 1
            db = b[-i] if i <= lb else 1
            if da != db and da != 1 and db != 1:
                return False
        return True

    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    dev = model_device(model)

    interesting = _collect_interesting_modules(model)
    ok, layer_shapes, tried = _try_infer_shapes(model, interesting, sample, input_spec, try_shapes, dev, _p)
    conv_order, ops, first_linear = _ops_and_firsts(model)
    inferred_conv_hw = _infer_conv_hw_static(model, interesting, conv_order, ops, first_linear, layer_shapes, ok, force_hook_hw, _p)

    totals = defaultdict(int)
    total_bias_count = 0
    layer_info = {}
    total_dense_mults = total_dense_adds = 0
    total_pos_mults = total_pos_adds = 0
    total_sparse_mults = total_sparse_adds = 0
    total_macs_executed = 0
    total_macs_naive_sparse = 0
    total_pairMAC_flops = 0
    total_additive_flops = 0
    weights_rows: list[dict] = []

    _p("Per-layer analysis:")
    _p("-" * 60)

    rd = int(rounding_decimals) if rounding_decimals is not None else None

    for name, module in model.named_modules():
        w, bcnt = to_dense_and_bias(module)
        is_fastpos_lin = _is_fast_pos_linear(module)
        is_fastpos_conv = _is_fast_pos_conv2d(module)
        is_indexed_sparse = _is_indexed_sparse_linear(module)
        if (w is None) and not (is_fastpos_lin or is_fastpos_conv or is_indexed_sparse):
            continue

        is_linear_like = (w is not None and w.ndim == 2) or is_fastpos_lin or is_indexed_sparse
        is_conv_like = (w is not None and w.ndim == 4) or is_fastpos_conv
        if not (is_linear_like or is_conv_like):
            continue

        has_weight_param = hasattr(module, "weight") and getattr(module, "weight") is not None
        has_bias = bcnt > 0
        w_eff = w
        is_mask_lin = _is_masked_linear_like(module)
        is_mask_conv = _is_masked_conv_like(module)
        if w is not None and (is_mask_lin or is_mask_conv) and hasattr(module, "mask") and isinstance(module.mask, torch.Tensor):
            m = module.mask.detach().cpu()
            if m.shape != w.shape and m.numel() == w.numel():
                m = m.view_as(w)
            if _is_broadcastable(tuple(w.shape), tuple(m.shape)):
                w_eff = w * m.to(w.dtype)

        display_unique = 0
        display_total_params = 0
        pct_unique_for_display = 0.0
        compression_x = 0.0
        nz = 0
        sparsity_pct = 0.0

        if is_indexed_sparse:
            nnz_per_out = _indexed_sparse_nnz_per_out(module)
            nnz_total = int(nnz_per_out.sum().item())
            theta_flat = module.theta.detach().cpu().view(-1)
            nz = int(torch.count_nonzero(theta_flat).item())
            rd_eff = 6 if rd is None else int(rd)
            scale = float(10 ** max(0, rd_eff))
            if hasattr(module, "row_ptr") and isinstance(getattr(module, "row_ptr"), torch.Tensor) and module.row_ptr.numel() >= 2:
                rp = module.row_ptr.detach().cpu().long().view(-1)
                vals = theta_flat
                groups_total = 0
                for r in range(rp.numel() - 1):
                    s = int(rp[r].item()); e = int(rp[r + 1].item())
                    if e > s:
                        v_round = torch.round(vals[s:e] * scale) / scale
                        groups_total += int(torch.unique(v_round).numel())
            else:
                idx = module.indices.detach().cpu().long()
                rows = idx[0]
                K = int(getattr(module, "out_features", getattr(module, "K", rows.max().item() + 1)))
                groups_total = 0
                for r in range(K):
                    mask_r = (rows == r)
                    if bool(mask_r.any()):
                        v = theta_flat[mask_r]
                        v_round = torch.round(v * scale) / scale
                        groups_total += int(torch.unique(v_round).numel())
            display_unique = groups_total
            display_total_params = nnz_total
            pct_unique_for_display = (groups_total / max(1, nnz_total)) * 100.0
            compression_x = (nnz_total / max(1, groups_total)) if groups_total > 0 else 0.0
            Din = int(getattr(module, "in_features", getattr(module, "D", 0)) or 0)
            Kout = int(getattr(module, "out_features", getattr(module, "K", 0)) or 0)
            dense_params = Din * Kout if (Din > 0 and Kout > 0) else None
            sparsity_pct = (1.0 - (nnz_total / max(1, dense_params))) * 100.0 if dense_params else 0.0
            totals["total_params"] += nnz_total
            totals["total_nonzero"] += nz
            totals["total_unique"] += display_unique

        elif is_fastpos_lin or is_fastpos_conv:
            groups_total, nz_means = _fast_pos_groups_and_nz_means(module)
            display_unique = groups_total
            display_total_params = max(1, groups_total)
            pct_unique_for_display = 100.0 if groups_total > 0 else 0.0
            compression_x = 1.0
            nz = int(nz_means)
            sparsity_pct = ((display_total_params - nz) / max(1, display_total_params)) * 100.0
            totals["total_params"] += display_total_params
            totals["total_nonzero"] += nz
            totals["total_unique"] += display_unique

        else:
            total_params = int(w_eff.numel()) if (w_eff is not None) else 0
            if total_params == 0:
                nz = 0
                uniq = 0
                sparsity_pct = 0.0
                compression_x = 0.0
            else:
                w_flat = w_eff.view(-1)
                nz = int(torch.count_nonzero(w_flat).item())
                nz_vals = w_flat[w_flat != 0]
                uniq = int(torch.unique(nz_vals).numel()) if nz > 0 else 0
                sparsity_pct = (total_params - nz) / max(1, total_params) * 100.0
                compression_x = (total_params / max(1, uniq)) if uniq > 0 else 0.0
            display_unique = int(uniq if total_params > 0 else 0)
            display_total_params = int(total_params)
            pct_unique_for_display = ((display_unique / max(1, display_total_params)) * 100.0) if display_total_params > 0 else 0.0
            totals["total_params"] += display_total_params
            totals["total_nonzero"] += nz
            totals["total_unique"] += display_unique

        total_bias_count += bcnt

        HoWo = None
        shp = layer_shapes.get(name)
        hook_has_hw = shp and shp.get("out") and isinstance(shp["out"], tuple) and len(shp["out"]) == 4
        if hook_has_hw:
            _, _, Ho, Wo = shp["out"]
            HoWo = int(Ho) * int(Wo)
            HoWo_source = "hook"
        elif (not force_hook_hw) and (name in inferred_conv_hw):
            H, W = inferred_conv_hw[name]
            HoWo = int(H) * int(W)
            HoWo_source = "static_infer"
        else:
            HoWo_source = "unknown" if is_conv_like else "-"

        dense_mults = 0
        dense_adds = 0
        if w is not None and is_linear_like and w.ndim == 2:
            of, inf = int(w.shape[0]), int(w.shape[1])
            dense_mults = of * inf
            dense_adds = of * (inf - 1 + (1 if has_bias else 0))
        elif w is not None and is_conv_like and w.ndim == 4 and HoWo is not None:
            Co = int(w.shape[0])
            eff_in = int(w.shape[1])
            kH, kW = int(w.shape[-2]), int(w.shape[-1])
            mul_per_out = eff_in * kH * kW
            add_per_out = (mul_per_out - 1) + (1 if has_bias else 0)
            dense_mults = int(Co * HoWo * mul_per_out)
            dense_adds = int(Co * HoWo * add_per_out)

        pos_mults = 0
        pos_adds = 0
        if w is not None:
            if is_linear_like and w.ndim == 2:
                of = int(w_eff.shape[0])
                for j in range(of):
                    groups = _group_counts_1d(w_eff[j, :], rd)
                    G = len(groups)
                    pos_mults += G
                    pos_adds += sum(c - 1 for c in groups.values()) + (G - 1 if G > 0 else 0) + (1 if has_bias else 0)
            elif is_conv_like and w.ndim == 4 and HoWo is not None:
                Co = int(w_eff.shape[0])
                for oc in range(Co):
                    flat = w_eff[oc, ...].contiguous().view(-1)
                    groups = _group_counts_1d(flat, rd)
                    G = len(groups)
                    adds_pos = sum(c - 1 for c in groups.values()) + (G - 1 if G > 0 else 0) + (1 if has_bias else 0)
                    pos_mults += G * HoWo
                    pos_adds += adds_pos * HoWo
        else:
            if is_fastpos_lin:
                G_list, adds_list = _fast_pos_group_stats_linear(module)
                pos_mults = int(sum(G_list))
                pos_adds = int(sum(a + (1 if has_bias else 0) for a, g in zip(adds_list, G_list)))
            elif is_fastpos_conv and HoWo is not None:
                G_list, adds_list = _fast_pos_group_stats_conv(module)
                pos_mults = int(sum(g * HoWo for g in G_list))
                pos_adds = int(sum((a + (1 if has_bias else 0)) * HoWo for a in adds_list))

        sparse_mults = 0
        sparse_adds = 0
        if (w is not None) and is_mask_lin and w.ndim == 2:
            nnz_per_out = (w_eff != 0).sum(dim=1).to(torch.long)
            sparse_mults = int(nnz_per_out.sum().item())
            adds_per_out = (nnz_per_out - 1).clamp_min(0)
            if has_bias:
                adds_per_out = adds_per_out + 1
            sparse_adds = int(adds_per_out.sum().item())
        elif (w is not None) and is_mask_conv and w.ndim == 4 and HoWo is not None:
            nnz_per_oc = (w_eff != 0).view(w_eff.shape[0], -1).sum(dim=1).to(torch.long)
            sparse_mults = int((nnz_per_oc * HoWo).sum().item())
            adds_per_oc = (nnz_per_oc - 1).clamp_min(0)
            if has_bias:
                adds_per_oc = adds_per_oc + 1
            sparse_adds = int((adds_per_oc * HoWo).sum().item())
        elif is_indexed_sparse and is_linear_like:
            nnz_per_out = _indexed_sparse_nnz_per_out(module)
            sparse_mults = int(nnz_per_out.sum().item())
            adds_per_out = (nnz_per_out - 1).clamp_min(0)
            if has_bias:
                adds_per_out = adds_per_out + 1
            sparse_adds = int(adds_per_out.sum().item())

        total_dense_mults += dense_mults
        total_dense_adds += dense_adds
        total_pos_mults += pos_mults
        total_pos_adds += pos_adds
        total_sparse_mults += sparse_mults
        total_sparse_adds += sparse_adds

        N = 1
        if shp and shp.get("out") and len(shp["out"]) >= 1 and isinstance(shp["out"][0], int):
            N = int(shp["out"][0])

        if is_indexed_sparse:
            mode = "Sparse"
            mults_executed = sparse_mults
            adds_executed = sparse_adds
        elif not has_weight_param:
            mode = "PoS"
            mults_executed = pos_mults
            adds_executed = pos_adds
        elif _is_masked_linear_like(module) or _is_masked_conv_like(module):
            mode = "Sparse"
            mults_executed = sparse_mults
            adds_executed = sparse_adds
        else:
            mode = "Dense"
            mults_executed = dense_mults
            adds_executed = dense_adds

        layer_macs = N * mults_executed
        layer_pairMAC_flops = 2 * layer_macs
        layer_additive_flops = N * (mults_executed + adds_executed)

        layer_macs_naive = None
        if is_indexed_sparse and is_linear_like:
            nnz_all = int(_indexed_sparse_nnz_per_out(module).sum().item())
            layer_macs_naive = N * nnz_all
        elif w_eff is not None:
            nnz_all = int(torch.count_nonzero(w_eff).item())
            if is_linear_like:
                layer_macs_naive = N * nnz_all
            elif is_conv_like and HoWo is not None:
                layer_macs_naive = N * HoWo * nnz_all

        total_macs_executed += layer_macs
        total_pairMAC_flops += layer_pairMAC_flops
        total_additive_flops += layer_additive_flops
        if layer_macs_naive is not None:
            total_macs_naive_sparse += layer_macs_naive

        units_out = None
        neurons_out = None
        channels_out = None
        unit_str = ""
        if is_linear_like:
            units_out = int(getattr(module, "out_features", getattr(module, "K", 0))) if (w is None or w.ndim != 2) else int(w.shape[0])
            totals["total_linear_units"] += int(units_out or 0)
            unit_str = f", units={units_out:,}"
        elif is_conv_like:
            if w is not None and w.ndim == 4:
                channels_out = int(w.shape[0])
            else:
                channels_out = int(getattr(module, "out_channels", 0))
            if HoWo is not None:
                neurons_out = int(channels_out * HoWo)
                totals["total_conv_neurons"] += neurons_out
                unit_str = f", neurons={neurons_out:,}"
            else:
                totals["total_conv_channels"] += int(channels_out or 0)
                unit_str = f", channels={channels_out:,}"

        if mode == "PoS":
            m_str = f"PoS M={pos_mults:,} A={pos_adds:,}"
        elif mode == "Sparse":
            m_str = f"Sparse M={sparse_mults:,} A={sparse_adds:,}"
        else:
            m_str = f"Dense M={dense_mults:,} A={dense_adds:,}"

        shape_str = ""
        if (is_conv_like and print_conv_shapes) or (is_linear_like and print_linear_shapes):
            in_s = tuple(shp["in"]) if (shp and shp.get("in")) else None
            out_s = tuple(shp["out"]) if (shp and shp.get("out")) else None
            shape_str = f"  ⮑ IN={in_s} OUT={out_s}  (HoWo_source={HoWo_source})"

        _p(
            f"  {name:<12s}: {display_unique:7d}/{display_total_params:8d} ({pct_unique_for_display:6.2f}% unique, "
            f"{nz:7d} nonzero, {sparsity_pct:5.1f}% sparse, {compression_x:7.1f}x compressed, "
            f"{bcnt} bias, {m_str}{unit_str})"
        )
        if shape_str:
            _p(shape_str)

        layer_label = _format_layer_label(name, is_conv_like, is_linear_like)

        filters = None
        kernel_str = "-"
        stride_str = "-"
        units_tbl = None
        mults_tbl = 0
        count_tbl = 0

        if is_conv_like:
            if w is not None and w.ndim == 4:
                Co = int(w.shape[0]); Ci = int(w.shape[1]); kH = int(w.shape[-2]); kW = int(w.shape[-1])
            else:
                Co = int(getattr(module, "out_channels", 0))
                Ci = int(getattr(module, "in_channels", 0))
                ks = getattr(module, "kernel_size", (1, 1))
                kH = int(ks[0] if isinstance(ks, (tuple, list)) else ks)
                kW = int(ks[1] if isinstance(ks, (tuple, list)) else kH)
            st = getattr(module, "stride", (1, 1))
            sH = int(st[0] if isinstance(st, (tuple, list)) else st)
            sW = int(st[1] if isinstance(st, (tuple, list)) else sH)
            kernel_str = f"{kH}x{kW}"
            stride_str = f"{sH}x{sW}"
            filters = Co
            if _is_fast_pos_conv2d(module):
                G_list, _ = _fast_pos_group_stats_conv(module)
                groups_total = int(sum(G_list))
                count_tbl = int(groups_total)
                if HoWo is not None:
                    mults_tbl = int(groups_total * HoWo)
                if HoWo is not None:
                    units_tbl = int(Co * HoWo) * Ci
            else:
                if HoWo is not None:
                    units_tbl = int(Co * HoWo) * Ci
                    mults_tbl = int(dense_mults)
                if (w is not None and w.ndim == 4) or hasattr(module, "in_channels"):
                    count_tbl = int(Co * Ci * kH * kW)

        elif is_linear_like:
            if is_indexed_sparse:
                of = int(getattr(module, "out_features", getattr(module, "K", 0)))
                filters = of
                units_tbl = of
                mults_tbl = int(sum(_indexed_sparse_nnz_per_out(module).tolist()))
                count_tbl = int(mults_tbl)
            elif _is_fast_pos_linear(module):
                of = int(getattr(module, "out_features", 0))
                filters = of
                units_tbl = of
                G_list, _ = _fast_pos_group_stats_linear(module)
                groups_total = int(sum(G_list))
                mults_tbl = int(groups_total)
                count_tbl = int(groups_total)
            else:
                of = int(getattr(module, "out_features", 0)) if (w is None or w.ndim != 2) else int(w.shape[0])
                inf = int(getattr(module, "in_features", 0)) if (w is None or w.ndim != 2) else int(w.shape[1])
                filters = of
                units_tbl = of
                mults_tbl = int(of * inf)
                count_tbl = int(of * inf)

        if is_conv_like or is_linear_like:
            weights_rows.append({
                "layer": layer_label,
                "name": name,
                "filters": int(filters or 0),
                "kernel": kernel_str,
                "stride": stride_str,
                "units": (int(units_tbl) if units_tbl is not None else None),
                "multiplications": int(mults_tbl),
                "count": int(count_tbl),
            })

        layer_info[name] = {
            "total": int(display_total_params),
            "nonzero": int(nz),
            "unique": int(display_unique),
            "bias_count": int(bcnt),
            "sparsity_%": float(sparsity_pct),
            "compression_x": float(compression_x),
            "dense_multiplications": int(dense_mults),
            "dense_additions": int(dense_adds),
            "pos_multiplications": int(pos_mults),
            "pos_additions": int(pos_adds),
            "sparse_multiplications": int(sparse_mults),
            "sparse_additions": int(sparse_adds),
            "mode": "PoS" if (not has_weight_param) else ("Sparse" if (is_indexed_sparse or is_mask_lin or is_mask_conv) else "Dense"),
            "HoWo_source": HoWo_source,
            "batch_size": int(N),
            "macs": int(layer_macs),
            "macs_definition": "MACs == multiplications (× batch size)",
            "macs_naive_sparse": (int(layer_macs_naive) if layer_macs_naive is not None else None),
            "flops_pairMAC": int(layer_pairMAC_flops),
            "flops_additive": int(layer_additive_flops),
            "in_shape": tuple(shp["in"]) if (shp and shp.get("in")) else None,
            "out_shape": tuple(shp["out"]) if (shp and shp.get("out")) else None,
            "units_out": (int(units_out) if units_out is not None else None),
            "neurons_out": (int(neurons_out) if neurons_out is not None else None),
            "channels_out": (int(channels_out) if channels_out is not None else None),
            "weights_table": {
                "layer": layer_label,
                "filters": int(filters or 0),
                "kernel": kernel_str,
                "stride": stride_str,
                "units": (int(units_tbl) if units_tbl is not None else None),
                "multiplications": int(mults_tbl),
                "count": int(count_tbl),
            },
        }

    _p("-" * 60)
    total = totals["total_params"]
    uniq = totals["total_unique"]
    nz = totals["total_nonzero"]
    total_sparsity = (total - nz) / max(1, total) * 100.0 if total > 0 else 0.0
    total_comp = (total / max(1, uniq)) if uniq > 0 else 0.0
    pct_unique_total = (uniq / max(1, total)) * 100.0 if total > 0 else 0.0

    if total_dense_mults > 0 or total_dense_adds > 0:
        _p(f"TOTAL (Dense): M={total_dense_mults:,} A={total_dense_adds:,}")
    if _model_has_pos(model) and (total_pos_mults > 0 or total_pos_adds > 0):
        _p(f"TOTAL   (PoS): M={total_pos_mults:,} A={total_pos_adds:,}")
    if total_sparse_mults > 0 or total_sparse_adds > 0:
        _p(f"TOTAL (Sparse): M={total_sparse_mults:,} A={total_sparse_adds:,}")

    total_linear_units = int(totals.get("total_linear_units", 0))
    total_conv_neurons = int(totals.get("total_conv_neurons", 0))
    total_conv_channels = int(totals.get("total_conv_channels", 0))
    total_units_neurons = total_linear_units + (total_conv_neurons if total_conv_neurons else 0)
    _p(
        f"PARAMS: {uniq:7d}/{total:8d} ({pct_unique_total:6.2f}% unique, "
        f"{nz:7d} nonzero, {total_sparsity:5.1f}% sparse, {total_comp:7.1f}x compressed, {total_bias_count} bias total)"
    )
    _p(f"TOTAL MACs (per-forward): {total_macs_executed:,}")
    _p("=" * 60)

    total_weight_params = sum(r["count"] for r in weights_rows if r["count"] > 0)
    if print_table:
        if weights_rows:
            _p("Weights")
            _p(f"{'Layer':<14}{'Filters':>8} {'Kernel':>7} {'Stride':>7} {'Units':>10} {'Multiplications':>16} {'Count':>10} {'Distribution':>13}")
            for r in weights_rows:
                dist = (r["count"] / total_weight_params * 100.0) if total_weight_params > 0 else 0.0
                units_str = f"{r['units']:,}" if r["units"] is not None else "-"
                _p(f"{r['layer']:<14}{r['filters']:>8} {r['kernel']:>7} {r['stride']:>7} {units_str:>10} {r['multiplications']:>16,} {r['count']:>10,} {dist:>12.2f}%")

    results_weights = []
    for r in weights_rows:
        dist = (r["count"] / total_weight_params * 100.0) if total_weight_params > 0 else 0.0
        results_weights.append({**r, "distribution_pct": dist})

    results = {
        "layers": layer_info,
        "total": dict(totals),
        "total_bias_count": int(total_bias_count),
        "dense_ops_totals": {"multiplications": int(total_dense_mults), "additions": int(total_dense_adds)},
        "pos_ops_totals": {"multiplications": int(total_pos_mults), "additions": int(total_pos_adds)},
        "sparse_ops_totals": {"multiplications": int(total_sparse_mults), "additions": int(total_sparse_adds)},
        "macs_totals": {"per_forward": int(total_macs_executed), "naive_sparse": int(total_macs_naive_sparse)},
        "flops_totals": {"pairMAC": int(total_pairMAC_flops), "additive": int(total_additive_flops)},
        "dry_run_shapes": {k: {"in": v.get("in"), "out": v.get("out")} for k, v in layer_shapes.items()},
        "static_inferred_hw": inferred_conv_hw,
        "tried_input_shapes": tried,
        "rounding_decimals": rounding_decimals,
        "flags": {
            "force_hook_hw": bool(force_hook_hw),
            "print_conv_shapes": bool(print_conv_shapes),
            "print_linear_shapes": bool(print_linear_shapes),
        },
        "unit_totals": {
            "linear_units": total_linear_units,
            "conv_neurons": total_conv_neurons,
            "conv_channels_when_hw_unknown": total_conv_channels,
            "units_plus_neurons": total_units_neurons,
        },
        "weights_table": {
            "total_weight_params": int(total_weight_params),
            "rows": results_weights,
        },
    }
    if profiler is not None:
        profiler.stats["verification"] = results
    return results
