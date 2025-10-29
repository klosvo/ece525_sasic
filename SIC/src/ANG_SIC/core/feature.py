from __future__ import annotations
from contextlib import contextmanager
from typing import Callable, Tuple, Optional, Iterable, Iterator
import torch
import torch.nn as nn


def model_device(m: nn.Module) -> torch.device:
    for p in m.parameters(recurse=True):
        return p.device
    for b in m.buffers(recurse=True):
        return b.device
    return torch.device("cpu")

@torch.inference_mode()
def feature_dim(
    model: nn.Module,
    layer: str,
    sample_shape: Tuple[int, ...] | None = None,
    *,
    device: Optional[torch.device] = None,
    sample_tensor: Optional[torch.Tensor] = None,
) -> int:
    dev = device or model_device(model)
    modules = dict(model.named_modules())
    if layer not in modules:
        raise ValueError(f"Layer '{layer}' not found in model. Available: {list(modules.keys())[:20]} ...")
    feats: dict[str, int] = {}

    def _hook(_m, _i, out):
        flat = out.reshape(out.shape[0], -1)
        feats["n"] = int(flat.shape[1])

    handle = modules[layer].register_forward_hook(_hook)
    model.eval()

    if sample_tensor is not None:
        x = sample_tensor
        if not isinstance(x, torch.Tensor):
            raise TypeError("sample_tensor must be a torch.Tensor")
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(dev, non_blocking=(dev.type == "cuda"))
        _ = model(x)
    else:
        if sample_shape is None:
            raise ValueError("Provide either sample_tensor or sample_shape.")
        x = torch.randn((1, *sample_shape), device=dev)
        _ = model(x)

    handle.remove()
    if "n" not in feats:
        raise RuntimeError(f"Could not capture features at layer '{layer}'")
    return feats["n"]

@torch.inference_mode()
def feature_fn(
    model: nn.Module,
    layer: str,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], torch.utils.hooks.RemovableHandle]:
    modules = dict(model.named_modules())
    if layer not in modules:
        raise ValueError(f"Layer '{layer}' not found in model. Available: {list(modules.keys())[:20]} ...")
    feats: dict[str, torch.Tensor] = {}

    def _hook(_m, _i, out):
        feats["out"] = out.detach().reshape(out.shape[0], -1)

    handle = modules[layer].register_forward_hook(_hook)

    def fn(x: torch.Tensor) -> torch.Tensor:
        model.eval()
        dev = model_device(model)
        with torch.no_grad():
            xb = x
            if xb.dim() == 3:
                xb = xb.unsqueeze(0)
            xb = xb.to(dev, non_blocking=(dev.type == "cuda"))
            _ = model(xb)
        out = feats.get("out")
        if out is None or out.shape[0] != xb.shape[0]:
            raise RuntimeError("Feature hook mismatch (batch size/layer).")
        return out

    return fn, handle

@contextmanager
def with_feature_fn(model: nn.Module, layer: str) -> Iterator[Callable[[torch.Tensor], torch.Tensor]]:
    fn, handle = feature_fn(model, layer)
    try:
        yield fn
    finally:
        try:
            handle.remove()
        except Exception:
            pass

@contextmanager
def with_forward_hooks(modules: Iterable[nn.Module], fn: Callable[[nn.Module, tuple, torch.Tensor], None]) -> Iterator[None]:
    handles = []
    try:
        for m in modules:
            handles.append(m.register_forward_hook(fn))
        yield
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
