from __future__ import annotations
import os
import hashlib
import random
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.ANG_SIC.SIC.pos_layers import replace_linear_and_conv_with_sic
from src.ANG_SIC.SIC.fast_pos import accelerate_pos_layers
from src.ANG_SIC.core.config import _load_file

def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)

def banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80 + "\n", flush=True)

def ensure_dirs(paths) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

def load_yaml(path: str) -> Dict[str, Any]:
    cfg = _load_file(path)
    log("CONFIG", f"Loaded: {path}")
    return cfg

def validate_cfg(cfg: Dict[str, Any]) -> None:
    for section in ("io", "ang", "sic", "data"):
        if section not in cfg or cfg[section] is None:
            raise KeyError(f"Config missing required section: '{section}'")
    cfg.setdefault("train", {})
    cfg.setdefault("serve", {})

def seed_everything(seed: int, deterministic: bool = True) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = bool(deterministic)
    if deterministic and os.name != "nt":
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

def device_from_auto(pref: Optional[str]) -> torch.device:
    req = (pref or "auto").lower()
    if req == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer_input_spec(model: torch.nn.Module, dataset_name: str) -> Tuple[str, Tuple[int, int, int]]:
    ds = (dataset_name or "").strip().lower()
    if "cifar" in ds:
        H, W = 32, 32
    elif "mnist" in ds:
        H, W = 28, 28
    else:
        H, W = 32, 32
    Ci = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            Ci = int(getattr(m, "in_channels", 0) or 0)
            break
    if Ci is None:
        Ci = 1 if "mnist" in ds else 3
    return "conv", (Ci, H, W)

def first_batch_x(loader: Optional[DataLoader], device: torch.device) -> Optional[torch.Tensor]:
    if loader is None:
        return None
    it = iter(loader)
    try:
        batch = next(it)
    except StopIteration:
        return None
    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        x = batch[0]
    elif isinstance(batch, dict):
        x = batch.get("pixel_values") or batch.get("images") or batch.get("input") or batch.get("x")
    else:
        x = batch
    if x is None:
        return None
    if hasattr(x, "to"):
        x = x.to(device)
    if hasattr(x, "size") and callable(x.size) and x.size(0) and x.size(0) > 1:
        x = x[:1]
    return x

def fingerprint(m: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for t in m.state_dict().values():
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def maybe_materialize_pos(model: torch.nn.Module, sic_cfg: dict) -> torch.nn.Module:
    if not bool(sic_cfg.get("enable_product_of_sums", False)):
        return model

    def _is_pos_layer(m: torch.nn.Module) -> bool:
        n = m.__class__.__name__
        return n in ("SICLinear", "SICConv2d")

    already_pos = any(_is_pos_layer(m) for m in model.modules())
    if not already_pos:
        rd = int(sic_cfg.get("rounding_decimals", 6))
        replace_linear_and_conv_with_sic(model, rounding_decimals=rd)
        setattr(model, "_pos_materialized", True)

    if bool(sic_cfg.get("fast_pos_accel", False)) and not getattr(model, "_pos_accelerated", False):
        use_compile = bool(sic_cfg.get("fast_pos_compile", False))
        try:
            out = accelerate_pos_layers(model, use_torch_compile=use_compile)
            if isinstance(out, tuple) and len(out) == 2:
                model, _ = out
            else:
                model = out
        except TypeError:
            model = accelerate_pos_layers(model)
        finally:
            setattr(model, "_pos_accelerated", True)

    return model
