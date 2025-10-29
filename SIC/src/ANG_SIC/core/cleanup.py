from typing import Set, List, Optional
import torch
from torch import nn
from src.ANG_SIC.core.module_path import set_child, find_parent
from src.ANG_SIC.core.feature import with_forward_hooks

def collect_executed_module_names(model: nn.Module, sample: torch.Tensor) -> Set[str]:
    executed: Set[str] = set()
    name_map = {id(m): n for n, m in model.named_modules()}

    def _hook(mod, *_):
        n = name_map.get(id(mod))
        if n:
            executed.add(n)

    model.eval()
    with torch.inference_mode():
        with with_forward_hooks(model.modules(), _hook):
            model(sample)
    return executed

def _module_weights_all_zero(mod: nn.Module) -> bool:
    tensors = []
    w = getattr(mod, "weight", None)
    b = getattr(mod, "bias", None)
    if isinstance(w, torch.Tensor):
        tensors.append(w.detach())
    if isinstance(b, torch.Tensor):
        tensors.append(b.detach())
    return bool(tensors) and all(torch.count_nonzero(t).item() == 0 for t in tensors)

def replace_with_identity(root: nn.Module, dotted: str) -> None:
    parent, name = find_parent(root, dotted)
    set_child(parent, name, nn.Identity())

def strip_dead_zero_modules(model: nn.Module, executed: Set[str], targets: Optional[List[str]] = None) -> List[str]:
    removed: List[str] = []
    mods = dict(model.named_modules())
    for name in list(mods.keys()):
        if not name:
            continue
        if targets and name not in targets:
            continue
        if name in executed:
            continue
        mod = mods[name]
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and _module_weights_all_zero(mod):
            replace_with_identity(model, name)
            removed.append(name)
    return removed
