from __future__ import annotations
from typing import Tuple
import torch.nn as nn

def _tokenize(dotted: str):
    parts = []
    for p in dotted.split("."):
        parts.append(int(p) if p.isdigit() else p)
    return parts

def find_parent(root: nn.Module, target) -> Tuple[nn.Module, str]:
    if isinstance(target, str):
        parts = _tokenize(target)
        if not parts:
            raise ValueError("Empty dotted path.")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]
    if isinstance(target, nn.Module):
        stack = [(root, name, mod) for name, mod in root.named_children()]
        while stack:
            parent, name, mod = stack.pop()
            if mod is target:
                return parent, name
            stack.extend((mod, cname, cmod) for cname, cmod in mod.named_children())
        raise ValueError("Target module is not a child of the provided root.")
    raise TypeError("target must be a dotted path (str) or an nn.Module instance.")

def get(root: nn.Module, path: str | int) -> nn.Module:
    if isinstance(path, int):
        if hasattr(root, "__getitem__"):
            return root[path]
        raise KeyError(f"Cannot index into {type(root).__name__} with [{path}]")
    child = getattr(root, path, None)
    if child is not None:
        return child
    for name, mod in root.named_children():
        if name == path:
            return mod
    raise KeyError(f"Child '{path}' not found in {type(root).__name__}")

def set_child(parent: nn.Module, name_or_idx: str | int, child: nn.Module) -> None:
    if isinstance(name_or_idx, int):
        if hasattr(parent, "__setitem__"):
            parent[name_or_idx] = child
            return
        raise KeyError(f"Parent {type(parent).__name__} is not indexable")
    setattr(parent, name_or_idx, child)

def replace_inplace(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parent, leaf = find_parent(root, dotted)
    set_child(parent, leaf, new)

def has_path(root: nn.Module, dotted: str) -> bool:
    try:
        parent, leaf = find_parent(root, dotted)
        _ = get(parent, leaf)
        return True
    except KeyError:
        return False
