from __future__ import annotations
from typing import Optional
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from src.ANG_SIC.SIC.sic_utils import _normalize_logits_for_eval

@torch.inference_mode()
def build_acceptance_subset_from_loaders(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    val_loader: Optional[DataLoader] = None,
    from_spec: str = "train+val",
    max_examples: Optional[int] = None,
) -> TensorDataset:
    model.eval()
    xs, ys = [], []
    nb = device.type == "cuda"

    def _harvest(loader: DataLoader) -> bool:
        nonlocal xs, ys
        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                xb, yb = batch[0], batch[1]
            else:
                raise ValueError("Loader must yield (x,y) tuples")
            xb = xb.to(device, non_blocking=nb)
            yb = yb.to(device, non_blocking=nb)
            logits = _normalize_logits_for_eval(model(xb), yb)
            pred = logits.argmax(1)
            m = min(pred.size(0), yb.size(0))
            pred, yb, xb = pred[:m], yb[:m], xb[:m]
            msk = pred.eq(yb)
            if msk.any():
                xs.append(xb[msk].detach().cpu())
                ys.append(yb[msk].detach().cpu())
                if max_examples is not None and sum(t.size(0) for t in xs) >= int(max_examples):
                    return True
        return False

    spec = (from_spec or "").lower()
    if "train" in spec and train_loader is not None:
        _harvest(train_loader)
    if "val" in spec and val_loader is not None and (max_examples is None or sum(t.size(0) for t in xs) < int(max_examples)):
        _harvest(val_loader)

    if not xs:
        return TensorDataset(torch.empty(0), torch.empty(0, dtype=torch.long))

    X = torch.cat(xs, 0)
    Y = torch.cat(ys, 0)
    if max_examples and X.size(0) > int(max_examples):
        X, Y = X[: int(max_examples)], Y[: int(max_examples)]
    return TensorDataset(X, Y)
