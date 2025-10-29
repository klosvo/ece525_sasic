from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

def apply_clusters_numpy_jenks(orig_np: np.ndarray, breaks_np: np.ndarray) -> np.ndarray:
    out = orig_np.copy()
    nz_mask = out != 0
    if not nz_mask.any():
        return out
    vals = out[nz_mask]
    k = len(breaks_np) - 1
    means = np.empty(k, dtype=np.float64)
    for i in range(k):
        lo, hi = breaks_np[i], breaks_np[i + 1]
        m = (vals >= lo) & (vals <= hi) if i == 0 else (vals > lo) & (vals <= hi)
        means[i] = float(vals[m].mean()) if m.any() else float((lo + hi) * 0.5)
    for i in range(k):
        lo, hi = breaks_np[i], breaks_np[i + 1]
        m = (out >= lo) & (out <= hi) & nz_mask if i == 0 else (out > lo) & (out <= hi) & nz_mask
        if m.any():
            out[m] = means[i]
    return out

def apply_clusters_torch(row_t: torch.Tensor, breaks_np: np.ndarray) -> torch.Tensor:
    nz_mask = row_t != 0
    if not nz_mask.any():
        return row_t
    vals = row_t[nz_mask]
    breaks = torch.tensor(breaks_np, device=vals.device, dtype=vals.dtype)
    k = int(breaks.numel()) - 1
    means: List[torch.Tensor] = []
    for i in range(k):
        lo, hi = breaks[i], breaks[i + 1]
        m = (vals >= lo) & (vals <= hi) if i == 0 else (vals > lo) & (vals <= hi)
        means.append(vals[m].mean() if m.any() else (lo + hi) / 2)
    means = torch.stack(means) if k > 0 else vals.new_empty(0)
    for i in range(k):
        lo, hi = breaks[i], breaks[i + 1]
        m = (vals >= lo) & (vals <= hi) if i == 0 else (vals > lo) & (vals <= hi)
        if m.any():
            vals[m] = means[i]
    row_t[nz_mask] = vals
    return row_t

def should_use_gpu_for_layer(flat_row_len: int, layer_param_count: int, sic_cfg: dict) -> bool:
    if not bool(sic_cfg.get("hybrid_enable", True)):
        return False
    row_thr = int(sic_cfg.get("row_len_gpu_threshold", sic_cfg.get("gpu_row_len_min", 2048)))
    layer_thr = int(sic_cfg.get("layer_params_gpu_threshold", 1_000_000))
    return (int(flat_row_len) >= row_thr) or (int(layer_param_count) >= layer_thr)

def _move_first_offender(filtered_dataset, y_true: int) -> None:
    if isinstance(filtered_dataset, list):
        n = len(filtered_dataset)
        j = -1
        for i in range(n):
            item = filtered_dataset[i]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                yi = item[1]
                v = int(yi.item()) if torch.is_tensor(yi) else int(yi)
                if v == int(y_true):
                    j = i
                    break
        if j >= 0:
            off = filtered_dataset.pop(j)
            filtered_dataset.insert(0, off)
        return
    if isinstance(filtered_dataset, TensorDataset):
        tensors = list(filtered_dataset.tensors)
        if len(tensors) >= 2 and tensors[0].size(0) == tensors[1].size(0):
            N = tensors[0].size(0)
            idx = -1
            for i in range(N):
                yi = tensors[1][i]
                v = int(yi.item()) if torch.is_tensor(yi) else int(yi)
                if v == int(y_true):
                    idx = i
                    break
            if idx >= 0:
                order = [idx] + [i for i in range(N) if i != idx]
                order_t = torch.tensor(order, dtype=torch.long)
                new_tensors = [t.index_select(0, order_t) for t in tensors]
                filtered_dataset.tensors = tuple(new_tensors)
        return
    if isinstance(filtered_dataset, Subset):
        inds = list(filtered_dataset.indices)
        j = -1
        for p, i in enumerate(inds):
            item = filtered_dataset.dataset[i]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                yi = item[1]
                v = int(yi.item()) if torch.is_tensor(yi) else int(yi)
                if v == int(y_true):
                    j = p
                    break
        if j >= 0:
            i0 = inds.pop(j)
            filtered_dataset.indices = [i0] + inds
        return
    if hasattr(filtered_dataset, "pop") and hasattr(filtered_dataset, "insert"):
        n = len(filtered_dataset)
        j = -1
        for i in range(n):
            item = filtered_dataset[i]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                yi = item[1]
                v = int(yi.item()) if torch.is_tensor(yi) else int(yi)
                if v == int(y_true):
                    j = i
                    break
        if j >= 0:
            off = filtered_dataset.pop(j)
            filtered_dataset.insert(0, off)

@torch.inference_mode()
def all_samples_correct(
    model: nn.Module,
    filtered_dataset,
    device: torch.device,
    eval_max: Optional[int] = None,
    profiler=None,
    move_first_offender: bool = True,
    batch_size: Optional[int] = None,
) -> Tuple[bool, float]:
    t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if t0 is not None:
        t0.record()
    total = len(filtered_dataset)
    if total == 0:
        if profiler is not None:
            profiler._add_time("all_samples_correct", 0.0)
        return False, 0.0
    if eval_max is not None and int(eval_max) > 0:
        total = min(total, int(eval_max))
    bs = max(1, min(int(batch_size or 1024), total))
    loader = DataLoader(
        filtered_dataset,
        batch_size=bs,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        num_workers=0,
    )
    model.eval()
    first_bad = None
    correct = 0
    total_seen = 0
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            if hasattr(yb, "to"):
                yb = yb.to(xb.device, non_blocking=(xb.device.type == "cuda")).long()
            logits = model(xb)
            logits = _normalize_logits_for_eval(logits, yb)
            preds = logits.argmax(dim=1)
            if preds.size(0) != yb.size(0):
                m = min(preds.size(0), yb.size(0))
                preds = preds[:m]
                yb = yb[:m]
            match = preds.eq(yb)
            if (not bool(match.all())) and (first_bad is None) and move_first_offender:
                bad_idx = int((~match).nonzero(as_tuple=False)[0].item())
                first_bad = (
                    xb[bad_idx].detach().cpu(),
                    int(yb[bad_idx].item()),
                    int(preds[bad_idx].item()),
                )
            correct += int(match.sum().item())
            total_seen += int(yb.numel())
    acc = correct / max(1, total_seen)
    if profiler is not None:
        if t1 is not None:
            t1.record()
            torch.cuda.synchronize()
            profiler._add_time("all_samples_correct", t0.elapsed_time(t1) / 1000.0)
        else:
            profiler._add_time("all_samples_correct", 0.0)
    if (first_bad is not None) and move_first_offender:
        _, y_true, _ = first_bad
        _move_first_offender(filtered_dataset, int(y_true))
    return acc == 1.0, acc

def _normalize_logits_for_eval(
    logits: Union[torch.Tensor, tuple, list, dict],
    target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    elif isinstance(logits, dict):
        if "logits" in logits and torch.is_tensor(logits["logits"]):
            logits = logits["logits"]
        else:
            for v in logits.values():
                if torch.is_tensor(v):
                    logits = v
                    break
            else:
                raise ValueError("No tensor-like value found in logits dict.")
    if not torch.is_tensor(logits):
        raise TypeError("Model output must be a torch.Tensor, tuple/list/dict containing a Tensor.")
    if logits.dim() == 1:
        p = torch.sigmoid(logits)
        return torch.stack([1.0 - p, p], dim=1)
    if logits.dim() == 2 and logits.size(1) == 1:
        p = torch.sigmoid(logits.squeeze(1))
        return torch.stack([1.0 - p, p], dim=1)
    if target is not None and torch.is_tensor(target):
        N = int(target.size(0))
        if logits.dim() == 2 and logits.size(0) == N:
            return logits
        if logits.dim() == 4 and logits.size(0) == N:
            return logits.flatten(2).mean(dim=2)
        if logits.dim() == 3 and logits.size(0) == N:
            return logits.mean(dim=1)
        if logits.dim() == 2 and N > 0 and logits.size(0) % N == 0:
            k = logits.size(0) // N
            return logits.view(N, k, -1).mean(dim=1)
        if logits.dim() > 2:
            c = logits.size(-1)
            logits = logits.reshape(-1, c)
        if logits.size(0) != N:
            m = min(logits.size(0), N)
            logits = logits[:m]
        return logits
    if logits.dim() == 2:
        return logits
    if logits.dim() >= 3:
        N, C = logits.size(0), logits.size(-1)
        return logits.reshape(N, -1, C).mean(dim=1)
    return logits

def train(model: nn.Module, train_loader, epochs: int, device: torch.device, lr: float = 1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(int(epochs)):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return model

def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x = batch.get("pixel_values") or batch.get("images") or batch.get("input") or batch.get("x")
                y = batch.get("labels") or batch.get("y")
            else:
                x, y = batch, None
            if x is None:
                continue
            x = x.to(device, non_blocking=(device.type == "cuda")) if hasattr(x, "to") else x
            logits = model(x)
            logits = _normalize_logits_for_eval(logits, y if torch.is_tensor(y) else None)
            preds = logits.argmax(1)
            if y is not None and torch.is_tensor(y):
                y = y.to(logits.device, non_blocking=(logits.device.type == "cuda"))
                if preds.size(0) != y.size(0):
                    m = min(preds.size(0), y.size(0))
                    preds = preds[:m]
                    y = y[:m]
                correct += int(preds.eq(y).sum().item())
                total += int(y.size(0))
    return float(100.0 * correct / max(1, total))
