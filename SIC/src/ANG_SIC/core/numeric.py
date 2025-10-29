from __future__ import annotations
from typing import Tuple
import torch

def clamp_decimals(d: int) -> int:
    return max(0, min(int(d), 12))

def round_tensor(x: torch.Tensor, decimals: int) -> torch.Tensor:
    if decimals < 0:
        return x
    dtype = x.dtype
    y = x.float()
    factor = 10.0 ** clamp_decimals(decimals)
    y = torch.round(y * factor) / factor
    return y.to(dtype)

def to_2tuple(x) -> Tuple[int, int]:
    if isinstance(x, (tuple, list)):
        return int(x[0]), int(x[1] if len(x) > 1 else x[0])
    v = int(x)
    return v, v

def conv2d_output_shape(h, w, kernel_size, stride, padding, dilation) -> Tuple[int, int]:
    kh, kw = to_2tuple(kernel_size)
    sh, sw = to_2tuple(stride)
    ph, pw = to_2tuple(padding)
    dh, dw = to_2tuple(dilation)
    out_h = (h + 2*ph - dh*(kh - 1) - 1)//sh + 1
    out_w = (w + 2*pw - dw*(kw - 1) - 1)//sw + 1
    return int(out_h), int(out_w)
