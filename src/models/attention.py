"""Attention compatibility for forward AD on Apple MPS."""

import torch
from torch.nn import functional as F


def math_attention(q, k, v, *, dropout_p=0.0):
    scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    probability = scores.softmax(dim=-1)
    if dropout_p:
        probability = F.dropout(probability, p=dropout_p, training=True)
    return probability @ v


def scaled_attention(q, k, v, *, dropout_p=0.0):
    # sdpa_kernel(MATH) still dispatches to a native MPS op without forward AD.
    # Use differentiable primitive ops only for an MPS JVP; ordinary training
    # and sampling keep PyTorch's optimized attention implementation.
    if q.device.type == "mps" and any(
        torch.autograd.forward_ad.unpack_dual(value).tangent is not None
        for value in (q, k, v)
    ):
        return math_attention(q, k, v, dropout_p=dropout_p)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
