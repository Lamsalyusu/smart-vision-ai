from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

from .tensor import Tensor


class Module:
    """Base neural network module."""

    def parameters(self) -> List[Tensor]:
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    """Fully-connected affine layer: y = x @ W^T + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        # Kaiming / He uniform initialization
        bound = 1.0 / math.sqrt(in_features)
        w_data = np.random.uniform(-bound, bound, (in_features, out_features))
        self.weight = Tensor(w_data, requires_grad=True)

        if bias:
            b_data = np.random.uniform(-bound, bound, (out_features,))
            self.bias: Optional[Tensor] = Tensor(b_data, requires_grad=True)
        else:
            self.bias = None

    def __call__(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(Module):
    """Layer Normalization across last dimension."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = Tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        norm_data = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = norm_data * self.gamma.data + self.beta.data
        return Tensor(out_data, requires_grad=x.requires_grad)


class Dropout(Module):
    """Inverted dropout regularizer."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.training = True

    def __call__(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0.0:
            return x
        mask = (np.random.rand(*x.shape) >= self.p).astype(np.float32) / (1.0 - self.p)
        return Tensor(x.data * mask, requires_grad=x.requires_grad)
