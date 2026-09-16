from __future__ import annotations

import math
from typing import List
import numpy as np

from .tensor import Tensor


class AdamW:
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.params = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad

            # Decoupled weight decay
            p.data -= self.lr * self.weight_decay * p.data

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1.0 - (self.beta1 ** self.t))
            v_hat = self.v[i] / (1.0 - (self.beta2 ** self.t))

            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
