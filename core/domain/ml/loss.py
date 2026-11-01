from __future__ import annotations

import numpy as np
from .tensor import Tensor


class LossFunctions:
    """Loss functions for perception model training and feature alignment."""

    @staticmethod
    def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred.data - y_true.data
        loss_val = np.mean(diff ** 2)
        return Tensor(loss_val, requires_grad=True)

    @staticmethod
    def cross_entropy_with_logits(logits: np.ndarray, targets: np.ndarray) -> float:
        """Numerically stable multi-class cross-entropy loss."""
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp
        # Target indices
        n = len(logits)
        loss = -np.sum(log_probs[np.arange(n), targets]) / n
        return float(loss)

    @staticmethod
    def huber_loss(y_pred: np.ndarray, y_true: np.ndarray, delta: float = 1.0) -> float:
        """Smooth L1 / Huber loss robust to bounding box outliers."""
        error = np.abs(y_pred - y_true)
        quadratic = np.minimum(error, delta)
        linear = error - quadratic
        loss = 0.5 * (quadratic ** 2) + delta * linear
        return float(np.mean(loss))
