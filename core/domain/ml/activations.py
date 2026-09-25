from __future__ import annotations

import math

import numpy as np


class Activations:
    """Non-linear activation functions with numerically stable forward and backward passes."""

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit (GELU) with tanh approximation."""
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        """SiLU / Swish activation: x * sigmoid(x)."""
        return x * Activations.sigmoid(x)

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax subtraction."""
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)
