from __future__ import annotations

import numpy as np


class Int8UniformQuantizer:
    """Quantization engine converting FP32 tensors to INT8 representations."""

    @staticmethod
    def calibrate_symmetric(weights: np.ndarray) -> tuple[float, np.ndarray]:
        """Symmetric quantization mapping [-max_abs, max_abs] to [-127, 127]."""
        max_abs = float(np.max(np.abs(weights)))
        scale = max_abs / 127.0 if max_abs > 1e-8 else 1.0
        q_weights = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
        return scale, q_weights

    @staticmethod
    def dequantize_symmetric(scale: float, q_weights: np.ndarray) -> np.ndarray:
        return q_weights.astype(np.float32) * scale
