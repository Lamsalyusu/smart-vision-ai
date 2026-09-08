from __future__ import annotations

import math
from typing import Tuple
import numpy as np


class ManipulatorJacobianAnalyzer:
    """Manipulability metrics and kinematic singularity analysis."""

    @staticmethod
    def yoshikawa_manipulability(jacobian: np.ndarray) -> float:
        """Yoshikawa manipulability index: w = sqrt(det(J * J^T))."""
        m = jacobian @ jacobian.T
        det_m = np.linalg.det(m)
        return float(math.sqrt(max(0.0, det_m)))

    @staticmethod
    def manipulability_ellipsoid(jacobian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Principal axes and lengths of the velocity manipulability ellipsoid via SVD."""
        u, s, _ = np.linalg.svd(jacobian)
        # Principal axes are columns of U, semi-axis lengths are singular values S
        return u, s

    @staticmethod
    def is_near_singularity(jacobian: np.ndarray, threshold: float = 1e-3) -> bool:
        """Singularity detection via reciprocal condition number."""
        _, s, _ = np.linalg.svd(jacobian)
        if s[0] < 1e-9:
            return True
        r_cond = s[-1] / s[0]
        return r_cond < threshold
