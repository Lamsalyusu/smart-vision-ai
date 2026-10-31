from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np


class DriftCondition(str, Enum):
    NOMINAL = "nominal"
    SEVERE_BLUR = "severe_blur"
    ILLUMINATION_SHIFT = "illumination_shift"
    OCCLUSION_BURST = "occlusion_burst"
    ANOMALOUS_DISTRIBUTION = "anomalous_distribution"


@dataclass(frozen=True, slots=True)
class ConditionDiagnosisReport:
    condition: DriftCondition
    confidence: float
    recommended_action: str
    details: dict[str, float]


class CameraConditionDiagnoser:
    """Diagnoses physical or environmental causes behind detected embedding drift."""

    def __init__(
        self,
        reference_embeddings: Sequence[Sequence[float]],
    ) -> None:
        ref = np.asarray(reference_embeddings, dtype=np.float64)
        if ref.ndim != 2 or ref.shape[0] < 2:
            raise ValueError("reference embeddings must have at least 2 samples")
        self._ref_mean = np.mean(ref, axis=0)
        self._ref_std = np.std(ref, axis=0) + 1e-8
        self._ref_energy = float(np.mean(np.sum(ref**2, axis=1)))

    def diagnose(
        self,
        current_embeddings: Sequence[Sequence[float]],
        average_detection_confidence: float = 1.0,
    ) -> ConditionDiagnosisReport:
        curr = np.asarray(current_embeddings, dtype=np.float64)
        if curr.ndim != 2 or curr.shape[0] == 0:
            return ConditionDiagnosisReport(
                condition=DriftCondition.NOMINAL,
                confidence=1.0,
                recommended_action="none",
                details={},
            )

        curr_mean = np.mean(curr, axis=0)
        curr_std = np.std(curr, axis=0)
        curr_energy = float(np.mean(np.sum(curr**2, axis=1)))

        # 1. Energy drop indicates low contrast / underexposure / severe blur
        energy_ratio = curr_energy / max(1e-6, self._ref_energy)

        # 2. Variance collapse indicates uniform/blank/blurred frames
        mean_std_ratio = float(np.mean(curr_std) / np.mean(self._ref_std))

        # 3. Mean shift distance
        z_shift = float(np.mean(np.abs(curr_mean - self._ref_mean) / self._ref_std))

        details = {
            "energy_ratio": energy_ratio,
            "variance_ratio": mean_std_ratio,
            "normalized_z_shift": z_shift,
            "mean_confidence": average_detection_confidence,
        }

        # Classification logic based on empirical distribution characteristics
        if mean_std_ratio < 0.4 and average_detection_confidence < 0.5:
            return ConditionDiagnosisReport(
                condition=DriftCondition.SEVERE_BLUR,
                confidence=min(1.0, 1.0 - mean_std_ratio),
                recommended_action="inspect_and_clean_lens_or_refocus",
                details=details,
            )

        if energy_ratio < 0.5 or energy_ratio > 1.8:
            return ConditionDiagnosisReport(
                condition=DriftCondition.ILLUMINATION_SHIFT,
                confidence=min(1.0, abs(1.0 - energy_ratio)),
                recommended_action="recalibrate_auto_exposure_and_white_balance",
                details=details,
            )

        if z_shift > 2.5 and average_detection_confidence < 0.6:
            return ConditionDiagnosisReport(
                condition=DriftCondition.OCCLUSION_BURST,
                confidence=min(1.0, z_shift / 5.0),
                recommended_action="check_camera_mount_or_persistent_obstacle",
                details=details,
            )

        if z_shift > 1.5:
            return ConditionDiagnosisReport(
                condition=DriftCondition.ANOMALOUS_DISTRIBUTION,
                confidence=0.8,
                recommended_action="trigger_active_learning_annotation_batch",
                details=details,
            )

        return ConditionDiagnosisReport(
            condition=DriftCondition.NOMINAL,
            confidence=0.95,
            recommended_action="continue_monitoring",
            details=details,
        )
