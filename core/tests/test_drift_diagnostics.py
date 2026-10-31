import unittest

from core.application.learning.drift_diagnostics import (
    CameraConditionDiagnoser,
    DriftCondition,
)


class DriftDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Reference embeddings with normal variance and energy
        self.ref_embeddings = [
            [1.0, 0.5, -0.5, 0.2],
            [0.8, 0.6, -0.4, 0.1],
            [1.1, 0.4, -0.6, 0.3],
            [0.9, 0.5, -0.5, 0.2],
        ]
        self.diagnoser = CameraConditionDiagnoser(self.ref_embeddings)

    def test_nominal_distribution_diagnosed(self) -> None:
        report = self.diagnoser.diagnose(self.ref_embeddings, average_detection_confidence=0.9)
        self.assertEqual(report.condition, DriftCondition.NOMINAL)
        self.assertEqual(report.recommended_action, "continue_monitoring")

    def test_severe_blur_diagnosed(self) -> None:
        # Severe blur: variance collapses to almost zero, confidence drops
        blurred_embeddings = [
            [0.01, 0.01, 0.01, 0.01],
            [0.011, 0.011, 0.011, 0.011],
            [0.01, 0.01, 0.01, 0.01],
        ]
        report = self.diagnoser.diagnose(blurred_embeddings, average_detection_confidence=0.2)
        self.assertEqual(report.condition, DriftCondition.SEVERE_BLUR)
        self.assertIn("clean_lens", report.recommended_action)

    def test_illumination_shift_diagnosed(self) -> None:
        # Extreme energy increase (overexposed scene)
        overexposed_embeddings = [
            [5.0, 5.0, 5.0, 5.0],
            [5.2, 4.8, 5.1, 4.9],
        ]
        report = self.diagnoser.diagnose(overexposed_embeddings, average_detection_confidence=0.8)
        self.assertEqual(report.condition, DriftCondition.ILLUMINATION_SHIFT)
        self.assertIn("auto_exposure", report.recommended_action)


if __name__ == "__main__":
    unittest.main()
