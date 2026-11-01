import unittest

from core.domain.analytics.behavior import (
    AnomalyType,
    TrajectoryBehaviorAnalyzer,
)


class BehavioralAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = TrajectoryBehaviorAnalyzer(
            max_accel_threshold=5.0,
            loitering_time_seconds=3.0,
            loitering_radius=1.0,
        )

    def test_rapid_acceleration_spike(self):
        track_id = 99
        # Stationary then massive jump
        self.analyzer.update_position(track_id, (0.0, 0.0), timestamp=0.0)
        self.analyzer.update_position(track_id, (0.1, 0.1), timestamp=1.0)
        anomalies = self.analyzer.update_position(track_id, (20.0, 20.0), timestamp=2.0)

        spike = [a for a in anomalies if a.anomaly_type == AnomalyType.RAPID_ACCELERATION]
        self.assertGreater(len(spike), 0)
        self.assertEqual(spike[0].track_id, track_id)

    def test_loitering_detection(self):
        track_id = 101
        # Object remains in close radius for 4.0 seconds
        self.analyzer.update_position(track_id, (5.0, 5.0), timestamp=0.0)
        self.analyzer.update_position(track_id, (5.2, 5.1), timestamp=1.5)
        anomalies = self.analyzer.update_position(track_id, (5.1, 5.2), timestamp=4.0)

        loitering = [a for a in anomalies if a.anomaly_type == AnomalyType.LOITERING]
        self.assertGreater(len(loitering), 0)
        self.assertIn("loitering", loitering[0].description)


if __name__ == "__main__":
    unittest.main()
