import unittest

from core.domain.analytics.geofence import (
    BreachSeverity,
    GeofenceZone,
    TripwireDetector,
    ZoneType,
)


class GeofenceAnalyticsTests(unittest.TestCase):
    def test_polygon_ray_casting_containment(self):
        # Square polygon (0,0) to (10,10)
        vertices = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        zone = GeofenceZone("zone_1", "Perimeter Alpha", vertices, zone_type=ZoneType.RESTRICTED, dwell_time_threshold_seconds=2.0)

        self.assertTrue(zone.contains_point((5.0, 5.0)))
        self.assertFalse(zone.contains_point((15.0, 5.0)))
        self.assertFalse(zone.contains_point((-1.0, 5.0)))

    def test_dwell_time_breach_detection(self):
        vertices = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        zone = GeofenceZone("zone_1", "Restricted Vault", vertices, zone_type=ZoneType.RESTRICTED, dwell_time_threshold_seconds=3.0)

        # First entry: no breach yet
        e1 = zone.evaluate_track(track_id=42, point=(5.0, 5.0), now=100.0)
        self.assertIsNone(e1)

        # Dwell 2.0s: still under threshold
        e2 = zone.evaluate_track(track_id=42, point=(5.0, 5.0), now=102.0)
        self.assertIsNone(e2)

        # Dwell 3.5s: breaches threshold!
        e3 = zone.evaluate_track(track_id=42, point=(5.0, 5.0), now=103.5)
        self.assertIsNotNone(e3)
        self.assertEqual(e3.track_id, 42)
        self.assertIn("breached", e3.message)

    def test_tripwire_crossing(self):
        # Tripwire along Y axis from (5, 0) to (5, 10)
        tripwire = TripwireDetector("tw_1", (5.0, 0.0), (5.0, 10.0))

        # Trajectory crossing from X=2 to X=8
        self.assertTrue(tripwire.has_crossed((2.0, 5.0), (8.0, 5.0)))

        # Trajectory staying on the same side
        self.assertFalse(tripwire.has_crossed((2.0, 5.0), (4.0, 5.0)))


if __name__ == "__main__":
    unittest.main()
