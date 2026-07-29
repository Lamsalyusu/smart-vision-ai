import unittest

from core.application.spatial import HazardZone, SpatialEntity, TemporalSceneGraph


def entity(track_id: int, x: float, z: float, bbox: tuple[float, float, float, float]) -> SpatialEntity:
    return SpatialEntity(track_id, "person", (x, 0, z), bbox)


class TemporalSceneGraphTests(unittest.TestCase):
    def test_detects_approaching_entities_over_time(self) -> None:
        graph = TemporalSceneGraph(proximity_threshold_m=10)
        graph.update(0, [entity(1, 0, 0, (0, 0, 10, 10)), entity(2, 4, 0, (20, 0, 30, 10))])
        snapshot = graph.update(
            1,
            [entity(1, 0, 0, (0, 0, 10, 10)), entity(2, 2, 0, (20, 0, 30, 10))],
        )

        approaching = [edge for edge in snapshot.edges if edge.relation == "approaching"]
        self.assertEqual(len(approaching), 1)
        self.assertEqual(approaching[0].source_id, "1")

    def test_identifies_nearer_object_as_occluder(self) -> None:
        graph = TemporalSceneGraph(occlusion_overlap_threshold=0.5)
        snapshot = graph.update(
            0,
            [entity(1, 0, 1, (0, 0, 10, 10)), entity(2, 0, 3, (2, 2, 8, 8))],
        )

        edge = next(edge for edge in snapshot.edges if edge.relation == "occluding")
        self.assertEqual((edge.source_id, edge.target_id), ("1", "2"))

    def test_detects_entity_inside_ground_plane_hazard_zone(self) -> None:
        graph = TemporalSceneGraph()
        zone = HazardZone("restricted", ((-1, -1), (1, -1), (1, 1), (-1, 1)))
        snapshot = graph.update(0, [entity(7, 0.5, 0.5, (0, 0, 1, 1))], [zone])

        self.assertIn(
            ("7", "restricted", "inside_hazard_zone"),
            [(edge.source_id, edge.target_id, edge.relation) for edge in snapshot.edges],
        )

    def test_query_filters_relation_and_time_range(self) -> None:
        graph = TemporalSceneGraph(proximity_threshold_m=10)
        graph.update(1, [entity(1, 0, 0, (0, 0, 1, 1)), entity(2, 1, 0, (2, 0, 3, 1))])
        graph.update(2, [entity(1, 0, 0, (0, 0, 1, 1)), entity(2, 1, 0, (2, 0, 3, 1))])

        matches = graph.query("near", start_seconds=1.5, end_seconds=2)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], 2)

    def test_history_is_bounded_and_timestamps_must_increase(self) -> None:
        graph = TemporalSceneGraph(history_capacity=2)
        for timestamp in (1, 2, 3):
            graph.update(timestamp, [])

        self.assertEqual([frame.timestamp_seconds for frame in graph.history], [2, 3])
        with self.assertRaisesRegex(ValueError, "increase monotonically"):
            graph.update(3, [])


if __name__ == "__main__":
    unittest.main()