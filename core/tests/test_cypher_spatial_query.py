import unittest

from core.application.spatial.cypher_query import SpatialGraphQueryEngine
from core.application.spatial.temporal_scene_graph import (
    SpatialEntity,
    TemporalSceneSnapshot,
    TemporalSpatialEdge,
)


class CypherSpatialQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        nodes = (
            SpatialEntity(1, "person", (0.0, 0.0, 2.0), (10, 10, 50, 100)),
            SpatialEntity(2, "can", (0.5, 0.0, 2.0), (60, 60, 90, 90)),
            SpatialEntity(3, "chair", (2.0, 0.0, 5.0), (100, 100, 200, 200)),
        )
        edges = (
            TemporalSpatialEdge(
                source_id="1", target_id="2", relation="approaching", distance_m=0.5
            ),
            TemporalSpatialEdge(
                source_id="1", target_id="3", relation="near", distance_m=3.5
            ),
        )
        self.snapshot = TemporalSceneSnapshot(
            timestamp_seconds=1.0, nodes=nodes, edges=edges
        )

    def test_query_relation_and_class_filters(self) -> None:
        query = "MATCH (p:person)-[r:approaching]->(c:can)"
        matches = SpatialGraphQueryEngine.execute(query, self.snapshot)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].source_id, "1")
        self.assertEqual(matches[0].target_id, "2")
        self.assertEqual(matches[0].relation, "approaching")

    def test_query_distance_predicate_filter(self) -> None:
        # Distance < 1.0 should only match the can, not the chair
        query = "MATCH (p:person)-[r]->(obj) WHERE r.distance < 1.0"
        matches = SpatialGraphQueryEngine.execute(query, self.snapshot)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].target_id, "2")

    def test_query_distance_lower_bound(self) -> None:
        query = "MATCH (p)-[r]->(obj) WHERE r.distance > 2.0"
        matches = SpatialGraphQueryEngine.execute(query, self.snapshot)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].target_id, "3")


if __name__ == "__main__":
    unittest.main()
