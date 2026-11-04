import unittest

import numpy as np

from core.domain.spatial.reasoning import (
    BoundingBox2D,
    BoundingBox3D,
    Circle2D,
    MetricReasoner,
    Point2D,
    Point3D,
    Polygon2D,
    SpatialIndex,
    SpatialReasoner,
    SpatialRelationType,
    TopologicalReasoner,
    create_metric_reasoner,
    create_spatial_planner,
    create_spatial_reasoner,
    create_topological_reasoner,
)


class Point2DTests(unittest.TestCase):
    def test_distance_to(self):
        p1 = Point2D(0.0, 0.0)
        p2 = Point2D(3.0, 4.0)
        self.assertAlmostEqual(p1.distance_to(p2), 5.0)

    def test_to_array(self):
        p = Point2D(1.0, 2.0)
        self.assertEqual(list(p.to_array()), [1.0, 2.0])


class Point3DTests(unittest.TestCase):
    def test_distance_to(self):
        p1 = Point3D(0.0, 0.0, 0.0)
        p2 = Point3D(1.0, 2.0, 2.0)
        self.assertAlmostEqual(p1.distance_to(p2), 3.0)


class BoundingBox2DTests(unittest.TestCase):
    def test_properties(self):
        bbox = BoundingBox2D(0.0, 0.0, 10.0, 5.0)
        self.assertAlmostEqual(bbox.width, 10.0)
        self.assertAlmostEqual(bbox.height, 5.0)
        self.assertEqual(bbox.center, Point2D(5.0, 2.5))
        self.assertAlmostEqual(bbox.area, 50.0)

    def test_contains(self):
        bbox = BoundingBox2D(0.0, 0.0, 10.0, 10.0)
        self.assertTrue(bbox.contains(Point2D(5.0, 5.0)))
        self.assertFalse(bbox.contains(Point2D(15.0, 5.0)))

    def test_intersects(self):
        b1 = BoundingBox2D(0.0, 0.0, 5.0, 5.0)
        b2 = BoundingBox2D(4.0, 4.0, 9.0, 9.0)
        self.assertTrue(b1.intersects(b2))
        b3 = BoundingBox2D(6.0, 6.0, 9.0, 9.0)
        self.assertFalse(b1.intersects(b3))

    def test_iou(self):
        b1 = BoundingBox2D(0.0, 0.0, 10.0, 10.0)
        b2 = BoundingBox2D(5.0, 5.0, 15.0, 15.0)
        # intersection: 5x5 = 25, union: 100+100-25 = 175, IoU = 25/175 = 1/7
        self.assertAlmostEqual(b1.iou(b2), 1.0 / 7.0, places=4)


class SpatialIndexTests(unittest.TestCase):
    def test_insert_and_query_radius(self):
        index = SpatialIndex(2)
        index.insert("a", np.array([0.0, 0.0]))
        index.insert("b", np.array([10.0, 0.0]))
        results = index.query_radius(np.array([0.0, 0.0]), 5.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "a")

    def test_query_knn(self):
        import numpy as np

        index = SpatialIndex(2)
        index.insert("a", np.array([0.0, 0.0]))
        index.insert("b", np.array([10.0, 0.0]))
        index.insert("c", np.array([1.0, 1.0]))
        results = index.query_knn(np.array([0.0, 0.0]), k=2)
        self.assertEqual(len(results), 2)


class SpatialReasonerTests(unittest.TestCase):
    def test_add_and_compute_relations(self):
        reasoner = SpatialReasoner()
        reasoner.add_object("a", Point2D(0.0, 0.0))
        reasoner.add_object("b", Point2D(10.0, 0.0))
        relations = reasoner.compute_relations("a", radius=20.0)
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].subject_id, "a")
        self.assertEqual(relations[0].object_id, "b")


class TopologicalReasonerTests(unittest.TestCase):
    def test_find_path(self):
        reasoner = TopologicalReasoner()
        reasoner.add_region("A", Polygon2D([Point2D(0, 0), Point2D(10, 0), Point2D(10, 10), Point2D(0, 10)]))
        reasoner.add_region("B", Polygon2D([Point2D(10, 0), Point2D(20, 0), Point2D(20, 10), Point2D(10, 10)]))
        path = reasoner.find_path("A", "B")
        self.assertEqual(path, ["A", "B"])

    def test_find_path_no_route(self):
        reasoner = TopologicalReasoner()
        reasoner.add_region("A", Polygon2D([Point2D(0, 0), Point2D(10, 0), Point2D(10, 10), Point2D(0, 10)]))
        path = reasoner.find_path("A", "Z")
        self.assertIsNone(path)


class MetricReasonerTests(unittest.TestCase):
    def test_estimate_position(self):
        reasoner = MetricReasoner()
        reasoner.add_landmark("L1", Point2D(0.0, 0.0))
        reasoner.add_landmark("L2", Point2D(10.0, 0.0))
        reasoner.add_landmark("L3", Point2D(0.0, 10.0))
        pos = reasoner.estimate_position({"L1": 5.0, "L2": 5.0, "L3": 5.0})
        self.assertIsNotNone(pos)


class FactoryTests(unittest.TestCase):
    def test_create_spatial_reasoner(self):
        self.assertIsInstance(create_spatial_reasoner(), SpatialReasoner)

    def test_create_topological_reasoner(self):
        self.assertIsInstance(create_topological_reasoner(), TopologicalReasoner)

    def test_create_metric_reasoner(self):
        self.assertIsInstance(create_metric_reasoner(), MetricReasoner)

    def test_create_spatial_planner(self):
        bounds = BoundingBox2D(0.0, 0.0, 100.0, 100.0)
        planner = create_spatial_planner(bounds, resolution=1.0)
        self.assertIsNotNone(planner)


if __name__ == "__main__":
    unittest.main()
