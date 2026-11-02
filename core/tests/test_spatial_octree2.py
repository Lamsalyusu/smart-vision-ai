import unittest

from core.domain.spatial.octree import (
    OctreeNode,
    OctreePoint,
    SpatialAABB,
    SpatialOctreeIndex,
)


class SpatialAABBTests(unittest.TestCase):
    def test_contains(self):
        aabb = SpatialAABB((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        self.assertTrue(aabb.contains((5.0, 5.0, 5.0)))
        self.assertFalse(aabb.contains((11.0, 5.0, 5.0)))
        self.assertFalse(aabb.contains((-1.0, 5.0, 5.0)))

    def test_intersects(self):
        aabb1 = SpatialAABB((0.0, 0.0, 0.0), (5.0, 5.0, 5.0))
        aabb2 = SpatialAABB((4.0, 4.0, 4.0), (9.0, 9.0, 9.0))
        self.assertTrue(aabb1.intersects(aabb2))

        aabb3 = SpatialAABB((6.0, 6.0, 6.0), (9.0, 9.0, 9.0))
        self.assertFalse(aabb1.intersects(aabb3))


class OctreeNodeTests(unittest.TestCase):
    def test_insert_and_count(self):
        boundary = SpatialAABB((0.0, 0.0, 0.0), (100.0, 100.0, 100.0))
        node = OctreeNode(boundary, capacity=2, max_depth=4)
        self.assertTrue(node.insert(OctreePoint((10.0, 10.0, 10.0), 1)))
        self.assertTrue(node.insert(OctreePoint((20.0, 20.0, 20.0), 2)))
        # Third point triggers subdivision
        self.assertTrue(node.insert(OctreePoint((30.0, 30.0, 30.0), 3)))

    def test_insert_outside_boundary_returns_false(self):
        boundary = SpatialAABB((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        node = OctreeNode(boundary, capacity=8)
        self.assertFalse(node.insert(OctreePoint((100.0, 100.0, 100.0), 1)))

    def test_query_range(self):
        boundary = SpatialAABB((0.0, 0.0, 0.0), (100.0, 100.0, 100.0))
        node = OctreeNode(boundary, capacity=4, max_depth=4)
        node.insert(OctreePoint((10.0, 10.0, 10.0), 1))
        node.insert(OctreePoint((20.0, 20.0, 20.0), 2))
        node.insert(OctreePoint((80.0, 80.0, 80.0), 3))

        results = []
        query = SpatialAABB((0.0, 0.0, 0.0), (30.0, 30.0, 30.0))
        node.query_range(query, results)
        self.assertEqual(len(results), 2)

    def test_query_radius(self):
        boundary = SpatialAABB((0.0, 0.0, 0.0), (100.0, 100.0, 100.0))
        node = OctreeNode(boundary, capacity=4, max_depth=4)
        node.insert(OctreePoint((10.0, 10.0, 10.0), 1))
        node.insert(OctreePoint((50.0, 50.0, 50.0), 2))

        results = []
        node.query_radius((10.0, 10.0, 10.0), 5.0, results)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].data, 1)


class SpatialOctreeIndexTests(unittest.TestCase):
    def test_insert_and_radius_search(self):
        boundary = SpatialAABB((0.0, 0.0, 0.0), (100.0, 100.0, 100.0))
        index = SpatialOctreeIndex(boundary, capacity=4, max_depth=4)
        index.insert(10.0, 10.0, 10.0, 1)
        index.insert(20.0, 20.0, 20.0, 2)
        index.insert(80.0, 80.0, 80.0, 3)

        results = index.radius_search((10.0, 10.0, 10.0), 5.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 1)

        results = index.radius_search((10.0, 10.0, 10.0), 20.0)
        self.assertEqual(len(results), 2)

    def test_insert_outside_bounds_returns_false(self):
        boundary = SpatialAABB((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        index = SpatialOctreeIndex(boundary, capacity=4)
        self.assertFalse(index.insert(100.0, 100.0, 100.0, 1))
        self.assertEqual(index.count, 0)


if __name__ == "__main__":
    unittest.main()
