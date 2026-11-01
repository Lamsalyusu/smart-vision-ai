import unittest

from core.domain.spatial.octree import SpatialAABB, SpatialOctreeIndex


class SpatialOctreeTests(unittest.TestCase):
    def test_octree_insertion_and_radius_query(self):
        bounds = SpatialAABB((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0))
        octree = SpatialOctreeIndex(bounds, capacity=4, max_depth=5)

        # Insert points
        octree.insert(0.0, 0.0, 1.0, entity_id=101)
        octree.insert(0.0, 0.0, 2.0, entity_id=102)
        octree.insert(5.0, 5.0, 5.0, entity_id=201)
        octree.insert(-8.0, -8.0, -8.0, entity_id=301)

        self.assertEqual(octree.count, 4)

        # Search around origin within radius 2.5
        near_pts = octree.radius_search((0.0, 0.0, 0.0), radius=2.5)
        entity_ids = [pt[0] for pt in near_pts]
        self.assertIn(101, entity_ids)
        self.assertIn(102, entity_ids)
        self.assertNotIn(201, entity_ids)
        self.assertNotIn(301, entity_ids)

    def test_aabb_contains_and_intersects(self):
        box1 = SpatialAABB((0.0, 0.0, 0.0), (5.0, 5.0, 5.0))
        self.assertTrue(box1.contains((2.0, 3.0, 4.0)))
        self.assertFalse(box1.contains((6.0, 3.0, 4.0)))

        box2 = SpatialAABB((3.0, 3.0, 3.0), (8.0, 8.0, 8.0))
        self.assertTrue(box1.intersects(box2))

        box3 = SpatialAABB((10.0, 10.0, 10.0), (15.0, 15.0, 15.0))
        self.assertFalse(box1.intersects(box3))


if __name__ == "__main__":
    unittest.main()
