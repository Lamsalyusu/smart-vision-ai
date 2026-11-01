import unittest

import numpy as np

from core.domain.tracking.association import (
    calculate_iou,
    fuse_reid_mahalanobis_cost,
    iou_cost_matrix,
    linear_assignment,
)


class JonkerVolgenantAssignmentTests(unittest.TestCase):
    def test_finds_global_minimum_where_greedy_assignment_fails(self) -> None:
        costs = np.array([[1.0, 2.0], [1.1, 100.0]])

        matches = linear_assignment(costs)

        np.testing.assert_array_equal(matches, np.array([[0, 1], [1, 0]]))
        self.assertAlmostEqual(float(costs[matches[:, 0], matches[:, 1]].sum()), 3.1)

    def test_handles_rectangular_cost_matrix(self) -> None:
        matches = linear_assignment(np.array([[3.0, 1.0, 2.0], [1.0, 4.0, 3.0]]))

        np.testing.assert_array_equal(matches, np.array([[0, 1], [1, 0]]))

    def test_returns_empty_assignment_for_empty_dimension(self) -> None:
        matches = linear_assignment(np.empty((0, 3)))

        self.assertEqual(matches.shape, (0, 2))

    def test_filters_gated_costs(self) -> None:
        matches = linear_assignment(np.array([[1.0, 1e5], [1e5, 1e5]]))

        np.testing.assert_array_equal(matches, np.array([[0, 0]]))

    def test_rejects_non_finite_costs(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            linear_assignment(np.array([[0.0, np.inf]]))

    def test_rejects_non_matrix_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            linear_assignment(np.array([1.0, 2.0]))

    def test_calculate_iou_accurate(self) -> None:
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [5.0, 0.0, 15.0, 10.0]
        iou = calculate_iou(box1, box2)
        # Intersection = 5 * 10 = 50. Union = 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
        self.assertAlmostEqual(iou, 1.0 / 3.0)

    def test_iou_cost_matrix(self) -> None:
        boxes_a = [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
        boxes_b = [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]
        costs = iou_cost_matrix(boxes_a, boxes_b)
        self.assertEqual(costs.shape, (2, 2))
        self.assertAlmostEqual(costs[0, 0], 0.0)  # perfect overlap -> cost 0
        self.assertAlmostEqual(costs[0, 1], 1.0)  # disjoint -> cost 1

    def test_fuse_reid_mahalanobis_cost_gates_outliers(self) -> None:
        reid = np.array([[0.1, 0.8], [0.2, 0.3]])
        maha = np.array([[2.0, 15.0], [4.0, 3.0]])  # 15.0 exceeds 9.488 gate
        fused = fuse_reid_mahalanobis_cost(reid, maha, alpha=0.5, mahalanobis_gate=9.488)
        self.assertAlmostEqual(fused[0, 0], 0.5 * 0.1 + 0.5 * 2.0)
        self.assertEqual(fused[0, 1], 1e5)  # gated


if __name__ == "__main__":
    unittest.main()