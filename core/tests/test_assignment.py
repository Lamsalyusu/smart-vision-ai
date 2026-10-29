import unittest

import numpy as np

from legacy_django.backend.ml_model.tracker import linear_assignment


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


if __name__ == "__main__":
    unittest.main()