import math
import unittest

from core.domain.tracking.reid import (
    AppearanceEmbedding,
    FeatureGallery,
    composite_association_cost,
    compute_reid_cost_matrix,
)


class TrackingReIDTests(unittest.TestCase):
    def test_appearance_embedding_normalization_and_distance(self) -> None:
        emb1 = AppearanceEmbedding.from_raw([3.0, 4.0, 0.0])
        # Norm is 5.0, so vector should be [0.6, 0.8, 0.0]
        self.assertAlmostEqual(emb1.vector[0], 0.6)
        self.assertAlmostEqual(emb1.vector[1], 0.8)

        # Identical vector has distance ~0
        self.assertAlmostEqual(emb1.cosine_distance(emb1), 0.0)

        # Orthogonal vector has distance 1.0
        emb_ortho = AppearanceEmbedding.from_raw([0.0, 0.0, 5.0])
        self.assertAlmostEqual(emb1.cosine_distance(emb_ortho), 1.0)

        # Opposite vector has distance 2.0
        emb_opposite = AppearanceEmbedding.from_raw([-3.0, -4.0, 0.0])
        self.assertAlmostEqual(emb1.cosine_distance(emb_opposite), 2.0)

    def test_feature_gallery_ema_updates(self) -> None:
        gallery = FeatureGallery(max_history=5, momentum=0.5)
        self.assertIsNone(gallery.current_embedding)

        emb1 = AppearanceEmbedding.from_raw([1.0, 0.0])
        gallery.update(emb1)
        self.assertEqual(gallery.current_embedding, emb1)

        emb2 = AppearanceEmbedding.from_raw([0.0, 1.0])
        gallery.update(emb2)

        # Blended vector should be normalized [0.5, 0.5] -> [1/sqrt(2), 1/sqrt(2)]
        curr = gallery.current_embedding
        self.assertIsNotNone(curr)
        expected = 1.0 / math.sqrt(2)
        self.assertAlmostEqual(curr.vector[0], expected)
        self.assertAlmostEqual(curr.vector[1], expected)

    def test_compute_reid_cost_matrix(self) -> None:
        g1 = FeatureGallery()
        g1.update(AppearanceEmbedding.from_raw([1.0, 0.0]))

        g2 = FeatureGallery()
        g2.update(AppearanceEmbedding.from_raw([0.0, 1.0]))

        cand1 = AppearanceEmbedding.from_raw([1.0, 0.0])
        cand2 = AppearanceEmbedding.from_raw([0.0, 1.0])

        costs = compute_reid_cost_matrix([g1, g2], [cand1, cand2])
        self.assertEqual(len(costs), 2)
        self.assertEqual(len(costs[0]), 2)

        self.assertAlmostEqual(costs[0][0], 0.0)
        self.assertAlmostEqual(costs[0][1], 1.0)
        self.assertAlmostEqual(costs[1][0], 1.0)
        self.assertAlmostEqual(costs[1][1], 0.0)

    def test_composite_association_cost_gating(self) -> None:
        # Both within gates
        cost = composite_association_cost(
            spatial_cost=4.0,
            appearance_cost=0.2,
            weight_spatial=0.5,
            spatial_gate=16.0,
            appearance_gate=0.5,
        )
        self.assertTrue(math.isfinite(cost))
        self.assertGreater(cost, 0.0)

        # Spatial cost exceeds gate -> inf
        cost_gated_spatial = composite_association_cost(
            spatial_cost=20.0,
            appearance_cost=0.1,
            spatial_gate=16.0,
        )
        self.assertEqual(cost_gated_spatial, math.inf)

        # Appearance cost exceeds gate -> inf
        cost_gated_app = composite_association_cost(
            spatial_cost=2.0,
            appearance_cost=0.8,
            appearance_gate=0.5,
        )
        self.assertEqual(cost_gated_app, math.inf)


if __name__ == "__main__":
    unittest.main()
