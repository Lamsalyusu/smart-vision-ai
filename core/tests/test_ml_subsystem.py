import unittest
import numpy as np

from core.domain.ml import (
    Tensor,
    Linear,
    Activations,
    LossFunctions,
    AdamW,
    Int8UniformQuantizer,
)


class TestMlSubsystem(unittest.TestCase):
    def test_tensor_autograd_graph(self):
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        w = Tensor([[3.0], [4.0]], requires_grad=True)
        y = x @ w
        y.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        np.testing.assert_allclose(x.grad, [[3.0, 4.0]])
        np.testing.assert_allclose(w.grad, [[1.0], [2.0]])

    def test_linear_layer_forward(self):
        layer = Linear(in_features=4, out_features=2)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = layer(x)
        self.assertEqual(out.shape, (1, 2))

    def test_activations_and_loss(self):
        x = np.array([0.0, 1.0, 2.0])
        sm = Activations.softmax(x)
        self.assertAlmostEqual(float(np.sum(sm)), 1.0, delta=1e-5)

        loss = LossFunctions.cross_entropy_with_logits(np.array([[2.0, 0.5]]), np.array([0]))
        self.assertGreater(loss, 0.0)

    def test_int8_quantization(self):
        weights = np.random.randn(10, 10).astype(np.float32)
        scale, q_weights = Int8UniformQuantizer.calibrate_symmetric(weights)
        dequant = Int8UniformQuantizer.dequantize_symmetric(scale, q_weights)
        self.assertEqual(q_weights.dtype, np.int8)
        np.testing.assert_allclose(dequant, weights, atol=scale * 1.5)


if __name__ == "__main__":
    unittest.main()
