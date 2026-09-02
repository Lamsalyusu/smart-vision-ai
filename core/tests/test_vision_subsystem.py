import unittest
import numpy as np

from core.domain.vision import (
    ImageFilterKernel,
    SpatialConvolutionEngine,
    BilateralFilter,
    CannyEdgeDetector,
    MathematicalMorphology,
    StructuringElement,
    ColorSpaceConverter,
    AutoWhiteBalance,
    HarrisCornerDetector,
    FastFeatureDetector,
    PlanarHomographySolver,
    ImageSegmentationEngine,
    PinholeCameraModel,
)


class TestVisionSubsystem(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((32, 32), dtype=np.uint8)
        self.img[8:24, 8:24] = 255  # Solid bright square in center

    def test_gaussian_convolution(self):
        k = ImageFilterKernel.gaussian_2d(5, 1.0)
        smoothed = SpatialConvolutionEngine.convolve2d(self.img.astype(float), k)
        self.assertEqual(smoothed.shape, self.img.shape)
        self.assertLess(smoothed[8, 8], 255.0)  # Blurred edge

    def test_morphology_erosion_dilation(self):
        elem = StructuringElement.rectangle(3, 3)
        eroded = MathematicalMorphology.erode(self.img, elem)
        dilated = MathematicalMorphology.dilate(self.img, elem)
        self.assertLess(np.sum(eroded > 0), np.sum(self.img > 0))
        self.assertGreater(np.sum(dilated > 0), np.sum(self.img > 0))

    def test_color_space_conversions(self):
        rgb = np.full((10, 10, 3), 128, dtype=np.uint8)
        gray = ColorSpaceConverter.rgb_to_grayscale(rgb)
        self.assertEqual(gray.shape, (10, 10))
        hsv = ColorSpaceConverter.rgb_to_hsv(rgb)
        self.assertEqual(hsv.shape, (10, 10, 3))

    def test_harris_and_fast_detectors(self):
        harris = HarrisCornerDetector(threshold=100.0)
        kp_h = harris.detect(self.img)
        self.assertIsInstance(kp_h, list)

        fast = FastFeatureDetector(threshold=15)
        kp_f = fast.detect(self.img)
        self.assertIsInstance(kp_f, list)

    def test_homography_dlt(self):
        src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        dst = np.array([[2, 3], [12, 3], [12, 13], [2, 13]], dtype=float)
        h = PlanarHomographySolver.estimate_dlt(src, dst)
        self.assertEqual(h.shape, (3, 3))

    def test_pinhole_camera_projection(self):
        cam = PinholeCameraModel(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        pt_3d = np.array([0.5, 0.2, 2.0])
        px = cam.project_3d_to_pixel(pt_3d)
        self.assertEqual(len(px), 2)
        self.assertAlmostEqual(px[0], 320.0 + 500.0 * (0.5 / 2.0), delta=1e-3)


if __name__ == "__main__":
    unittest.main()
