import unittest

import numpy as np

from core.domain.vision.color_spaces import AutoWhiteBalance, ColorSpaceConverter


class ColorSpaceConverterTests(unittest.TestCase):
    def test_rgb_to_grayscale_rec709(self):
        rgb = np.array([[[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0]]])
        gray = ColorSpaceConverter.rgb_to_grayscale(rgb, standard="rec709")
        self.assertEqual(gray.shape, (1, 3))
        # R: 0.2126*255 = 54.213, G: 0.7152*255 = 182.376, B: 0.0722*255 = 18.411
        self.assertAlmostEqual(gray[0, 0], 54.213, places=1)
        self.assertAlmostEqual(gray[0, 1], 182.376, places=1)
        self.assertAlmostEqual(gray[0, 2], 18.411, places=1)

    def test_rgb_to_grayscale_rec601(self):
        rgb = np.array([[[255.0, 0.0, 0.0]]])
        gray = ColorSpaceConverter.rgb_to_grayscale(rgb, standard="rec601")
        self.assertAlmostEqual(gray[0, 0], 0.2990 * 255.0, places=1)

    def test_rgb_to_grayscale_default(self):
        rgb = np.array([[[255.0, 255.0, 255.0]]])
        gray = ColorSpaceConverter.rgb_to_grayscale(rgb)
        self.assertAlmostEqual(gray[0, 0], 255.0, places=1)

    def test_rgb_to_hsv(self):
        rgb = np.array([[[255.0, 0.0, 0.0]]])
        hsv = ColorSpaceConverter.rgb_to_hsv(rgb)
        # Pure red: H=0, S=1, V=1
        self.assertAlmostEqual(hsv[0, 0, 0], 0.0)
        self.assertAlmostEqual(hsv[0, 0, 1], 1.0)
        self.assertAlmostEqual(hsv[0, 0, 2], 1.0)

    def test_hsv_to_rgb(self):
        hsv = np.array([[[0.0, 1.0, 1.0]]])
        rgb = ColorSpaceConverter.hsv_to_rgb(hsv)
        # Pure red
        self.assertAlmostEqual(rgb[0, 0, 0], 255.0)
        self.assertAlmostEqual(rgb[0, 0, 1], 0.0)
        self.assertAlmostEqual(rgb[0, 0, 2], 0.0)

    def test_rgb_to_cie_xyz(self):
        rgb = np.array([[[255.0, 255.0, 255.0]]])
        xyz = ColorSpaceConverter.rgb_to_cie_xyz(rgb)
        self.assertEqual(xyz.shape, (1, 1, 3))
        self.assertTrue(np.all(np.isfinite(xyz)))

    def test_cie_xyz_to_lab(self):
        xyz = np.array([[[0.95047, 1.0, 1.08883]]])
        lab = ColorSpaceConverter.cie_xyz_to_lab(xyz)
        self.assertEqual(lab.shape, (1, 1, 3))
        # D65 white point should give L*=100, a*=0, b*=0
        self.assertAlmostEqual(lab[0, 0, 0], 100.0, places=1)

    def test_rgb_to_yuv(self):
        rgb = np.array([[[255.0, 0.0, 0.0]]])
        yuv = ColorSpaceConverter.rgb_to_yuv(rgb)
        self.assertEqual(yuv.shape, (1, 1, 3))
        self.assertTrue(np.all(np.isfinite(yuv)))

    def test_rgb_to_ycbcr(self):
        rgb = np.array([[[255.0, 0.0, 0.0]]])
        ycbcr = ColorSpaceConverter.rgb_to_ycbcr(rgb)
        self.assertEqual(ycbcr.shape, (1, 1, 3))
        self.assertTrue(np.all(np.isfinite(ycbcr)))


class AutoWhiteBalanceTests(unittest.TestCase):
    def test_gray_world(self):
        image = np.full((10, 10, 3), [100.0, 150.0, 200.0], dtype=np.float64)
        balanced = AutoWhiteBalance.gray_world(image)
        self.assertEqual(balanced.shape, (10, 10, 3))
        self.assertTrue(np.all(np.isfinite(balanced)))

    def test_retinex_white_patch(self):
        image = np.full((10, 10, 3), [100.0, 150.0, 200.0], dtype=np.float64)
        balanced = AutoWhiteBalance.retinex_white_patch(image)
        self.assertEqual(balanced.shape, (10, 10, 3))
        self.assertTrue(np.all(np.isfinite(balanced)))


if __name__ == "__main__":
    unittest.main()
