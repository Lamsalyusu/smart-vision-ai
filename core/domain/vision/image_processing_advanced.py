from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math


class FilterType(Enum):
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"
    NON_LOCAL_MEANS = "non_local_means"
    GUIDED = "guided"
    ANISOTROPIC = "anisotropic"
    WAVELET = "wavelet"
    BM3D = "bm3d"


@dataclass(frozen=True, slots=True)
class ImageFilterParams:
    kernel_size: int = 5
    sigma: float = 1.0
    sigma_color: float = 75.0
    sigma_space: float = 75.0
    h: float = 10.0
    template_window: int = 7
    search_window: int = 21


class ImageFilter(ABC):
    @abstractmethod
    def apply(self, image: np.ndarray, params: ImageFilterParams = None) -> np.ndarray:
        pass


class GaussianFilter(ImageFilter):
    def apply(self, image: np.ndarray, params: ImageFilterParams = None) -> np.ndarray:
        params = params or ImageFilterParams()
        ksize = params.kernel_size if params.kernel_size % 2 == 1 else params.kernel_size + 1
        return cv2.GaussianBlur(image, (ksize, ksize), params.sigma)


class MedianFilter(ImageFilter):
    def apply(self, image: np.ndarray, params: ImageFilterParams = None) -> np.ndarray:
        params = params or ImageFilterParams()
        ksize = params.kernel_size if params.kernel_size % 2 == 1 else params.kernel_size + 1
        return cv2.medianBlur(image, ksize)


class BilateralFilter(ImageFilter):
    def apply(self, image: np.ndarray, params: ImageFilterParams = None) -> np.ndarray:
        params = params or ImageFilterParams()
        d = params.kernel_size if params.kernel_size > 0 else 9
        return cv2.bilateralFilter(image, d, params.sigma_color, params.sigma_space)


class NonLocalMeansFilter(ImageFilter):
    def apply(self, image: np.ndarray, params: ImageFilterParams = None) -> np.ndarray:
        params = params or ImageFilterParams()
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, params.h, params.h, params.template_window, params.search_window
            )
        else:
            return cv2.fastNlMeansDenoising(
                image, None, params.h, params.template_window, params.search_window
            )


class GuidedFilter(ImageFilter):
    def __init__(self, radius: int = 4, eps: float = 1e-3):
        self.radius = radius
        self.eps = eps

    def apply(self, image: np.ndarray, params: ImageFilterParams = None) -> np.ndarray:
        if len(image.shape) == 3:
            guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            guide = image.astype(np.float32) / 255.0

        image_f = image.astype(np.float32) / 255.0

        mean_I = cv2.boxFilter(guide, -1, (self.radius, self.radius))
        mean_p = cv2.boxFilter(image_f, -1, (self.radius, self.radius))
        mean_Ip = cv2.boxFilter(guide * image_f, -1, (self.radius, self.radius))

        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(guide * guide, -1, (self.radius, self.radius))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, (self.radius, self.radius))
        mean_b = cv2.boxFilter(b, -1, (self.radius, self.radius))

        q = mean_a * guide + mean_b
        return (q * 255).astype(np.uint8)


class AnisotropicDiffusionFilter(ImageFilter):
    def __init__(
        self,
        num_iter: int = 10,
        delta_t: float = 0.125,
        kappa: float = 50.0,
        option: int = 1,
    ):
        self.num_iter = num_iter
        self.delta_t = delta_t
        self.kappa = kappa
        self.option = option

    def apply(self, image: np.ndarray, params: ImageFilterParams = None) -> np.ndarray:
        img = image.astype(np.float32) / 255.0

        for _ in range(self.num_iter):
            grad_n = np.roll(img, -1, axis=0) - img
            grad_s = np.roll(img, 1, axis=0) - img
            grad_e = np.roll(img, -1, axis=1) - img
            grad_w = np.roll(img, 1, axis=1) - img

            if self.option == 1:
                c_n = np.exp(-(grad_n / self.kappa)**2)
                c_s = np.exp(-(grad_s / self.kappa)**2)
                c_e = np.exp(-(grad_e / self.kappa)**2)
                c_w = np.exp(-(grad_w / self.kappa)**2)
            else:
                c_n = 1.0 / (1.0 + (grad_n / self.kappa)**2)
                c_s = 1.0 / (1.0 + (grad_s / self.kappa)**2)
                c_e = 1.0 / (1.0 + (grad_e / self.kappa)**2)
                c_w = 1.0 / (1.0 + (grad_w / self.kappa)**2)

            img = img + self.delta_t * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)

        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


class WaveletFilter(ImageFilter):
    def __init__(self, wavelet: str = 'db1', level: int = 3, threshold_mode: str = 'soft'):
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode

    def apply(self, image: np.ndarray, params: ImageFilterParams = None) -> np.ndarray:
        try:
            import pywt
        except ImportError:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        if len(image.shape) == 3:
            channels = cv2.split(image)
            denoised = []
            for ch in channels:
                coeffs = pywt.wavedec2(ch, self.wavelet, level=self.level)
                cA, details = coeffs[0], coeffs[1:]

                for i in range(len(details)):
                    cH, cV, cD = details[i]
                    threshold = np.std(cD) * np.sqrt(2 * np.log(cD.size))
                    if self.threshold_mode == 'soft':
                        cH = np.sign(cH) * np.maximum(np.abs(cH) - threshold, 0)
                        cV = np.sign(cV) * np.maximum(np.abs(cV) - threshold, 0)
                        cD = np.sign(cD) * np.maximum(np.abs(cD) - threshold, 0)
                    else:
                        cH = cH * (np.abs(cH) > threshold)
                        cV = cV * (np.abs(cV) > threshold)
                        cD = cD * (np.abs(cD) > threshold)
                    details[i] = (cH, cV, cD)

                denoised.append(pywt.waverec2([cA] + details, self.wavelet))

            return cv2.merge(denoised).astype(np.uint8)
        else:
            coeffs = pywt.wavedec2(image, self.wavelet, level=self.level)
            cA, details = coeffs[0], coeffs[1:]

            for i in range(len(details)):
                cH, cV, cD = details[i]
                threshold = np.std(cD) * np.sqrt(2 * np.log(cD.size))
                if self.threshold_mode == 'soft':
                    cH = np.sign(cH) * np.maximum(np.abs(cH) - threshold, 0)
                    cV = np.sign(cV) * np.maximum(np.abs(cV) - threshold, 0)
                    cD = np.sign(cD) * np.maximum(np.abs(cD) - threshold, 0)
                else:
                    cH = cH * (np.abs(cH) > threshold)
                    cV = cV * (np.abs(cV) > threshold)
                    cD = cD * (np.abs(cD) > threshold)
                details[i] = (cH, cV, cD)

            return pywt.waverec2([cA] + details, self.wavelet).astype(np.uint8)


class BM3DFilter(ImageFilter):
    def __init__(self, sigma: float = 25.0):
        self.sigma = sigma

    def apply(self, image: np.ndarray, params: ImageFilterParams = None) -> np.ndarray:
        try:
            import bm3d
            if len(image.shape) == 3:
                return bm3d.bm3d_rgb(image, self.sigma/255.0).astype(np.uint8)
            else:
                return bm3d.bm3d(image, self.sigma/255.0).astype(np.uint8)
        except ImportError:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


class ImageEnhancer:
    def __init__(self):
        self.filters: List[Tuple[ImageFilter, ImageFilterParams]] = []

    def add_filter(self, filter_obj: ImageFilter, params: ImageFilterParams = None):
        self.filters.append((filter_obj, params))

    def enhance(self, image: np.ndarray) -> np.ndarray:
        result = image
        for filter_obj, params in self.filters:
            result = filter_obj.apply(result, params)
        return result


class HistogramEqualizer:
    @staticmethod
    def equalize(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            return cv2.equalizeHist(image)

    @staticmethod
    def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            return clahe.apply(image)

    @staticmethod
    def adaptive_equalize(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        return HistogramEqualizer.clahe(image, clip_limit)


class ColorCorrector:
    @staticmethod
    def white_balance(image: np.ndarray, method: str = "gray_world") -> np.ndarray:
        if method == "gray_world":
            return ColorCorrector._gray_world(image)
        elif method == "white_patch":
            return ColorCorrector._white_patch(image)
        elif method == "shades_of_gray":
            return ColorCorrector._shades_of_gray(image)
        return image

    @staticmethod
    def _gray_world(image: np.ndarray) -> np.ndarray:
        avg_b = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_r = np.mean(image[:, :, 2])

        avg_gray = (avg_r + avg_g + avg_b) / 3

        scale_b = avg_gray / avg_b if avg_b > 0 else 1
        scale_g = avg_gray / avg_g if avg_g > 0 else 1
        scale_r = avg_gray / avg_r if avg_r > 0 else 1

        result = image.copy().astype(np.float32)
        result[:, :, 0] *= scale_b
        result[:, :, 1] *= scale_g
        result[:, :, 2] *= scale_r
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _white_patch(image: np.ndarray) -> np.ndarray:
        max_b = np.max(image[:, :, 0])
        max_g = np.max(image[:, :, 1])
        max_r = np.max(image[:, :, 2])

        scale_b = 255 / max_b if max_b > 0 else 1
        scale_g = 255 / max_g if max_g > 0 else 1
        scale_r = 255 / max_r if max_r > 0 else 1

        result = image.copy().astype(np.float32)
        result[:, :, 0] *= scale_b
        result[:, :, 1] *= scale_g
        result[:, :, 2] *= scale_r
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _shades_of_gray(image: np.ndarray, p: int = 6) -> np.ndarray:
        mean_r = np.mean(image[:, :, 2] ** p) ** (1/p)
        mean_g = np.mean(image[:, :, 1] ** p) ** (1/p)
        mean_b = np.mean(image[:, :, 0] ** p) ** (1/p)

        avg = (mean_r + mean_g + mean_b) / 3

        scale_r = avg / mean_r if mean_r > 0 else 1
        scale_g = avg / mean_g if mean_g > 0 else 1
        scale_b = avg / mean_b if mean_b > 0 else 1

        result = image.copy().astype(np.float32)
        result[:, :, 2] *= scale_r
        result[:, :, 1] *= scale_g
        result[:, :, 0] *= scale_b
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)

        src_mean, src_std = cv2.meanStdDev(src_lab)
        tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)

        result = (src_lab - src_mean.reshape(3, 1)) * (tgt_std / src_std).reshape(3, 1) + tgt_mean.reshape(3, 1)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)


class ImageSharpener:
    @staticmethod
    def unsharp_mask(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0, threshold: int = 0) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        if threshold > 0:
            diff = cv2.absdiff(image, blurred)
            mask = np.any(diff > threshold, axis=2)
            sharpened[~mask] = image[~mask]
        return sharpened

    @staticmethod
    def laplacian_sharpen(image: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpened = image - alpha * laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def high_boost(image: np.ndarray, k: float = 1.5) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        mask = cv2.subtract(image, blurred)
        return cv2.add(image, cv2.multiply(mask, k))


class ImageMorphology:
    @staticmethod
    def dilate(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.dilate(image, kernel, iterations=iterations)

    @staticmethod
    def erode(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.erode(image, kernel, iterations=iterations)

    @staticmethod
    def open(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def close(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def gradient(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

    @staticmethod
    def tophat(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    @staticmethod
    def blackhat(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    @staticmethod
    def skeletonize(image: np.ndarray) -> np.ndarray:
        image_bin = image > 128
        skeleton = np.zeros_like(image, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(image_bin.astype(np.uint8), element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(image_bin.astype(np.uint8), temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            image_bin = eroded
            if cv2.countNonZero(image_bin) == 0:
                break

        return skeleton * 255


class ImageTransform:
    @staticmethod
    def rotate(image: np.ndarray, angle: float, center: Tuple[int, int] = None, scale: float = 1.0) -> np.ndarray:
        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(image, M, (w, h))

    @staticmethod
    def translate(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    @staticmethod
    def scale(image: np.ndarray, fx: float, fy: float, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        return cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)

    @staticmethod
    def flip(image: np.ndarray, flip_code: int) -> np.ndarray:
        return cv2.flip(image, flip_code)

    @staticmethod
    def affine_transform(image: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        M = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    @staticmethod
    def perspective_transform(image: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
        return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    @staticmethod
    def crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        return image[y:y+h, x:x+w]

    @staticmethod
    def pad(image: np.ndarray, top: int, bottom: int, left: int, right: int, value: Tuple = (0, 0, 0)) -> np.ndarray:
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)


class ImageFeatureExtractor:
    @staticmethod
    def harris_corners(image: np.ndarray, block_size: int = 2, ksize: int = 3, k: float = 0.04) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        dst = cv2.cornerHarris(gray.astype(np.float32), block_size, ksize, k)
        dst = cv2.dilate(dst, None)
        return dst

    @staticmethod
    def shi_tomasi(image: np.ndarray, max_corners: int = 100, quality_level: float = 0.01, min_distance: int = 10) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
        return corners

    @staticmethod
    def fast_corners(image: np.ndarray, threshold: int = 10, nonmax_suppression: bool = True) -> List[cv2.KeyPoint]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax_suppression)
        return fast.detect(gray, None)

    @staticmethod
    def orb_features(image: np.ndarray, nfeatures: int = 500) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        orb = cv2.ORB_create(nfeatures=nfeatures)
        return orb.detectAndCompute(gray, None)

    @staticmethod
    def sift_features(image: np.ndarray, nfeatures: int = 500) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        sift = cv2.SIFT_create(nfeatures=nfeatures)
        return sift.detectAndCompute(gray, None)

    @staticmethod
    def surf_features(image: np.ndarray, hessian_threshold: float = 400) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
        return surf.detectAndCompute(gray, None)


class ImageSegmenter:
    @staticmethod
    def watershed(image: np.ndarray, markers: np.ndarray = None) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        if markers is None:
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0

        markers = cv2.watershed(image, markers)
        return markers

    @staticmethod
    def grabcut(image: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return image * mask[:, :, np.newaxis]

    @staticmethod
    def kmeans_segmentation(image: np.ndarray, k: int = 3) -> np.ndarray:
        data = image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        return segmented.reshape(image.shape)

    @staticmethod
    def meanshift_segmentation(image: np.ndarray, sp: int = 10, sr: int = 20) -> np.ndarray:
        return cv2.pyrMeanShiftFiltering(image, sp, sr)


class ImageCompression:
    @staticmethod
    def jpeg_compress(image: np.ndarray, quality: int = 90) -> bytes:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
        return buffer.tobytes()

    @staticmethod
    def jpeg_decompress(data: bytes) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def png_compress(image: np.ndarray, compression: int = 3) -> bytes:
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), compression]
        _, buffer = cv2.imencode('.png', image, encode_param)
        return buffer.tobytes()

    @staticmethod
    def webp_compress(image: np.ndarray, quality: int = 90) -> bytes:
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        _, buffer = cv2.imencode('.webp', image, encode_param)
        return buffer.tobytes()


class ImageQualityAssessor:
    @staticmethod
    def psnr(original: np.ndarray, compressed: np.ndarray) -> float:
        mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))

    @staticmethod
    def ssim(original: np.ndarray, compressed: np.ndarray) -> float:
        from skimage.metrics import structural_similarity
        return structural_similarity(original, compressed, multichannel=True, channel_axis=2)

    @staticmethod
    def mse(original: np.ndarray, compressed: np.ndarray) -> float:
        return np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)

    @staticmethod
    def sharpness(image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    @staticmethod
    def contrast(image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return gray.std()

    @staticmethod
    def brightness(image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return gray.mean()

    @staticmethod
    def noise_level(image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        h, w = gray.shape
        noise = gray[1:, 1:] - gray[:-1, :-1]
        return noise.std()
