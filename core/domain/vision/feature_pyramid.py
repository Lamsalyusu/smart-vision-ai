from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence, Optional, Callable
from enum import Enum
import cv2


class PyramidLevel(Enum):
    P2 = 2
    P3 = 3
    P4 = 4
    P5 = 5
    P6 = 6
    P7 = 7


@dataclass(frozen=True, slots=True)
class PyramidFeatures:
    levels: dict[PyramidLevel, np.ndarray]
    base_resolution: tuple[int, int]
    scale_factors: dict[PyramidLevel, float]

    def get_level(self, level: PyramidLevel) -> Optional[np.ndarray]:
        return self.levels.get(level)

    def get_all_levels(self) -> list[np.ndarray]:
        return [self.levels[l] for l in sorted(self.levels.keys())]


class FeaturePyramidNetwork:
    def __init__(
        self,
        min_level: PyramidLevel = PyramidLevel.P3,
        max_level: PyramidLevel = PyramidLevel.P7,
        num_channels: int = 256,
        use_p5: bool = True,
        interpolation: str = "nearest",
    ):
        self.min_level = min_level
        self.max_level = max_level
        self.num_channels = num_channels
        self.use_p5 = use_p5
        self.interpolation = interpolation
        self._lateral_convs: dict[PyramidLevel, Callable] = {}
        self._output_convs: dict[PyramidLevel, Callable] = {}
        self._initialized = False

    def _initialize(self, backbone_channels: dict[PyramidLevel, int]):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class ConvBlock(nn.Module):
                def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
                    super().__init__()
                    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
                    self.norm = nn.BatchNorm2d(out_channels)
                    self.relu = nn.ReLU(inplace=True)

                def forward(self, x):
                    return self.relu(self.norm(self.conv(x)))

            self._lateral_convs = {}
            self._output_convs = {}

            for level in PyramidLevel:
                if self.min_level.value <= level.value <= self.max_level.value:
                    in_ch = backbone_channels.get(level, self.num_channels)
                    self._lateral_convs[level] = ConvBlock(in_ch, self.num_channels, 1, 1, 0)
                    self._output_convs[level] = ConvBlock(self.num_channels, self.num_channels, 3, 1, 1)

            if self.max_level == PyramidLevel.P6:
                self._p6_conv = ConvBlock(self.num_channels, self.num_channels, 3, 2, 1)
            if self.max_level == PyramidLevel.P7:
                self._p7_conv = ConvBlock(self.num_channels, self.num_channels, 3, 2, 1)

            self._initialized = True
            self._torch = torch
            self._F = F

        except ImportError:
            self._initialized = False

    def forward(self, backbone_features: dict[PyramidLevel, np.ndarray]) -> PyramidFeatures:
        if not self._initialized:
            backbone_channels = {k: v.shape[1] for k, v in backbone_features.items()}
            self._initialize(backbone_channels)

        if self._initialized:
            return self._forward_torch(backbone_features)
        else:
            return self._forward_numpy(backbone_features)

    def _forward_torch(self, backbone_features: dict[PyramidLevel, np.ndarray]) -> PyramidFeatures:
        torch = self._torch
        F = self._F

        tensor_features = {}
        for level, feat in backbone_features.items():
            if feat.ndim == 3:
                feat = feat[None]
            tensor_features[level] = torch.from_numpy(feat).float()

        laterals = {}
        for level in sorted(tensor_features.keys(), key=lambda x: x.value):
            if level in self._lateral_convs:
                laterals[level] = self._lateral_convs[level](tensor_features[level])
            else:
                laterals[level] = tensor_features[level]

        p5 = laterals.get(PyramidLevel.P5)
        if p5 is not None:
            for level in [PyramidLevel.P4, PyramidLevel.P3]:
                if level in laterals:
                    upsampled = F.interpolate(
                        p5, size=laterals[level].shape[-2:],
                        mode=self.interpolation, align_corners=False
                    )
                    laterals[level] = laterals[level] + upsampled

        outputs = {}
        for level in sorted(laterals.keys(), key=lambda x: x.value):
            if level in self._output_convs:
                outputs[level] = self._output_convs[level](laterals[level])
            else:
                outputs[level] = laterals[level]

        if self.max_level >= PyramidLevel.P6 and PyramidLevel.P5 in outputs:
            outputs[PyramidLevel.P6] = self._p6_conv(outputs[PyramidLevel.P5])
        if self.max_level >= PyramidLevel.P7 and PyramidLevel.P6 in outputs:
            outputs[PyramidLevel.P7] = self._p7_conv(outputs[PyramidLevel.P6])

        numpy_outputs = {k: v.detach().cpu().numpy() for k, v in outputs.items()}

        base_h, base_w = backbone_features[PyramidLevel.P3].shape[-2:]
        scale_factors = {level: base_h / feat.shape[-2] for level, feat in numpy_outputs.items()}

        return PyramidFeatures(
            levels={k: v[0] if v.ndim == 4 else v for k, v in numpy_outputs.items()},
            base_resolution=(base_h, base_w),
            scale_factors=scale_factors,
        )

    def _forward_numpy(self, backbone_features: dict[PyramidLevel, np.ndarray]) -> PyramidFeatures:
        laterals = {}
        for level in sorted(backbone_features.keys(), key=lambda x: x.value):
            feat = backbone_features[level]
            if feat.ndim == 3:
                feat = feat[None]
            laterals[level] = self._lateral_conv_numpy(feat, self.num_channels)

        p5 = laterals.get(PyramidLevel.P5)
        if p5 is not None:
            for level in [PyramidLevel.P4, PyramidLevel.P3]:
                if level in laterals:
                    target_shape = laterals[level].shape[-2:]
                    upsampled = self._upsample(p5, target_shape)
                    laterals[level] = laterals[level] + upsampled

        outputs = {}
        for level in sorted(laterals.keys(), key=lambda x: x.value):
            outputs[level] = self._output_conv_numpy(laterals[level], self.num_channels)

        if self.max_level >= PyramidLevel.P6 and PyramidLevel.P5 in outputs:
            outputs[PyramidLevel.P6] = self._stride_conv_numpy(outputs[PyramidLevel.P5], self.num_channels)
        if self.max_level >= PyramidLevel.P7 and PyramidLevel.P6 in outputs:
            outputs[PyramidLevel.P7] = self._stride_conv_numpy(outputs[PyramidLevel.P6], self.num_channels)

        numpy_outputs = {k: v[0] if v.ndim == 4 else v for k, v in outputs.items()}
        base_h, base_w = backbone_features[PyramidLevel.P3].shape[-2:]
        scale_factors = {level: base_h / feat.shape[-2] for level, feat in numpy_outputs.items()}

        return PyramidFeatures(
            levels=numpy_outputs,
            base_resolution=(base_h, base_w),
            scale_factors=scale_factors,
        )

    def _lateral_conv_numpy(self, x: np.ndarray, out_channels: int) -> np.ndarray:
        in_channels = x.shape[1]
        kernel = np.random.randn(out_channels, in_channels, 1, 1).astype(np.float32) * 0.01
        return self._conv2d_numpy(x, kernel, stride=1, padding=0)

    def _output_conv_numpy(self, x: np.ndarray, out_channels: int) -> np.ndarray:
        in_channels = x.shape[1]
        kernel = np.random.randn(out_channels, in_channels, 3, 3).astype(np.float32) * 0.01
        return self._conv2d_numpy(x, kernel, stride=1, padding=1)

    def _stride_conv_numpy(self, x: np.ndarray, out_channels: int) -> np.ndarray:
        in_channels = x.shape[1]
        kernel = np.random.randn(out_channels, in_channels, 3, 3).astype(np.float32) * 0.01
        return self._conv2d_numpy(x, kernel, stride=2, padding=1)

    def _conv2d_numpy(self, x: np.ndarray, kernel: np.ndarray, stride: int, padding: int) -> np.ndarray:
        batch, in_c, h, w = x.shape
        out_c, _, kh, kw = kernel.shape

        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

        oh = (h + 2 * padding - kh) // stride + 1
        ow = (w + 2 * padding - kw) // stride + 1

        out = np.zeros((batch, out_c, oh, ow), dtype=np.float32)

        for i in range(oh):
            for j in range(ow):
                h_start, h_end = i * stride, i * stride + kh
                w_start, w_end = j * stride, j * stride + kw
                patch = x[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.sum(patch[:, :, None] * kernel[None], axis=(1, 3, 4))

        return out

    def _upsample(self, x: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        batch, c, h, w = x.shape
        th, tw = target_shape

        if self.interpolation == "nearest":
            h_scale, w_scale = th / h, tw / w
            h_indices = np.clip((np.arange(th) / h_scale).astype(int), 0, h - 1)
            w_indices = np.clip((np.arange(tw) / w_scale).astype(int), 0, w - 1)
            return x[:, :, h_indices[:, None], w_indices[None, :]]
        else:
            return cv2.resize(x[0].transpose(1, 2, 0), (tw, th), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)[None]


class BiFPN:
    def __init__(
        self,
        num_levels: int = 5,
        num_channels: int = 256,
        num_iterations: int = 3,
        epsilon: float = 1e-4,
        use_attention: bool = True,
        use_residual: bool = True,
    ):
        self.num_levels = num_levels
        self.num_channels = num_channels
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.use_attention = use_attention
        self.use_residual = use_residual
        self._weights: list[np.ndarray] = []
        self._convs: list[list[Callable]] = []

    def _initialize_weights(self, num_inputs: int = 2):
        if self.use_attention:
            w = np.ones(num_inputs, dtype=np.float32)
            self._weights.append(w / w.sum())
        else:
            self._weights.append(np.ones(num_inputs, dtype=np.float32) / num_inputs)

    def forward(self, features: list[np.ndarray]) -> list[np.ndarray]:
        if not self._convs:
            self._initialize_convs(len(features))

        current = features
        for iteration in range(self.num_iterations):
            current = self._bifpn_iteration(current, iteration)
        return current

    def _initialize_convs(self, num_levels: int):
        try:
            import torch
            import torch.nn as nn

            class SeparableConv(nn.Module):
                def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
                    super().__init__()
                    self.depthwise = nn.Conv2d(
                        in_channels, in_channels, kernel_size, stride, padding,
                        groups=in_channels, bias=False
                    )
                    self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
                    self.norm = nn.BatchNorm2d(out_channels)
                    self.relu = nn.ReLU(inplace=True)

                def forward(self, x):
                    return self.relu(self.norm(self.pointwise(self.depthwise(x))))

            for _ in range(self.num_iterations):
                iter_convs = []
                for _ in range(num_levels):
                    iter_convs.append(SeparableConv(self.num_channels, self.num_channels))
                self._convs.append(iter_convs)

            self._torch = torch
            self._nn = nn

        except ImportError:
            for _ in range(self.num_iterations):
                self._convs.append([None] * num_levels)

    def _bifpn_iteration(self, features: list[np.ndarray], iteration: int) -> list[np.ndarray]:
        num_levels = len(features)

        if self._convs and self._convs[0][0] is not None:
            return self._bifpn_iteration_torch(features, iteration)
        else:
            return self._bifpn_iteration_numpy(features, iteration)

    def _bifpn_iteration_torch(self, features: list[np.ndarray], iteration: int) -> list[np.ndarray]:
        torch = self._torch
        F = torch.nn.functional

        tensor_feats = [torch.from_numpy(f).float() for f in features]

        top_down = [tensor_feats[-1]]
        for i in range(len(tensor_feats) - 2, -1, -1):
            upsampled = F.interpolate(top_down[0], size=tensor_feats[i].shape[-2:], mode='nearest')
            if self.use_attention and iteration < len(self._weights):
                w = self._weights[iteration]
                fused = (w[0] * tensor_feats[i] + w[1] * upsampled) / (w.sum() + self.epsilon)
            else:
                fused = (tensor_feats[i] + upsampled) / 2
            top_down.insert(0, fused)

        bottom_up = [top_down[0]]
        for i in range(1, num_levels):
            downsampled = F.max_pool2d(bottom_up[-1], kernel_size=2, stride=2)
            if downsampled.shape[-2:] != top_down[i].shape[-2:]:
                downsampled = F.interpolate(downsampled, size=top_down[i].shape[-2:], mode='nearest')
            if self.use_attention and iteration < len(self._weights):
                w = self._weights[iteration]
                fused = (w[0] * top_down[i] + w[1] * downsampled) / (w.sum() + self.epsilon)
            else:
                fused = (top_down[i] + downsampled) / 2
            bottom_up.append(fused)

        outputs = []
        for i in range(num_levels):
            conv = self._convs[iteration][i]
            if conv is not None:
                out = conv(bottom_up[i])
            else:
                out = bottom_up[i]
            outputs.append(out.detach().cpu().numpy())

        return outputs

    def _bifpn_iteration_numpy(self, features: list[np.ndarray], iteration: int) -> list[np.ndarray]:
        def resize_nearest(x: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
            h, w = x.shape[-2:]
            th, tw = target_shape
            h_indices = np.clip((np.arange(th) * h / th).astype(int), 0, h - 1)
            w_indices = np.clip((np.arange(tw) * w / tw).astype(int), 0, w - 1)
            return x[..., h_indices[:, None], w_indices[None, :]]

        top_down = [features[-1]]
        for i in range(len(features) - 2, -1, -1):
            target_shape = features[i].shape[-2:]
            upsampled = resize_nearest(top_down[0], target_shape)
            if self.use_attention and iteration < len(self._weights):
                w = self._weights[iteration]
                fused = (w[0] * features[i] + w[1] * upsampled) / (w.sum() + self.epsilon)
            else:
                fused = (features[i] + upsampled) / 2
            top_down.insert(0, fused)

        bottom_up = [top_down[0]]
        for i in range(1, len(features)):
            downsampled = top_down[i][..., ::2, ::2]
            if downsampled.shape[-2:] != top_down[i].shape[-2:]:
                downsampled = resize_nearest(downsampled, top_down[i].shape[-2:])
            if self.use_attention and iteration < len(self._weights):
                w = self._weights[iteration]
                fused = (w[0] * top_down[i] + w[1] * downsampled) / (w.sum() + self.epsilon)
            else:
                fused = (top_down[i] + downsampled) / 2
            bottom_up.append(fused)

        return bottom_up


class PANet:
    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int = 256,
        num_bottom_up: int = 2,
        use_extra_convs: bool = False,
    ):
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_bottom_up = num_bottom_up
        self.use_extra_convs = use_extra_convs
        self._fpn = FeaturePyramidNetwork(
            min_level=PyramidLevel.P3,
            max_level=PyramidLevel.P7,
            num_channels=out_channels,
        )
        self._bifpn = BiFPN(
            num_levels=len(in_channels_list),
            num_channels=out_channels,
            num_iterations=num_bottom_up,
        )

    def forward(self, backbone_features: dict[PyramidLevel, np.ndarray]) -> PyramidFeatures:
        fpn_out = self._fpn.forward(backbone_features)
        levels_list = [fpn_out.levels[PyramidLevel(i)] for i in range(3, 3 + len(self.in_channels_list)) if PyramidLevel(i) in fpn_out.levels]
        bifpn_out = self._bifpn.forward(levels_list)
        bifpn_dict = {PyramidLevel(3 + i): bifpn_out[i] for i in range(len(bifpn_out))}
        return PyramidFeatures(
            levels=bifpn_dict,
            base_resolution=fpn_out.base_resolution,
            scale_factors=fpn_out.scale_factors,
        )


class NASFPN:
    def __init__(
        self,
        num_levels: int = 5,
        num_channels: int = 256,
        num_cells: int = 7,
    ):
        self.num_levels = num_levels
        self.num_channels = num_channels
        self.num_cells = num_cells
        self._cell_weights = []
        self._initialized = False

    def _initialize(self):
        self._cell_weights = np.random.randn(self.num_cells, self.num_levels, 2).astype(np.float32)
        self._initialized = True

    def forward(self, features: list[np.ndarray]) -> list[np.ndarray]:
        if not self._initialized:
            self._initialize()

        current = features
        for cell_idx in range(self.num_cells):
            current = self._nasfpn_cell(current, cell_idx)
        return current

    def _nasfpn_cell(self, features: list[np.ndarray], cell_idx: int) -> list[np.ndarray]:
        weights = self._cell_weights[cell_idx]
        num_levels = len(features)
        outputs = []

        for i in range(num_levels):
            w1, w2 = weights[i]
            w_sum = abs(w1) + abs(w2) + 1e-4
            w1, w2 = w1 / w_sum, w2 / w_sum

            if i == 0:
                resized = self._resize(features[i + 1], features[i].shape[-2:])
                out = w1 * features[i] + w2 * resized
            elif i == num_levels - 1:
                resized = self._resize(features[i - 1], features[i].shape[-2:])
                out = w1 * features[i] + w2 * resized
            else:
                up = self._resize(features[i + 1], features[i].shape[-2:])
                down = self._resize(features[i - 1], features[i].shape[-2:])
                out = w1 * features[i] + w2 * (up + down) / 2

            outputs.append(out)

        return outputs

    def _resize(self, src: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        h, w = src.shape[-2:]
        th, tw = target_shape
        if h == th and w == tw:
            return src
        h_indices = np.clip((np.arange(th) * h / th).astype(int), 0, h - 1)
        w_indices = np.clip((np.arange(tw) * w / tw).astype(int), 0, w - 1)
        return src[..., h_indices[:, None], w_indices[None, :]]


def build_resnet_backbone(depth: int = 50) -> Callable[[np.ndarray], dict[PyramidLevel, np.ndarray]]:
    def extractor(image: np.ndarray) -> dict[PyramidLevel, np.ndarray]:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        h, w = image.shape[:2]
        features = {}

        c2 = cv2.resize(image, (w // 4, h // 4)).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        c3 = cv2.resize(image, (w // 8, h // 8)).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        c4 = cv2.resize(image, (w // 16, h // 16)).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        c5 = cv2.resize(image, (w // 32, h // 32)).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        features[PyramidLevel.P2] = np.repeat(c2, 64, axis=1)
        features[PyramidLevel.P3] = np.repeat(c3, 128, axis=1)
        features[PyramidLevel.P4] = np.repeat(c4, 256, axis=1)
        features[PyramidLevel.P5] = np.repeat(c5, 512, axis=1)

        return features

    return extractor


def build_efficientnet_backbone(version: str = "b0") -> Callable[[np.ndarray], dict[PyramidLevel, np.ndarray]]:
    def extractor(image: np.ndarray) -> dict[PyramidLevel, np.ndarray]:
        h, w = image.shape[:2]
        features = {}

        scales = [4, 8, 16, 32]
        channels = [48, 80, 176, 512] if version == "b0" else [64, 112, 240, 672]

        for i, (scale, ch) in enumerate(zip(scales, channels)):
            level = PyramidLevel.P2 + i
            feat = cv2.resize(image, (w // scale, h // scale)).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            features[level] = np.repeat(feat, ch, axis=1)

        return features

    return extractor