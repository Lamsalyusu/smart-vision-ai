from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import math
import time


class NeRFType(Enum):
    ORIGINAL = "original"
    MIP_NERF = "mip_nerf"
    MIP_NERF_360 = "mip_nerf_360"
    INSTANT_NGP = "instant_ngp"
    TENSORF = "tensorf"
    KILO_NERF = "kilo_nerf"
    BARF = "barf"
    PIXEL_NERF = "pixel_nerf"
    META_NERF = "meta_nerf"
    GENERALIZABLE = "generalizable"


class EncodingType(Enum):
    POSITIONAL = "positional"
    HASH_GRID = "hash_grid"
    TRILINEAR = "trilinear"
    FOURIER = "fourier"
    SPHERICAL_HARMONICS = "spherical_harmonics"
    ONE_BLOB = "one_blob"


@dataclass(frozen=True, slots=True)
class NeRFConfig:
    nerf_type: NeRFType = NeRFType.ORIGINAL
    encoding_type: EncodingType = EncodingType.POSITIONAL
    num_layers: int = 8
    hidden_dim: int = 256
    skip_connections: List[int] = field(default_factory=lambda: [4])
    num_freqs_pos: int = 10
    num_freqs_dir: int = 4
    use_viewdirs: bool = True
    use_appearance_embedding: bool = False
    appearance_embedding_dim: int = 32
    near: float = 0.1
    far: float = 100.0
    num_samples: int = 64
    num_importance_samples: int = 64
    perturb: bool = True
    raw_noise_std: float = 0.0
    white_bkgd: bool = False


@dataclass(frozen=True, slots=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray
    near: float = 0.1
    far: float = 100.0


@dataclass(frozen=True, slots=True)
class RaySamples:
    t_vals: np.ndarray
    points: np.ndarray
    directions: np.ndarray
    weights: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class RenderOutput:
    rgb: np.ndarray
    depth: np.ndarray
    acc: np.ndarray
    weights: Optional[np.ndarray] = None


class PositionalEncoder:
    def __init__(self, num_freqs: int, include_input: bool = True):
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** np.arange(num_freqs)

    def encode(self, x: np.ndarray) -> np.ndarray:
        encoded = []
        if self.include_input:
            encoded.append(x)

        for freq in self.freq_bands:
            encoded.append(np.sin(x * freq))
            encoded.append(np.cos(x * freq))

        return np.concatenate(encoded, axis=-1)

    def output_dim(self, input_dim: int) -> int:
        return input_dim * (2 * self.num_freqs + (1 if self.include_input else 0))


class HashGridEncoder:
    def __init__(
        self,
        num_levels: int = 16,
        level_dim: int = 2,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        log2_hashmap_size: int = 19
    ):
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 1 << log2_hashmap_size

        self.growth_factor = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (num_levels - 1)
        )

        self.embeddings = [
            np.random.randn(self.hashmap_size, level_dim) * 1e-4
            for _ in range(num_levels)
        ]

    def _hash(self, coords: np.ndarray) -> np.ndarray:
        primes = np.array([1, 2654435761, 805459861], dtype=np.uint64)
        return np.bitwise_xor.reduce(
            [coords[..., i].astype(np.uint64) * primes[i] for i in range(coords.shape[-1])],
            axis=0
        ) % self.hashmap_size

    def encode(self, x: np.ndarray) -> np.ndarray:
        encoded = []
        for level in range(self.num_levels):
            resolution = min(
                int(self.base_resolution * (self.growth_factor ** level)),
                self.max_resolution
            )

            coords = (x * resolution).astype(np.int32)
            coords = np.clip(coords, 0, resolution - 1)

            if x.shape[-1] == 3:
                c000 = self._hash(coords[..., [0, 1, 2]])
                c001 = self._hash(coords[..., [0, 1, 2]] + np.array([0, 0, 1]))
                c010 = self._hash(coords[..., [0, 1, 2]] + np.array([0, 1, 0]))
                c011 = self._hash(coords[..., [0, 1, 2]] + np.array([0, 1, 1]))
                c100 = self._hash(coords[..., [0, 1, 2]] + np.array([1, 0, 0]))
                c101 = self._hash(coords[..., [0, 1, 2]] + np.array([1, 0, 1]))
                c110 = self._hash(coords[..., [0, 1, 2]] + np.array([1, 1, 0]))
                c111 = self._hash(coords[..., [0, 1, 2]] + np.array([1, 1, 1]))

                weights = (x * resolution) - coords
                wx, wy, wz = weights[..., 0:1], weights[..., 1:2], weights[..., 2:3]

                e000 = self.embeddings[level][c000]
                e001 = self.embeddings[level][c001]
                e010 = self.embeddings[level][c010]
                e011 = self.embeddings[level][c011]
                e100 = self.embeddings[level][c100]
                e101 = self.embeddings[level][c101]
                e110 = self.embeddings[level][c110]
                e111 = self.embeddings[level][c111]

                e00 = e000 * (1 - wx) + e100 * wx
                e01 = e010 * (1 - wx) + e110 * wx
                e10 = e001 * (1 - wx) + e101 * wx
                e11 = e011 * (1 - wx) + e111 * wx

                e0 = e00 * (1 - wy) + e01 * wy
                e1 = e10 * (1 - wy) + e11 * wy

                encoded.append(e0 * (1 - wz) + e1 * wz)

        return np.concatenate(encoded, axis=-1)


class NeRFMLP:
    def __init__(self, config: NeRFConfig):
        self.config = config

        pos_encoder = PositionalEncoder(config.num_freqs_pos)
        dir_encoder = PositionalEncoder(config.num_freqs_dir)

        self.pos_encoder = pos_encoder
        self.dir_encoder = dir_encoder

        pos_dim = pos_encoder.output_dim(3)
        dir_dim = dir_encoder.output_dim(3) if config.use_viewdirs else 0

        self.layers = []
        for i in range(config.num_layers):
            if i == 0:
                in_dim = pos_dim
            elif i in config.skip_connections:
                in_dim = config.hidden_dim + pos_dim
            else:
                in_dim = config.hidden_dim

            out_dim = config.hidden_dim
            if i == config.num_layers - 1:
                out_dim = 1 + (config.hidden_dim if config.use_viewdirs else 3)

            W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim)
            b = np.zeros(out_dim)
            self.layers.append((W, b))

        if config.use_viewdirs:
            self.rgb_layers = []
            rgb_in = config.hidden_dim + dir_dim
            W = np.random.randn(rgb_in, config.hidden_dim // 2) * np.sqrt(2 / rgb_in)
            b = np.zeros(config.hidden_dim // 2)
            self.rgb_layers.append((W, b))

            W = np.random.randn(config.hidden_dim // 2, 3) * np.sqrt(2 / (config.hidden_dim // 2))
            b = np.zeros(3)
            self.rgb_layers.append((W, b))

    def forward(
        self,
        points: np.ndarray,
        directions: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        x = self.pos_encoder.encode(points)

        for i, (W, b) in enumerate(self.layers):
            if i in self.config.skip_connections:
                x = np.concatenate([x, self.pos_encoder.encode(points)], axis=-1)

            x = x @ W + b

            if i < self.config.num_layers - 1:
                x = np.maximum(x, 0)

        sigma = x[..., 0:1]

        if self.config.use_viewdirs and directions is not None:
            feat = x[..., 1:]
            dir_enc = self.dir_encoder.encode(directions)
            x = np.concatenate([feat, dir_enc], axis=-1)

            for i, (W, b) in enumerate(self.rgb_layers):
                x = x @ W + b
                if i < len(self.rgb_layers) - 1:
                    x = np.maximum(x, 0)

            rgb = 1 / (1 + np.exp(-x))
        else:
            rgb = 1 / (1 + np.exp(-x[..., 1:4])) if x.shape[-1] >= 4 else np.zeros((*x.shape[:-1], 3))

        return sigma, rgb


class NeRFRenderer:
    def __init__(self, config: NeRFConfig, model: NeRFMLP):
        self.config = config
        self.model = model

    def sample_along_rays(self, rays: List[Ray]) -> RaySamples:
        num_rays = len(rays)
        origins = np.array([r.origin for r in rays])
        directions = np.array([r.direction for r in rays])

        t_vals = np.linspace(self.config.near, self.config.far, self.config.num_samples)
        t_vals = t_vals[None, :].repeat(num_rays, axis=0)

        if self.config.perturb:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = np.concatenate([mids, t_vals[..., -1:]], axis=-1)
            lower = np.concatenate([t_vals[..., :1], mids], axis=-1)
            t_vals = lower + (upper - lower) * np.random.rand(*t_vals.shape)

        points = origins[:, None, :] + directions[:, None, :] * t_vals[..., None]

        return RaySamples(
            t_vals=t_vals,
            points=points,
            directions=directions[:, None, :].repeat(self.config.num_samples, axis=1)
        )

    def importance_sample(
        self,
        rays: List[Ray],
        weights: np.ndarray,
        t_vals: np.ndarray
    ) -> np.ndarray:
        weights = weights + 1e-5
        cdf = np.cumsum(weights, axis=-1)
        cdf = np.concatenate([np.zeros_like(cdf[..., :1]), cdf], axis=-1)
        cdf = cdf / cdf[..., -1:]

        u = np.random.rand(*weights.shape[:-1], self.config.num_importance_samples)
        u = u[..., None] if u.ndim == cdf.ndim - 1 else u

        inds = np.searchsorted(cdf, u, side='right')
        below = np.maximum(0, inds - 1)
        above = np.minimum(cdf.shape[-1] - 1, inds)

        cdf_below = np.take_along_axis(cdf, below, axis=-1)
        cdf_above = np.take_along_axis(cdf, above, axis=-1)
        t_below = np.take_along_axis(t_vals, below, axis=-1)
        t_above = np.take_along_axis(t_vals, above, axis=-1)

        denom = cdf_above - cdf_below
        denom = np.where(denom < 1e-5, 1, denom)
        t = t_below + (u - cdf_below) / denom * (t_above - t_below)

        return t

    def render_rays(self, rays: List[Ray]) -> RenderOutput:
        samples = self.sample_along_rays(rays)

        points_flat = samples.points.reshape(-1, 3)
        dirs_flat = samples.directions.reshape(-1, 3)

        sigma, rgb = self.model.forward(points_flat, dirs_flat)

        sigma = sigma.reshape(samples.points.shape[:-1])
        rgb = rgb.reshape(samples.points.shape[:-1] + (3,))

        deltas = np.concatenate([
            samples.t_vals[..., 1:] - samples.t_vals[..., :-1],
            np.full_like(samples.t_vals[..., :1], 1e10)
        ], axis=-1)

        alpha = 1 - np.exp(-sigma * deltas)
        weights = alpha * np.cumprod(
            np.concatenate([np.ones_like(alpha[..., :1]), 1 - alpha + 1e-10], axis=-1),
            axis=-1
        )[..., :-1]

        rgb_map = np.sum(weights[..., None] * rgb, axis=-2)
        depth_map = np.sum(weights * samples.t_vals, axis=-1)
        acc_map = np.sum(weights, axis=-1)

        if self.config.white_bkgd:
            rgb_map = rgb_map + (1 - acc_map[..., None])

        return RenderOutput(
            rgb=rgb_map,
            depth=depth_map,
            acc=acc_map,
            weights=weights
        )


class NeRFTrainer:
    def __init__(
        self,
        model: NeRFMLP,
        renderer: NeRFRenderer,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0
    ):
        self.model = model
        self.renderer = renderer
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.step = 0

    def train_step(
        self,
        rays: List[Ray],
        target_rgb: np.ndarray
    ) -> Dict[str, float]:
        self.step += 1

        output = self.renderer.render_rays(rays)

        loss = np.mean((output.rgb - target_rgb) ** 2)

        return {"loss": float(loss), "psnr": -10 * math.log10(loss + 1e-8)}

    def validate(
        self,
        rays: List[Ray],
        target_rgb: np.ndarray
    ) -> Dict[str, float]:
        output = self.renderer.render_rays(rays)
        loss = np.mean((output.rgb - target_rgb) ** 2)
        return {"val_loss": float(loss), "val_psnr": -10 * math.log10(loss + 1e-8)}


class NeRFDataset:
    def __init__(
        self,
        images: np.ndarray,
        poses: np.ndarray,
        focal: float,
        near: float = 0.1,
        far: float = 100.0
    ):
        self.images = images
        self.poses = poses
        self.focal = focal
        self.near = near
        self.far = far
        self.height, self.width = images.shape[1:3]

    def get_rays(self, image_idx: int) -> List[Ray]:
        pose = self.poses[image_idx]
        rays = []

        for i in range(self.height):
            for j in range(self.width):
                dir_vec = np.array([
                    (j - self.width * 0.5) / self.focal,
                    -(i - self.height * 0.5) / self.focal,
                    -1.0
                ])
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                dir_vec = pose[:3, :3] @ dir_vec

                rays.append(Ray(
                    origin=pose[:3, 3],
                    direction=dir_vec,
                    near=self.near,
                    far=self.far
                ))

        return rays

    def get_batch(self, batch_size: int) -> Tuple[List[Ray], np.ndarray]:
        all_rays = []
        all_pixels = []

        for i in range(len(self.images)):
            rays = self.get_rays(i)
            pixels = self.images[i].reshape(-1, 3)
            all_rays.extend(rays)
            all_pixels.append(pixels)

        all_pixels = np.concatenate(all_pixels, axis=0)

        indices = np.random.choice(len(all_rays), batch_size, replace=False)
        batch_rays = [all_rays[i] for i in indices]
        batch_pixels = all_pixels[indices]

        return batch_rays, batch_pixels


class InstantNGP:
    def __init__(
        self,
        config: NeRFConfig,
        grid_resolution: int = 128,
        num_levels: int = 16
    ):
        self.config = config
        self.hash_encoder = HashGridEncoder(
            num_levels=num_levels,
            level_dim=2,
            base_resolution=16,
            max_resolution=grid_resolution * 2**num_levels
        )

        self.mlp = self._build_mlp()

    def _build_mlp(self):
        pass

    def forward(self, points: np.ndarray, directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        encoded = self.hash_encoder.encode(points)
        return np.zeros((len(points), 1)), np.zeros((len(points), 3))


class NeRFMetrics:
    @staticmethod
    def psnr(pred: np.ndarray, target: np.ndarray) -> float:
        mse = np.mean((pred - target) ** 2)
        return -10 * math.log10(mse + 1e-8)

    @staticmethod
    def ssim(pred: np.ndarray, target: np.ndarray) -> float:
        return 0.0

    @staticmethod
    def lpips(pred: np.ndarray, target: np.ndarray) -> float:
        return 0.0


class NeRFVisualizer:
    @staticmethod
    def render_novel_view(
        model: NeRFMLP,
        renderer: NeRFRenderer,
        pose: np.ndarray,
        focal: float,
        height: int,
        width: int
    ) -> np.ndarray:
        rays = []
        for i in range(height):
            for j in range(width):
                dir_vec = np.array([
                    (j - width * 0.5) / focal,
                    -(i - height * 0.5) / focal,
                    -1.0
                ])
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                dir_vec = pose[:3, :3] @ dir_vec

                rays.append(Ray(origin=pose[:3, 3], direction=dir_vec))

        output = renderer.render_rays(rays)
        return output.rgb.reshape(height, width, 3)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet