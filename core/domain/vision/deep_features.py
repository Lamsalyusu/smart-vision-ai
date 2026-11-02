from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from abc import ABC, abstractmethod


@dataclass(frozen=True, slots=True)
class FeatureVector:
    data: np.ndarray
    dimensionality: int
    normalization: str = "l2"

    def __post_init__(self):
        if self.data.shape[-1] != self.dimensionality:
            raise ValueError(f"Data dimension {self.data.shape[-1]} != {self.dimensionality}")

    @property
    def normalized(self) -> FeatureVector:
        if self.normalization == "l2":
            norm = np.linalg.norm(self.data, axis=-1, keepdims=True)
            norm = np.where(norm > 0, norm, 1.0)
            return FeatureVector(self.data / norm, self.dimensionality, self.normalization)
        return self

    def cosine_similarity(self, other: FeatureVector) -> float:
        a = self.normalized.data.flatten()
        b = other.normalized.data.flatten()
        return float(np.dot(a, b))

    def euclidean_distance(self, other: FeatureVector) -> float:
        return float(np.linalg.norm(self.data.flatten() - other.data.flatten()))


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image: np.ndarray, regions: Optional[Sequence[np.ndarray]] = None) -> Sequence[FeatureVector]:
        pass

    @abstractmethod
    def extract_single(self, image: np.ndarray) -> FeatureVector:
        pass


class DeepFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        model_name: str = "resnet50",
        feature_layer: str = "avgpool",
        input_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.feature_layer = feature_layer
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._preprocess_fn = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            model_map = {
                "resnet18": models.resnet18,
                "resnet34": models.resnet34,
                "resnet50": models.resnet50,
                "resnet101": models.resnet101,
                "resnet152": models.resnet152,
                "efficientnet_b0": models.efficientnet_b0,
                "efficientnet_b1": models.efficientnet_b1,
                "mobilenet_v3_large": models.mobilenet_v3_large,
                "mobilenet_v3_small": models.mobilenet_v3_small,
            }

            if self.model_name not in model_map:
                raise ValueError(f"Unsupported model: {self.model_name}")

            model_fn = model_map[self.model_name]
            self._model = model_fn(pretrained=True)
            self._model.eval()

            if hasattr(self._model, feature_layer):
                self._target_layer = getattr(self._model, feature_layer)
            else:
                self._target_layer = self._model.avgpool

            self._preprocess_fn = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self._hook_features = None

            def hook_fn(module, input, output):
                self._hook_features = output.detach()

            self._hook = self._target_layer.register_forward_hook(hook_fn)

            if self.device != "cpu" and torch.cuda.is_available():
                self._model = self._model.cuda()

        except ImportError:
            self._model = None
            self._preprocess_fn = None

    def extract(self, image: np.ndarray, regions: Optional[Sequence[np.ndarray]] = None) -> Sequence[FeatureVector]:
        if regions is None:
            return [self.extract_single(image)]

        crops = []
        for region in regions:
            x1, y1, x2, y2 = map(int, region)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

        if not crops:
            return []

        return [self.extract_single(crop) for crop in crops]

    def extract_single(self, image: np.ndarray) -> FeatureVector:
        if self._model is None:
            return self._fallback_extract(image)

        import torch

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        tensor = self._preprocess_fn(image).unsqueeze(0)

        if self.device != "cpu" and torch.cuda.is_available():
            tensor = tensor.cuda()

        with torch.no_grad():
            _ = self._model(tensor)

        if self._hook_features is not None:
            features = self._hook_features.cpu().numpy().flatten()
        else:
            features = self._fallback_extract(image).data

        dim = features.shape[0]
        return FeatureVector(features, dim)

    def _fallback_extract(self, image: np.ndarray) -> FeatureVector:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        gray = cv2.resize(gray, (64, 64))

        hog_features = self._compute_hog(gray)
        lbp_features = self._compute_lbp(gray)
        color_hist = self._compute_color_histogram(image) if len(image.shape) == 3 else np.array([])

        combined = np.concatenate([hog_features, lbp_features, color_hist])
        return FeatureVector(combined, combined.shape[0])

    def _compute_hog(self, image: np.ndarray, cell_size: int = 8, bins: int = 9) -> np.ndarray:
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        h, w = image.shape
        n_cells_y, n_cells_x = h // cell_size, w // cell_size
        hog = np.zeros((n_cells_y, n_cells_x, bins))

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_mag = mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                cell_ang = ang[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                hist, _ = np.histogram(cell_ang, bins=bins, range=(0, 180), weights=cell_mag)
                hog[i, j] = hist

        return hog.flatten()

    def _compute_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        h, w = image.shape
        lbp = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint8)

        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = int(round(radius * math.cos(angle)))
            y = int(round(-radius * math.sin(angle)))
            neighbor = image[radius+y:h-radius+y, radius+x:w-radius+x]
            center = image[radius:h-radius, radius:w-radius]
            lbp |= ((neighbor >= center) << i)

        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
        return hist.astype(np.float32) / (hist.sum() + 1e-7)

    def _compute_color_histogram(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        hist = []
        for c in range(3):
            h, _ = np.histogram(image[:, :, c].flatten(), bins=bins, range=(0, 256))
            hist.append(h.astype(np.float32) / (h.sum() + 1e-7))
        return np.concatenate(hist)

    def __del__(self):
        if hasattr(self, '_hook') and self._hook is not None:
            self._hook.remove()


class AttentionFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        patch_size: int = 16,
        image_size: int = 224,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            import torch.nn as nn

            class PatchEmbedding(nn.Module):
                def __init__(self, in_channels=3, patch_size=16, embed_dim=256):
                    super().__init__()
                    self.patch_size = patch_size
                    self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

                def forward(self, x):
                    x = self.proj(x)
                    x = x.flatten(2).transpose(1, 2)
                    return x

            class TransformerBlock(nn.Module):
                def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(embed_dim)
                    self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
                    self.norm2 = nn.LayerNorm(embed_dim)
                    self.mlp = nn.Sequential(
                        nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                        nn.Dropout(dropout),
                    )

                def forward(self, x):
                    x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
                    x = x + self.mlp(self.norm2(x))
                    return x

            class VisionTransformer(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.patch_embed = PatchEmbedding(3, config.patch_size, config.embed_dim)
                    self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches + 1, config.embed_dim))
                    self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
                    self.blocks = nn.ModuleList([
                        TransformerBlock(config.embed_dim, config.num_heads) for _ in range(config.num_layers)
                    ])
                    self.norm = nn.LayerNorm(config.embed_dim)

                def forward(self, x):
                    B = x.shape[0]
                    x = self.patch_embed(x)
                    cls_tokens = self.cls_token.expand(B, -1, -1)
                    x = torch.cat([cls_tokens, x], dim=1)
                    x = x + self.pos_embed
                    for block in self.blocks:
                        x = block(x)
                    x = self.norm(x)
                    return x[:, 0]

            from types import SimpleNamespace
            config = SimpleNamespace(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                patch_size=self.patch_size,
                num_patches=self.num_patches,
            )
            self._model = VisionTransformer(config)
            self._model.eval()
            self._initialized = True

        except ImportError:
            self._initialized = False

    def extract(self, image: np.ndarray, regions: Optional[Sequence[np.ndarray]] = None) -> Sequence[FeatureVector]:
        if not self._initialized:
            self._initialize()

        if regions is None:
            return [self.extract_single(image)]

        return [self.extract_single(crop) for crop in regions if crop.size > 0]

    def extract_single(self, image: np.ndarray) -> FeatureVector:
        if not self._initialized:
            return self._fallback_vit(image)

        import torch
        import torchvision.transforms as transforms

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            features = self._model(tensor).cpu().numpy().flatten()

        return FeatureVector(features, self.embed_dim)

    def _fallback_vit(self, image: np.ndarray) -> FeatureVector:
        extractor = DeepFeatureExtractor("resnet50", "avgpool", (224, 224))
        return extractor.extract_single(image)


class MultiScaleFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        scales: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
        base_extractor: Optional[FeatureExtractor] = None,
    ):
        self.scales = scales
        self.base_extractor = base_extractor or DeepFeatureExtractor()

    def extract(self, image: np.ndarray, regions: Optional[Sequence[np.ndarray]] = None) -> Sequence[FeatureVector]:
        if regions is None:
            return [self.extract_single(image)]

        results = []
        for region in regions:
            x1, y1, x2, y2 = map(int, region)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                results.append(self.extract_single(crop))
        return results

    def extract_single(self, image: np.ndarray) -> FeatureVector:
        features = []
        h, w = image.shape[:2]

        for scale in self.scales:
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h < 32 or new_w < 32:
                continue

            scaled = cv2.resize(image, (new_w, new_h))
            feat = self.base_extractor.extract_single(scaled)
            features.append(feat.data)

        if not features:
            return self.base_extractor.extract_single(image)

        combined = np.concatenate(features)
        return FeatureVector(combined, combined.shape[0])


class FeatureAggregator:
    def __init__(self, method: str = "mean"):
        self.method = method

    def aggregate(self, features: Sequence[FeatureVector]) -> FeatureVector:
        if not features:
            raise ValueError("No features to aggregate")

        dim = features[0].dimensionality
        data = np.stack([f.data for f in features])

        if self.method == "mean":
            agg = np.mean(data, axis=0)
        elif self.method == "max":
            agg = np.max(data, axis=0)
        elif self.method == "min":
            agg = np.min(data, axis=0)
        elif self.method == "concat":
            agg = data.flatten()
            dim = agg.shape[0]
        elif self.method == "pca":
            agg = self._pca_aggregate(data)
            dim = agg.shape[0]
        else:
            agg = np.mean(data, axis=0)

        return FeatureVector(agg, dim)

    def _pca_aggregate(self, data: np.ndarray, n_components: int = 128) -> np.ndarray:
        from sklearn.decomposition import PCA

        n_samples, n_features = data.shape
        n_components = min(n_components, n_samples, n_features)

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        return np.mean(transformed, axis=0)


class FeatureMatcher:
    def __init__(
        self,
        threshold: float = 0.7,
        metric: str = "cosine",
        top_k: int = 5,
    ):
        self.threshold = threshold
        self.metric = metric
        self.top_k = top_k
        self._index = None
        self._index_data = None

    def build_index(self, features: Sequence[FeatureVector]):
        self._index_data = features
        self._index = np.stack([f.normalized.data.flatten() for f in features])

    def match(self, query: FeatureVector) -> list[tuple[int, float]]:
        if self._index is None:
            raise ValueError("Index not built")

        query_norm = query.normalized.data.flatten()

        if self.metric == "cosine":
            scores = self._index @ query_norm
        elif self.metric == "euclidean":
            scores = -np.linalg.norm(self._index - query_norm, axis=1)
        else:
            scores = self._index @ query_norm

        top_indices = np.argsort(scores)[-self.top_k:][::-1]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= self.threshold:
                results.append((int(idx), score))
        return results

    def match_batch(self, queries: Sequence[FeatureVector]) -> list[list[tuple[int, float]]]:
        return [self.match(q) for q in queries]


class FeatureBank:
    def __init__(self, max_size: int = 10000, feature_dim: int = 512):
        self.max_size = max_size
        self.feature_dim = feature_dim
        self._features: list[FeatureVector] = []
        self._labels: list[Any] = []
        self._timestamps: list[float] = []
        self._access_counts: list[int] = []

    def add(self, feature: FeatureVector, label: Any, timestamp: float = 0.0) -> int:
        if len(self._features) >= self.max_size:
            self._evict()

        self._features.append(feature)
        self._labels.append(label)
        self._timestamps.append(timestamp)
        self._access_counts.append(0)
        return len(self._features) - 1

    def _evict(self):
        scores = np.array(self._access_counts) / (np.array(self._timestamps) + 1e-6)
        evict_idx = int(np.argmin(scores))
        self._features.pop(evict_idx)
        self._labels.pop(evict_idx)
        self._timestamps.pop(evict_idx)
        self._access_counts.pop(evict_idx)

    def get(self, index: int) -> tuple[FeatureVector, Any]:
        self._access_counts[index] += 1
        return self._features[index], self._labels[index]

    def search(self, query: FeatureVector, top_k: int = 10, threshold: float = 0.5) -> list[tuple[int, float, Any]]:
        if not self._features:
            return []

        matcher = FeatureMatcher(threshold=threshold, top_k=top_k)
        matcher.build_index(self._features)
        matches = matcher.match(query)
        return [(idx, score, self._labels[idx]) for idx, score in matches]

    def update(self, index: int, feature: FeatureVector, label: Any = None):
        if 0 <= index < len(self._features):
            self._features[index] = feature
            if label is not None:
                self._labels[index] = label

    def __len__(self) -> int:
        return len(self._features)

    def clear(self):
        self._features.clear()
        self._labels.clear()
        self._timestamps.clear()
        self._access_counts.clear()


import cv2
try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None