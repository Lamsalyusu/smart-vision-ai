import math
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AppearanceEmbedding:
    """L2-normalized visual feature vector representing object appearance."""

    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.vector:
            raise ValueError("appearance embedding vector must not be empty")
        if not all(math.isfinite(x) for x in self.vector):
            raise ValueError("all embedding vector components must be finite")
        norm_sq = sum(x * x for x in self.vector)
        if norm_sq <= 1e-12:
            raise ValueError("embedding norm is approximately zero")

    @classmethod
    def from_raw(cls, values: Sequence[float]) -> "AppearanceEmbedding":
        norm = math.sqrt(sum(v * v for v in values))
        if norm <= 1e-12:
            raise ValueError("cannot normalize zero vector")
        normalized = tuple(float(v / norm) for v in values)
        return cls(vector=normalized)

    def cosine_distance(self, other: "AppearanceEmbedding") -> float:
        """Compute cosine distance in range [0, 2], where 0 is identical."""
        if len(self.vector) != len(other.vector):
            raise ValueError(
                f"dimension mismatch: {len(self.vector)} vs {len(other.vector)}"
            )
        dot_product = sum(a * b for a, b in zip(self.vector, other.vector, strict=True))
        # Clamp to [-1.0, 1.0] to prevent floating point imprecision
        dot_product = max(-1.0, min(1.0, dot_product))
        return 1.0 - dot_product


@dataclass(slots=True)
class FeatureGallery:
    """Rolling appearance feature gallery with Exponential Moving Average (EMA)."""

    max_history: int = 16
    momentum: float = 0.85
    _history: list[AppearanceEmbedding] = field(default_factory=list)
    _mean_embedding: AppearanceEmbedding | None = None

    def __post_init__(self) -> None:
        if self.max_history <= 0:
            raise ValueError("max_history must be strictly positive")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError("momentum must be in [0.0, 1.0)")

    @property
    def current_embedding(self) -> AppearanceEmbedding | None:
        return self._mean_embedding

    def update(self, new_emb: AppearanceEmbedding) -> None:
        if self._mean_embedding is None:
            self._mean_embedding = new_emb
            self._history.append(new_emb)
            return

        dim = len(new_emb.vector)
        if len(self._mean_embedding.vector) != dim:
            raise ValueError("dimension mismatch during gallery update")

        # EMA update: ema = momentum * old + (1 - momentum) * new
        blended = [
            self.momentum * old + (1.0 - self.momentum) * new
            for old, new in zip(self._mean_embedding.vector, new_emb.vector, strict=True)
        ]
        self._mean_embedding = AppearanceEmbedding.from_raw(blended)

        self._history.append(new_emb)
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def distance_to(self, candidate: AppearanceEmbedding) -> float:
        if self._mean_embedding is None:
            return 1.0
        return self._mean_embedding.cosine_distance(candidate)


def compute_reid_cost_matrix(
    galleries: Sequence[FeatureGallery],
    candidates: Sequence[AppearanceEmbedding],
) -> list[list[float]]:
    """Generate pairwise cosine distance cost matrix [num_tracks x num_candidates]."""
    num_tracks = len(galleries)
    num_cand = len(candidates)
    matrix: list[list[float]] = [[0.0] * num_cand for _ in range(num_tracks)]

    for t_idx, gallery in enumerate(galleries):
        for c_idx, candidate in enumerate(candidates):
            matrix[t_idx][c_idx] = gallery.distance_to(candidate)

    return matrix


def composite_association_cost(
    spatial_cost: float,
    appearance_cost: float,
    weight_spatial: float = 0.5,
    spatial_gate: float = 16.0,
    appearance_gate: float = 0.5,
) -> float:
    """Fuse spatial (Mahalanobis^2) and appearance (Cosine) costs with gating.

    If either cost exceeds its gating threshold, returns math.inf.
    """
    if spatial_cost > spatial_gate or appearance_cost > appearance_gate:
        return math.inf

    # Normalize spatial cost to approximately [0, 1] relative to gate
    norm_spatial = min(1.0, spatial_cost / max(spatial_gate, 1e-6))
    # Appearance cost is already in [0, 1] for typical cosine distances
    norm_appearance = min(1.0, appearance_cost / max(appearance_gate, 1e-6))

    return weight_spatial * norm_spatial + (1.0 - weight_spatial) * norm_appearance
