import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("bounding-box coordinates must be finite")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("bounding box must have positive width and height")


@dataclass(frozen=True, slots=True)
class Confidence:
    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or not 0.0 <= self.value <= 1.0:
            raise ValueError("confidence must be a finite value between 0 and 1")


@dataclass(frozen=True, slots=True)
class CameraId:
    value: str

    def __post_init__(self) -> None:
        if not self.value or self.value.strip() != self.value:
            raise ValueError("camera id must be non-empty and have no surrounding whitespace")


@dataclass(frozen=True, slots=True)
class FrameSequence:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("frame sequence must be non-negative")