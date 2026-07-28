from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class GroundTruthObject:
    object_id: int
    class_name: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SyntheticFrame:
    frame_index: int
    width: int
    height: int
    ground_truth: tuple[GroundTruthObject, ...]
    detections: tuple[dict[str, object], ...]
    occluded_ids: tuple[int, ...]


class SyntheticTrackingScenario:
    """Produce repeatable box-level camera sequences with known ground truth."""

    def __init__(
        self,
        *,
        frame_count: int = 60,
        object_count: int = 4,
        width: int = 1280,
        height: int = 720,
        seed: int = 0,
        occlusions: dict[int, Sequence[int]] | None = None,
        miss_probability: float = 0.0,
        false_positives_per_frame: int = 0,
        detection_noise_px: float = 0.0,
    ) -> None:
        if frame_count < 1 or object_count < 1 or width < 32 or height < 32:
            raise ValueError("frame count, object count, and image dimensions must be positive")
        if not 0 <= miss_probability <= 1:
            raise ValueError("miss probability must be between zero and one")
        if false_positives_per_frame < 0 or detection_noise_px < 0:
            raise ValueError("false-positive count and detection noise must be non-negative")

        self.frame_count = frame_count
        self.object_count = object_count
        self.width = width
        self.height = height
        self._random = np.random.default_rng(seed)
        self._occlusions = {frame: set(ids) for frame, ids in (occlusions or {}).items()}
        self._miss_probability = miss_probability
        self._false_positives_per_frame = false_positives_per_frame
        self._detection_noise_px = detection_noise_px
        self._objects = self._initialize_objects()

    def _initialize_objects(self) -> list[dict[str, object]]:
        objects = []
        for object_id in range(1, self.object_count + 1):
            width = float(self._random.integers(30, 91))
            height = float(self._random.integers(40, 121))
            x = float(self._random.uniform(0, self.width - width))
            y = float(self._random.uniform(0, self.height - height))
            velocity_x = float(self._random.uniform(-3, 3))
            velocity_y = float(self._random.uniform(-2, 2))
            objects.append(
                {
                    "id": object_id,
                    "class": "person" if object_id % 2 else "vehicle",
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "velocity_x": velocity_x,
                    "velocity_y": velocity_y,
                }
            )
        return objects

    def generate(self) -> tuple[SyntheticFrame, ...]:
        frames = []
        for frame_index in range(self.frame_count):
            truths = []
            detections = []
            occluded_ids = self._occlusions.get(frame_index, set())
            for item in self._objects:
                x = (item["x"] + item["velocity_x"] * frame_index) % (
                    self.width - item["width"]
                )
                y = (item["y"] + item["velocity_y"] * frame_index) % (
                    self.height - item["height"]
                )
                bbox = (x, y, x + item["width"], y + item["height"])
                truths.append(GroundTruthObject(item["id"], item["class"], bbox))
                if item["id"] in occluded_ids or self._random.random() < self._miss_probability:
                    continue

                noise = self._random.normal(0, self._detection_noise_px, size=4)
                noisy_bbox = [float(value) for value in (np.asarray(bbox) + noise)]
                noisy_bbox[0] = min(max(0.0, noisy_bbox[0]), self.width - 1.0)
                noisy_bbox[1] = min(max(0.0, noisy_bbox[1]), self.height - 1.0)
                noisy_bbox[2] = max(noisy_bbox[0] + 1.0, min(self.width, noisy_bbox[2]))
                noisy_bbox[3] = max(noisy_bbox[1] + 1.0, min(self.height, noisy_bbox[3]))
                detections.append(
                    {
                        "class": item["class"],
                        "confidence": float(self._random.uniform(0.75, 0.99)),
                        "bbox": noisy_bbox,
                    }
                )

            for _ in range(self._false_positives_per_frame):
                x1 = float(self._random.uniform(0, self.width - 20))
                y1 = float(self._random.uniform(0, self.height - 20))
                box_width = float(self._random.uniform(10, 50))
                box_height = float(self._random.uniform(10, 60))
                detections.append(
                    {
                        "class": "unknown",
                        "confidence": float(self._random.uniform(0.1, 0.6)),
                        "bbox": [x1, y1, min(self.width, x1 + box_width), min(self.height, y1 + box_height)],
                    }
                )

            frames.append(
                SyntheticFrame(
                    frame_index=frame_index,
                    width=self.width,
                    height=self.height,
                    ground_truth=tuple(truths),
                    detections=tuple(detections),
                    occluded_ids=tuple(sorted(occluded_ids)),
                )
            )
        return tuple(frames)