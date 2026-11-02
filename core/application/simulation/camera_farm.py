"""Deterministic synthetic camera frames for load, chaos, and regression tests."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SyntheticDetection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SyntheticCameraFrame:
    camera_id: str
    sequence: int
    captured_at_seconds: float
    jpeg_bytes: bytes
    detections: tuple[SyntheticDetection, ...]


class SyntheticCameraFarm:
    """Generate reproducible JPEG frames with ground-truth moving objects."""

    def __init__(
        self,
        camera_ids: tuple[str, ...] | list[str],
        *,
        width: int = 320,
        height: int = 240,
        jpeg_quality: int = 70,
    ) -> None:
        cameras = tuple(camera_ids)
        if not cameras:
            raise ValueError("at least one camera id is required")
        if len(set(cameras)) != len(cameras) or any(not camera_id for camera_id in cameras):
            raise ValueError("camera ids must be unique and non-empty")
        if width < 32 or height < 32:
            raise ValueError("synthetic frames must be at least 32x32")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")

        self.camera_ids = cameras
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self._background = self._make_background()
        self._phases = {
            camera_id: (index * 37) % max(1, width - 40)
            for index, camera_id in enumerate(cameras)
        }

    def _make_background(self) -> np.ndarray:
        rows = np.linspace(28, 64, self.height, dtype=np.uint8)[:, None]
        columns = np.linspace(18, 44, self.width, dtype=np.uint8)[None, :]
        base = ((rows.astype(np.int16) + columns.astype(np.int16)) // 2).astype(np.uint8)
        background = np.dstack([base, base // 2, base // 3])
        return np.ascontiguousarray(background)

    def frame(
        self,
        camera_id: str,
        sequence: int,
        captured_at_seconds: float | None = None,
    ) -> SyntheticCameraFrame:
        if camera_id not in self._phases:
            raise ValueError(f"unknown camera id: {camera_id}")
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        if captured_at_seconds is None:
            captured_at_seconds = sequence / 30.0
        if not math.isfinite(captured_at_seconds) or captured_at_seconds < 0:
            raise ValueError("capture timestamp must be finite and non-negative")

        image = self._background.copy()
        phase = self._phases[camera_id]
        box_width = max(8, min(36, self.width // 3))
        box_height = max(8, min(72, self.height - 12))
        x1 = int((phase + sequence * 5) % max(1, self.width - box_width))
        y1 = int(self.height // 2 - box_height // 2 + math.sin(sequence / 6.0) * 12)
        y1 = max(0, min(y1, self.height - box_height))
        x2, y2 = x1 + box_width, y1 + box_height
        cv2.rectangle(image, (x1, y1), (x2, y2), (80, 190, 120), thickness=-1)
        cv2.putText(
            image,
            camera_id[:12],
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        success, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not success:
            raise RuntimeError("synthetic frame encoding failed")

        detection = SyntheticDetection(
            class_name="person",
            confidence=0.99,
            bbox=(float(x1), float(y1), float(x2), float(y2)),
        )
        return SyntheticCameraFrame(
            camera_id=camera_id,
            sequence=sequence,
            captured_at_seconds=captured_at_seconds,
            jpeg_bytes=encoded.tobytes(),
            detections=(detection,),
        )

    def frames(
        self,
        camera_id: str,
        frame_count: int,
        *,
        start_seconds: float = 0.0,
        fps: float = 30.0,
    ) -> Iterator[SyntheticCameraFrame]:
        if frame_count < 0:
            raise ValueError("frame count must be non-negative")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        for sequence in range(frame_count):
            yield self.frame(
                camera_id,
                sequence,
                start_seconds + sequence / fps,
            )
