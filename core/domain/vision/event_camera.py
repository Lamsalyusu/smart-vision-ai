from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque


class EventPolarity(Enum):
    ON = 1
    OFF = 0


class EventCameraType(Enum):
    DAVIS = "davis"
    DAVIS346 = "davis346"
    DAVIS640 = "davis640"
    DVS128 = "dvs128"
    DVS240 = "dvs240"
    DVS346 = "dvs346"
    ATIS = "atis"
    GENERIC = "generic"


@dataclass(frozen=True, slots=True)
class Event:
    x: int
    y: int
    timestamp: float
    polarity: EventPolarity

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.timestamp, self.polarity.value])

    @staticmethod
    def from_array(arr: np.ndarray) -> 'Event':
        return Event(
            x=int(arr[0]),
            y=int(arr[1]),
            timestamp=arr[2],
            polarity=EventPolarity(int(arr[3]))
        )


@dataclass(frozen=True, slots=True)
class EventPacket:
    events: List[Event]
    timestamp: float
    sensor_id: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class EventFrame:
    timestamp: float
    on_events: np.ndarray
    off_events: np.ndarray
    width: int
    height: int

    def to_image(self, polarity: EventPolarity = None) -> np.ndarray:
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        if polarity == EventPolarity.ON or polarity is None:
            img[self.on_events[:, 1], self.on_events[:, 0]] = 255
        if polarity == EventPolarity.OFF or polarity is None:
            img[self.off_events[:, 1], self.off_events[:, 0]] = 128
        return img

    def to_rgb(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[self.on_events[:, 1], self.on_events[:, 0]] = [0, 255, 0]
        img[self.off_events[:, 1], self.off_events[:, 0]] = [0, 0, 255]
        return img


class EventCameraSimulator:
    def __init__(
        self,
        width: int = 346,
        height: int = 260,
        contrast_threshold: float = 0.2,
        refractory_period: float = 1e-4,
        noise_rate: float = 0.0
    ):
        self.width = width
        self.height = height
        self.contrast_threshold = contrast_threshold
        self.refractory_period = refractory_period
        self.noise_rate = noise_rate

        self.last_timestamp = np.zeros((height, width))
        self.last_polarity = np.zeros((height, width))
        self.last_intensity = np.zeros((height, width))
        self.refractory_until = np.zeros((height, width))

    def simulate(self, intensity_image: np.ndarray, timestamp: float) -> List[Event]:
        events = []

        if self.last_intensity.size == 0:
            self.last_intensity = intensity_image.copy()
            self.last_timestamp = np.full_like(intensity_image, timestamp)
            return events

        contrast = np.abs(intensity_image.astype(float) - self.last_intensity.astype(float)) / (self.last_intensity.astype(float) + 1e-6)

        mask = contrast > self.contrast_threshold

        if self.refractory_period > 0:
            refractory_mask = timestamp - self.refractory_until > self.refractory_period
            mask = mask & refractory_mask

        if self.noise_rate > 0:
            noise_mask = np.random.random(mask.shape) < self.noise_rate
            mask = mask | noise_mask

        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            self.last_intensity = intensity_image.copy()
            self.last_timestamp = np.full_like(intensity_image, timestamp)
            return []

        polarities = []
        for x, y in zip(x_coords, y_coords):
            polarity = EventPolarity.ON if intensity_image[y, x] > self.last_intensity[y, x] else EventPolarity.OFF
            polarities.append(polarity)

        events = [
            Event(x=int(x), y=int(y), timestamp=timestamp, polarity=p)
            for x, y, p in zip(x_coords, y_coords, polarities)
        ]

        for x, y in zip(x_coords, y_coords):
            self.last_timestamp[y, x] = timestamp
            self.refractory_until[y, x] = timestamp
            self.last_intensity[y, x] = intensity_image[y, x]

        return events


class EventAccumulator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.on_image = np.zeros((height, width), dtype=np.uint32)
        self.off_image = np.zeros((height, width), dtype=np.uint32)
        self.timestamp_image = np.zeros((height, width), dtype=np.float64)

    def add_event(self, event: Event):
        if event.polarity == EventPolarity.ON:
            self.on_image[event.y, event.x] += 1
        else:
            self.off_image[event.y, event.x] += 1
        self.timestamp_image[event.y, event.x] = event.timestamp

    def add_events(self, events: List[Event]):
        for event in events:
            self.add_event(event)

    def clear(self):
        self.on_image.fill(0)
        self.off_image.fill(0)
        self.timestamp_image.fill(0)

    def get_event_count(self) -> int:
        return np.sum(self.on_image) + np.sum(self.off_image)

    def get_on_count(self) -> int:
        return int(np.sum(self.on_image))

    def get_off_count(self) -> int:
        return int(np.sum(self.off_image))

    def to_frame(self) -> EventFrame:
        on_coords = np.where(self.on_image > 0)
        off_coords = np.where(self.off_image > 0)

        on_events = np.column_stack([on_coords[1], on_coords[0]])
        off_events = np.column_stack([off_coords[1], off_coords[0]])

        return EventFrame(
            timestamp=time.time(),
            on_events=on_events,
            off_events=off_events,
            width=self.width,
            height=self.height
        )


class EventTimeSurface:
    def __init__(self, width: int, height: int, decay: float = 1e6):
        self.width = width
        self.height = height
        self.decay = decay
        self.surface = np.zeros((height, width), dtype=np.float64)
        self.last_update = np.zeros((height, width), dtype=np.float64)

    def update(self, event: Event):
        self.surface[event.y, event.x] = event.timestamp
        self.last_update[event.y, event.x] = event.timestamp

    def update_batch(self, events: List[Event]):
        for event in events:
            self.surface[event.y, event.x] = event.timestamp

    def get_surface(self, current_time: float) -> np.ndarray:
        return np.exp(-(current_time - self.surface) / self.decay)

    def get_gradient(self, current_time: float) -> Tuple[np.ndarray, np.ndarray]:
        surface = self.get_surface(current_time)
        grad_x = np.gradient(surface, axis=1)
        grad_y = np.gradient(surface, axis=0)
        return grad_x, grad_y


class EventPlaneFitting:
    @staticmethod
    def fit_plane(events: List[Event], window_size: float = 1e-3) -> Tuple[np.ndarray, float]:
        if len(events) < 3:
            return np.zeros(3), 0.0

        points = np.array([[e.x, e.y, e.timestamp] for e in events])
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]

        return normal, normal[2]

    @staticmethod
    def fit_paraboloid(events: List[Event], window_size: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, float]:
        if len(events) < 6:
            return np.zeros(3), np.zeros(2), 0.0

        points = np.array([[e.x, e.y, e.timestamp] for e in events])
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        U, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]

        x = points[:, 0] - centroid[0]
        y = points[:, 1] - centroid[1]
        t = points[:, 2] - centroid[2]

        A = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
        coeffs, _, _, _ = np.linalg.lstsq(A, t, rcond=None)

        return normal, coeffs[:2], coeffs[2]


class EventOpticalFlow:
    @staticmethod
    def compute_flow(
        events: List[Event],
        window_size: float = 1e-3,
        grid_size: Tuple[int, int] = (10, 10)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(events) < 10:
            return np.zeros((10, 10, 2)), np.zeros(10), np.zeros(10)

        events_sorted = sorted(events, key=lambda e: e.timestamp)
        timestamps = np.array([e.timestamp for e in events_sorted])
        min_t, max_t = timestamps[0], timestamps[-1]

        grid_w, grid_h = grid_size
        x_bins = np.linspace(0, 346, grid_size[0] + 1)
        y_bins = np.linspace(0, 260, grid_size[1] + 1)

        flow_x = np.zeros((grid_size[1], grid_size[0]))
        flow_y = np.zeros((grid_size[1], grid_size[0]))
        confidence = np.zeros((grid_size[1], grid_size[0]))

        for i in range(grid_size[0] - 1):
            for j in range(grid_size[1] - 1):
                x_min, x_max = x_bins[i], x_bins[i + 1]
                y_min, y_max = y_bins[j], y_bins[j + 1]

                cell_events = [e for e in events_sorted
                               if x_min <= e.x < x_max and y_min <= e.y < y_max]

                if len(cell_events) < 5:
                    continue

                points = np.array([[e.x, e.y, e.timestamp] for e in cell_events])
                centroid = np.mean(points, axis=0)
                centered = points - np.mean(points, axis=0)

                _, _, Vt = np.linalg.svd(centered)
                normal = Vt[-1]

                if abs(normal[2]) > 1e-6:
                    flow_x[j, i] = -normal[0] / normal[2]
                    flow_y[j, i] = -normal[1] / normal[2]
                    confidence[j, i] = min(len(cell_events) / 20.0, 1.0)

        return flow_x, flow_y, confidence


class EventCornerDetector:
    @staticmethod
    def detect_corners(
        events: List[Event],
        window_size: float = 1e-3,
        threshold: float = 0.01
    ) -> List[Event]:
        if len(events) < 10:
            return []

        events_sorted = sorted(events, key=lambda e: e.timestamp)
        timestamps = np.array([e.timestamp for e in events_sorted])
        min_t = timestamps[0]
        window_end = min_t + window_size

        window_events = [e for e in events_sorted if e.timestamp <= window_end]
        if len(window_events) < 10:
            return []

        harris = EventCornerDetector._compute_harris(window_events, threshold)
        return harris

    @staticmethod
    def _compute_harris(events: List[Event], threshold: float) -> List[Event]:
        points = np.array([[e.x, e.y] for e in events])

        if len(points) < 4:
            return []

        from scipy.spatial import KDTree
        tree = KDTree(points)

        corner_scores = []
        corners = []

        for i, pt in enumerate(points):
            indices = tree.query_ball_point(pt, 3.0)
            if len(indices) < 4:
                continue

            neighbors = points[indices]
            centroid = np.mean(neighbors, axis=0)
            cov = (neighbors - centroid).T @ (neighbors - centroid)

            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.sort(eigvals)

            if eigvals[0] > threshold and eigvals[1] > threshold:
                score = eigvals[0] * eigvals[1]
                corner_scores.append(score)
                corners.append(Event(
                    x=int(pt[0]),
                    y=int(pt[1]),
                    timestamp=time.time(),
                    polarity=EventPolarity.ON
                ))

        if corner_scores:
            indices = np.argsort(corner_scores)[::-1]
            return [corners[i] for i in indices[:100]]

        return []


class EventVisualizer:
    @staticmethod
    def draw_events(
        events: List[Event],
        width: int = 346,
        height: int = 260,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not events:
            return vis

        scale_x = image_size[0] / width
        scale_y = image_size[1] / height

        for event in events:
            x = int(event.x * scale_x)
            y = int(event.y * scale_y)

            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                color = (0, 255, 0) if event.polarity == EventPolarity.ON else (0, 0, 255)
                cv2.circle(vis, (x, y), 1, color, -1)

        return vis

    @staticmethod
    def draw_time_surface(
        time_surface: EventTimeSurface,
        current_time: float,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        surface = time_surface.get_surface(current_time)
        vis = cv2.normalize(surface, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        vis = cv2.resize(vis, image_size)
        return vis

    @staticmethod
    def draw_flow(
        flow_x: np.ndarray,
        flow_y: np.ndarray,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        h, w = flow_x.shape
        scale_x = image_size[0] / w
        scale_y = image_size[1] / h

        for y in range(0, h, 4):
            for x in range(0, w, 4):
                fx = flow_x[y, x]
                fy = flow_y[y, x]
                mag = math.sqrt(fx**2 + fy**2)

                if mag > 0.01:
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    end_x = int(x + fx * 10)
                    end_y = int(y - fy * 10)

                    if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                        cv2.arrowedLine(vis, (x, y), (end_x, end_y), (0, 255, 0), 1)

        return vis


class EventDataset:
    def __init__(self, path: str):
        self.path = path
        self.events: List[Event] = []
        self.metadata: Dict = {}

    def load_aedat(self, filepath: str):
        pass

    def load_npz(self, filepath: str):
        data = np.load(filepath)
        self.events = [Event.from_array(arr) for arr in data['events']]
        self.metadata = dict(data.get('metadata', {}))

    def save_npz(self, filepath: str):
        events_arr = np.array([e.to_array() for e in self.events])
        np.savez_compressed(filepath, events=events_arr, metadata=self.metadata)

    def filter_by_time(self, start: float, end: float) -> 'EventDataset':
        filtered = EventDataset(self.path)
        filtered.events = [e for e in self.events if start <= e.timestamp <= end]
        filtered.metadata = self.metadata.copy()
        return filtered

    def filter_by_region(self, x_min: int, x_max: int, y_min: int, y_max: int) -> 'EventDataset':
        filtered = EventDataset(self.path)
        filtered.events = [
            e for e in self.events
            if x_min <= e.x <= x_max and y_min <= e.y <= y_max
        ]
        filtered.metadata = self.metadata.copy()
        return filtered

    def downsample(self, factor: int) -> 'EventDataset':
        filtered = EventDataset(self.path)
        filtered.events = self.events[::factor]
        filtered.metadata = self.metadata.copy()
        return filtered


class EventMetrics:
    @staticmethod
    def event_rate(events: List[Event], time_window: float = 1.0) -> float:
        if not events:
            return 0.0

        timestamps = np.array([e.timestamp for e in events])
        time_span = timestamps[-1] - timestamps[0]
        if time_span < time_window:
            return len(events) / max(time_span, 1e-6)
        return len([e for e in events if e.timestamp >= timestamps[-1] - time_window]) / time_window

    @staticmethod
    def polarity_ratio(events: List[Event]) -> float:
        on_count = sum(1 for e in events if e.polarity == EventPolarity.ON)
        off_count = sum(1 for e in events if e.polarity == EventPolarity.OFF)
        total = on_count + off_count
        return on_count / total if total > 0 else 0.0

    @staticmethod
    def spatial_distribution(events: List[Event], bins: int = 10) -> np.ndarray:
        if not events:
            return np.zeros((bins, bins))

        x_coords = np.array([e.x for e in events])
        y_coords = np.array([e.y for e in events])

        hist, _, _ = np.histogram2d(x_coords, y_coords, bins=bins)
        return hist / (np.sum(hist) + 1e-8)

    @staticmethod
    def temporal_distribution(events: List[Event], bins: int = 100) -> np.ndarray:
        if not events:
            return np.zeros(bins)

        timestamps = np.array([e.timestamp for e in events])
        hist, _ = np.histogram(timestamps, bins=bins)
        return hist / (np.sum(hist) + 1e-8)


class EventProcessor:
    def __init__(self):
        self.filters: List[Callable[[Event], bool]] = []
        self.transforms: List[Callable[[Event], Event]] = []

    def add_filter(self, filter_fn: Callable[[Event], bool]):
        self.filters.append(filter_fn)

    def add_transform(self, transform_fn: Callable[[Event], Event]):
        self.transforms.append(transform_fn)

    def process(self, events: List[Event]) -> List[Event]:
        result = events

        for f in self.filters:
            result = [e for e in result if f(e)]

        for t in self.transforms:
            result = [t(e) for e in result]

        return result


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet