from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import time
import threading
import queue
import uuid


class StreamType(Enum):
    RTSP = "rtsp"
    HTTP = "http"
    WEBCAM = "webcam"
    FILE = "file"
    VIRTUAL = "virtual"


@dataclass(frozen=True, slots=True)
class StreamConfig:
    stream_id: str
    stream_type: StreamType
    url: str
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    buffer_size: int = 10
    reconnect_delay: float = 5.0
    max_retries: int = 3
    timeout: float = 10.0
    codec: str = "h264"
    extra_args: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_args is None:
            object.__setattr__(self, 'extra_args', {})


@dataclass(frozen=True, slots=True)
class Frame:
    frame_id: str
    stream_id: str
    timestamp: float
    image: np.ndarray
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


class StreamReader(ABC):
    @abstractmethod
    def open(self, config: StreamConfig) -> bool:
        pass

    @abstractmethod
    def read(self) -> Optional[Frame]:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        pass


class OpenCVReader(StreamReader):
    def __init__(self):
        self.cap = None
        self.config = None
        self.frame_count = 0

    def open(self, config: StreamConfig) -> bool:
        self.config = config
        try:
            if config.stream_type == StreamType.WEBCAM:
                self.cap = cv2.VideoCapture(int(config.url))
            elif config.stream_type == StreamType.FILE:
                self.cap = cv2.VideoCapture(config.url)
            elif config.stream_type == StreamType.RTSP:
                self.cap = cv2.VideoCapture(config.url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.buffer_size)
            elif config.stream_type == StreamType.HTTP:
                self.cap = cv2.VideoCapture(config.url)

            if not self.cap.isOpened():
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            self.cap.set(cv2.CAP_PROP_FPS, config.fps)
            self.frame_count = 0
            return True
        except Exception:
            return False

    def read(self) -> Optional[Frame]:
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        self.frame_count += 1
        return Frame(
            frame_id=f"{self.config.stream_id}_{self.frame_count}",
            stream_id=self.config.stream_id,
            timestamp=time.time(),
            image=frame,
            metadata={'frame_number': self.frame_count},
        )

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()


class FFmpegReader(StreamReader):
    def __init__(self):
        self.process = None
        self.config = None
        self.frame_size = 0
        self.frame_count = 0

    def open(self, config: StreamConfig) -> bool:
        self.config = config
        try:
            import subprocess

            cmd = [
                'ffmpeg',
                '-rtsp_transport', 'tcp',
                '-i', config.url,
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-vf', f'scale={config.width}:{config.height}',
                '-r', str(config.fps),
                '-'
            ]

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
            self.frame_size = config.width * config.height * 3
            self.frame_count = 0
            return True
        except Exception:
            return False

    def read(self) -> Optional[Frame]:
        if self.process is None or self.process.poll() is not None:
            return None

        try:
            raw = self.process.stdout.read(self.frame_size)
            if len(raw) != self.frame_size:
                return None

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                self.config.height, self.config.width, 3
            )

            self.frame_count += 1
            return Frame(
                frame_id=f"{self.config.stream_id}_{self.frame_count}",
                stream_id=self.config.stream_id,
                timestamp=time.time(),
                image=frame,
                metadata={'frame_number': self.frame_count},
            )
        except Exception:
            return None

    def close(self):
        if self.process:
            self.process.terminate()
            self.process = None

    def is_opened(self) -> bool:
        return self.process is not None and self.process.poll() is None


class StreamManager:
    def __init__(
        self,
        max_streams: int = 100,
        default_buffer_size: int = 30,
    ):
        self.max_streams = max_streams
        self.default_buffer_size = default_buffer_size
        self.streams: Dict[str, Dict] = {}
        self.readers: Dict[str, StreamReader] = {}
        self.frame_queues: Dict[str, queue.Queue] = {}
        self.worker_threads: Dict[str, threading.Thread] = {}
        self.running = False
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)

    def add_stream(self, config: StreamConfig) -> bool:
        if len(self.streams) >= self.max_streams:
            return False
        if config.stream_id in self.streams:
            return False

        reader = self._create_reader(config.stream_type)
        if not reader.open(config):
            return False

        self.streams[config.stream_id] = {
            'config': config,
            'reader': reader,
            'status': 'running',
            'frames_received': 0,
            'frames_dropped': 0,
            'last_frame_time': 0,
            'errors': 0,
        }
        self.readers[config.stream_id] = reader
        self.frame_queues[config.stream_id] = queue.Queue(maxsize=config.buffer_size)

        self.worker_threads[config.stream_id] = threading.Thread(
            target=self._stream_worker,
            args=(config.stream_id,),
            daemon=True
        )
        self.running = True
        self.worker_threads[config.stream_id].start()
        return True

    def _create_reader(self, stream_type: StreamType) -> StreamReader:
        if stream_type in (StreamType.RTSP, StreamType.HTTP):
            return FFmpegReader()
        return OpenCVReader()

    def _stream_worker(self, stream_id: str):
        stream = self.streams[stream_id]
        reader = self.readers[stream_id]
        frame_queue = self.frame_queues[stream_id]
        config = stream['config']

        while self.running and stream_id in self.streams:
            if not reader.is_opened():
                stream['status'] = 'disconnected'
                stream['errors'] += 1
                if stream['errors'] > config.max_retries:
                    break
                time.sleep(config.reconnect_delay)
                if not reader.open(config):
                    continue
                stream['status'] = 'running'
                stream['errors'] = 0

            frame = reader.read()
            if frame is None:
                time.sleep(0.01)
                continue

            stream['frames_received'] += 1
            stream['last_frame_time'] = frame.timestamp

            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                stream['frames_dropped'] += 1

            for callback in self.callbacks[stream_id]:
                try:
                    callback(frame)
                except Exception:
                    pass

        stream['status'] = 'stopped'

    def remove_stream(self, stream_id: str):
        if stream_id in self.streams:
            self.streams[stream_id]['status'] = 'stopping'
            if stream_id in self.worker_threads:
                self.worker_threads[stream_id].join(timeout=5)
                del self.worker_threads[stream_id]
            if stream_id in self.readers:
                self.readers[stream_id].close()
                del self.readers[stream_id]
            if stream_id in self.frame_queues:
                del self.frame_queues[stream_id]
            del self.streams[stream_id]

    def get_frame(self, stream_id: str, timeout: float = 1.0) -> Optional[Frame]:
        if stream_id not in self.frame_queues:
            return None
        try:
            return self.frame_queues[stream_id].get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest_frame(self, stream_id: str) -> Optional[Frame]:
        if stream_id not in self.frame_queues:
            return None
        frames = []
        while True:
            try:
                frames.append(self.frame_queues[stream_id].get_nowait())
            except queue.Empty:
                break
        return frames[-1] if frames else None

    def register_callback(self, stream_id: str, callback: Callable[[Frame], None]):
        self.callbacks[stream_id].append(callback)

    def unregister_callback(self, stream_id: str, callback: Callable):
        if stream_id in self.callbacks:
            self.callbacks[stream_id] = [c for c in self.callbacks[stream_id] if c != callback]

    def get_stream_info(self, stream_id: str) -> Optional[Dict]:
        if stream_id not in self.streams:
            return None
        s = self.streams[stream_id]
        return {
            'stream_id': stream_id,
            'status': s['status'],
            'frames_received': s['frames_received'],
            'frames_dropped': s['frames_dropped'],
            'current_fps': self._calculate_fps(stream_id),
            'queue_size': self.frame_queues[stream_id].qsize() if stream_id in self.frame_queues else 0,
        }

    def _calculate_fps(self, stream_id: str) -> float:
        return 0.0

    def list_streams(self) -> List[str]:
        return list(self.streams.keys())

    def shutdown(self):
        self.running = False
        for stream_id in list(self.streams.keys()):
            self.remove_stream(stream_id)


class SynchronizedFrameGrabber:
    def __init__(
        self,
        stream_manager: StreamManager,
        sync_tolerance: float = 0.05,
        max_buffer: int = 100,
    ):
        self.stream_manager = stream_manager
        self.sync_tolerance = sync_tolerance
        self.max_buffer = max_buffer
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_buffer))
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.sync_callbacks: List[Callable] = []

    def start(self, stream_ids: List[str]):
        self.running = True
        self.worker_thread = threading.Thread(target=self._sync_worker, args=(stream_ids,), daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def _sync_worker(self, stream_ids: List[str]):
        while self.running:
            frames = {}
            timestamps = {}

            for sid in stream_ids:
                frame = self.stream_manager.get_latest_frame(sid)
                if frame:
                    frames[sid] = frame
                    timestamps[sid] = frame.timestamp

            if len(frames) == len(stream_ids):
                ts_values = list(timestamps.values())
                max_ts = max(ts_values)
                min_ts = min(ts_values)

                if max_ts - min_ts <= self.sync_tolerance:
                    for callback in self.sync_callbacks:
                        try:
                            callback(frames)
                        except Exception:
                            pass

            time.sleep(0.01)

    def register_sync_callback(self, callback: Callable[[Dict[str, Frame]], None]):
        self.sync_callbacks.append(callback)


class FrameProcessor:
    def __init__(
        self,
        processors: List[Callable[[Frame], Optional[Frame]]] = None,
    ):
        self.processors = processors or []

    def add_processor(self, processor: Callable[[Frame], Optional[Frame]]):
        self.processors.append(processor)

    def process(self, frame: Frame) -> Optional[Frame]:
        current = frame
        for processor in self.processors:
            if current is None:
                break
            current = processor(current)
        return current

    def process_batch(self, frames: List[Frame]) -> List[Frame]:
        return [self.process(f) for f in frames if self.process(f) is not None]


def create_preprocessor(
    target_size: tuple[int, int] = None,
    normalize: bool = True,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> Callable[[Frame], Optional[Frame]]:
    def preprocess(frame: Frame) -> Optional[Frame]:
        image = frame.image

        if target_size and (image.shape[1] != target_size[0] or image.shape[0] != target_size[1]):
            image = cv2.resize(image, target_size)

        if normalize:
            image = image.astype(np.float32) / 255.0
            mean_arr = np.array(mean).reshape(1, 1, 3)
            std_arr = np.array(std).reshape(1, 1, 3)
            image = (image - mean_arr) / std_arr

        return Frame(
            frame_id=frame.frame_id,
            stream_id=frame.stream_id,
            timestamp=frame.timestamp,
            image=image,
            metadata=frame.metadata,
        )
    return preprocess


def create_augmenter(
    flip_prob: float = 0.5,
    rotate_range: float = 10.0,
    brightness_range: tuple = (0.8, 1.2),
    noise_std: float = 0.01,
) -> Callable[[Frame], Optional[Frame]]:
    def augment(frame: Frame) -> Optional[Frame]:
        image = frame.image.copy()

        if np.random.random() < flip_prob:
            image = cv2.flip(image, 1)

        if rotate_range > 0:
            angle = np.random.uniform(-rotate_range, rotate_range)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))

        if brightness_range[0] != 1.0 or brightness_range[1] != 1.0:
            factor = np.random.uniform(*brightness_range)
            image = np.clip(image * factor, 0, 255)

        if noise_std > 0:
            noise = np.random.normal(0, noise_std * 255, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 255)

        return Frame(
            frame_id=frame.frame_id,
            stream_id=frame.stream_id,
            timestamp=frame.timestamp,
            image=image,
            metadata=frame.metadata,
        )
    return augment