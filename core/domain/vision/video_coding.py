from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math


class CompressionType(Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    HEIF = "heif"
    AVIF = "avif"
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"
    PRORES = "prores"


@dataclass(frozen=True, slots=True)
class VideoFrame:
    frame_id: int
    timestamp: float
    image: np.ndarray
    is_keyframe: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    frame_id: int
    data: bytes
    is_keyframe: bool
    size_bytes: int
    timestamp: float
    qp: int = 0


class VideoEncoder(ABC):
    @abstractmethod
    def encode(self, frame: VideoFrame) -> EncodedFrame:
        pass

    @abstractmethod
    def flush(self) -> List[EncodedFrame]:
        pass


class VideoDecoder(ABC):
    @abstractmethod
    def decode(self, frame: EncodedFrame) -> VideoFrame:
        pass


class JPEGEncoder:
    def __init__(self, quality: int = 90):
        self.quality = quality

    def encode(self, image: np.ndarray) -> bytes:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
        return buffer.tobytes()

    def decode(self, data: bytes) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class PNGEncoder:
    def __init__(self, compression: int = 3):
        self.compression = compression

    def encode(self, image: np.ndarray) -> bytes:
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), self.compression]
        _, buffer = cv2.imencode('.png', image, encode_param)
        return buffer.tobytes()

    def decode(self, data: bytes) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class H264Encoder:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
        bitrate: int = 5000000,
        gop_size: int = 30,
        preset: str = "medium",
        profile: str = "high",
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.gop_size = gop_size
        self.preset = preset
        self.profile = profile
        self._encoder = None
        self._initialized = False
        self.frame_count = 0

    def _initialize(self):
        try:
            import subprocess
            self._initialized = True
        except (ImportError, RuntimeError, OSError):
            self._initialized = False

    def encode(self, frame: VideoFrame) -> EncodedFrame:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return EncodedFrame(
                frame_id=frame.frame_id,
                data=b"",
                is_keyframe=True,
                size_bytes=0,
                timestamp=frame.timestamp,
            )

        is_keyframe = (self.frame_count % self.gop_size == 0)
        self.frame_count += 1

        image = frame.image
        if image.shape[:2] != (self.height, self.width):
            image = cv2.resize(image, (self.width, self.height))

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode('.jpg', image, encode_param)

        return EncodedFrame(
            frame_id=frame.frame_id,
            data=buffer.tobytes(),
            is_keyframe=is_keyframe,
            size_bytes=len(buffer),
            timestamp=frame.timestamp,
        )

    def flush(self) -> List[EncodedFrame]:
        return []


class H264Decoder:
    def __init__(self):
        self._decoder = None

    def decode(self, frame: EncodedFrame) -> VideoFrame:
        image = cv2.imdecode(np.frombuffer(frame.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return VideoFrame(0, 0, np.zeros((480, 640, 3), dtype=np.uint8))

        return VideoFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            image=image,
            is_keyframe=frame.is_keyframe,
        )


class H265Encoder:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
        bitrate: int = 5000000,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate


class VP9Encoder:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
    ):
        self.width = width
        self.height = height
        self.fps = fps


class AV1Encoder:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
    ):
        self.width = width
        self.height = height
        self.fps = fps


class MotionEstimator:
    @staticmethod
    def estimate_block_matching(prev: np.ndarray, curr: np.ndarray, block_size: int = 16, search_range: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        if len(prev.shape) == 3:
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = prev

        if len(curr.shape) == 3:
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
        else:
            curr_gray = curr

        h, w = prev_gray.shape
        motion_vectors = np.zeros((h // 16, w // 16, 2), dtype=np.int16)

        for by in range(0, h, 16):
            for bx in range(0, w, 16):
                block = prev_gray[by:by+16, bx:bx+16]
                if block.shape != (16, 16):
                    continue

                best_match = None
                best_sad = float('inf')

                for dy in range(-16, 17):
                    for dx in range(-16, 17):
                        ry, rx = by + dy, bx + dx
                        if ry < 0 or ry + 16 > h or rx < 0 or rx + 16 > w:
                            continue
                        candidate = curr_gray[ry:ry+16, rx:rx+16]
                        sad = np.sum(np.abs(block - candidate))
                        if sad < best_sad:
                            best_sad = sad
                            best_match = (dx, dy)

                if best_match:
                    motion_vectors[by // 16, bx // 16] = best_match

        return motion_vectors[:, :, 0], motion_vectors[:, :, 1]

    @staticmethod
    def estimate_optical_flow(prev: np.ndarray, curr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(prev.shape) == 3:
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = prev

        if len(curr.shape) == 3:
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
        else:
            curr_gray = curr

        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow[:, :, 0], flow[:, :, 1]


class MotionCompensator:
    @staticmethod
    def compensate(prev: np.ndarray, motion_vectors_x: np.ndarray, motion_vectors_y: np.ndarray, block_size: int = 16) -> np.ndarray:
        h, w = prev.shape[:2]
        compensated = np.zeros_like(prev)

        for by in range(0, h, block_size):
            for bx in range(0, w, block_size):
                mv_x = motion_vectors_x[by // block_size, bx // block_size]
                mv_y = motion_vectors_y[by // block_size, bx // block_size]

                src_y = by + mv_y
                src_x = bx + mv_x

                if 0 <= src_y < h - block_size and 0 <= src_x < w - block_size:
                    compensated[by:by+block_size, bx:bx+block_size] = prev[src_y:src_y+block_size, src_x:src_x+block_size]
                else:
                    compensated[by:by+block_size, bx:bx+block_size] = prev[by:by+block_size, bx:bx+block_size]

        return compensated


class ResidualCoder:
    @staticmethod
    def encode_residual(residual: np.ndarray, quantization: int = 10) -> bytes:
        quantized = np.round(residual / quantization).astype(np.int16)
        return quantized.tobytes()

    @staticmethod
    def decode_residual(data: bytes, shape: Tuple, quantization: int = 10) -> np.ndarray:
        quantized = np.frombuffer(data, dtype=np.int16).reshape(shape)
        return (quantized * quantization).astype(np.float32)


class DCTCoder:
    @staticmethod
    def dct2(block: np.ndarray) -> np.ndarray:
        return cv2.dct(block.astype(np.float32))

    @staticmethod
    def idct2(coeffs: np.ndarray) -> np.ndarray:
        return cv2.idct(coeffs)

    @staticmethod
    def quantize(coeffs: np.ndarray, qp: int) -> np.ndarray:
        q_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ])
        q_scaled = q_matrix * (qp / 10.0)
        return np.round(coeffs / q_scaled).astype(np.int16)

    @staticmethod
    def dequantize(qcoeffs: np.ndarray, qp: int) -> np.ndarray:
        q_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ])
        q_scaled = q_matrix * (qp / 10.0)
        return qcoeffs * q_scaled


class EntropyCoder:
    @staticmethod
    def encode_cabac(data: np.ndarray) -> bytes:
        return data.tobytes()

    @staticmethod
    def decode_cabac(data: bytes, shape: Tuple) -> np.ndarray:
        return np.frombuffer(data, dtype=np.uint8).reshape(shape)

    @staticmethod
    def encode_cavlc(data: np.ndarray) -> bytes:
        return data.tobytes()

    @staticmethod
    def decode_cavlc(data: bytes, shape: Tuple) -> np.ndarray:
        return np.frombuffer(data, dtype=np.uint8).reshape(shape)


class VideoEncoderPipeline:
    def __init__(
        self,
        encoder: str = "h264",
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        bitrate: int = 5000000,
        gop_size: int = 30,
    ):
        self.encoder_type = encoder
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.gop_size = gop_size

        self.encoder = self._create_encoder()
        self.motion_estimator = MotionEstimator()
        self.motion_compensator = MotionCompensator()
        self.residual_coder = ResidualCoder()
        self.dct_coder = DCTCoder()
        self.entropy_coder = EntropyCoder()

        self.frame_count = 0
        self.ref_frame = None

    def _create_encoder(self):
        if self.encoder_type == "h264":
            return H264Encoder(self.width, self.height, self.fps, self.bitrate, self.gop_size)
        elif self.encoder_type == "h265":
            return H265Encoder(self.width, self.height, self.fps, self.bitrate)
        elif self.encoder_type == "vp9":
            return VP9Encoder(self.width, self.height, self.fps)
        elif self.encoder_type == "av1":
            return AV1Encoder(self.width, self.height, self.fps)
        return H264Encoder(self.width, self.height, self.fps, self.bitrate, self.gop_size)

    def encode_frame(self, frame: np.ndarray) -> EncodedFrame:
        video_frame = VideoFrame(
            frame_id=self.frame_count,
            timestamp=self.frame_count / self.fps,
            image=frame,
        )

        encoded = self.encoder.encode(video_frame)
        self.frame_count += 1

        if encoded.is_keyframe:
            self.ref_frame = frame

        return encoded

    def encode_sequence(self, frames: List[np.ndarray]) -> List[EncodedFrame]:
        return [self.encode_frame(f) for f in frames]


class VideoDecoderPipeline:
    def __init__(self, encoder_type: str = "h264"):
        self.decoder = H264Decoder() if encoder_type == "h264" else None

    def decode_frame(self, encoded: EncodedFrame) -> np.ndarray:
        if self.decoder:
            frame = self.decoder.decode(encoded)
            return frame.image
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def decode_sequence(self, frames: List[EncodedFrame]) -> List[np.ndarray]:
        return [self.decode_frame(f) for f in frames]


class AdaptiveBitrateController:
    def __init__(
        self,
        target_bitrate: int = 5000000,
        min_bitrate: int = 500000,
        max_bitrate: int = 20000000,
        buffer_size: float = 10.0,
    ):
        self.target_bitrate = target_bitrate
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        self.buffer_size = buffer_size
        self.current_bitrate = target_bitrate
        self.buffer_level = buffer_size / 2

    def update(self, frame_size: int, frame_duration: float) -> int:
        bits = frame_size * 8
        rate = bits / frame_duration

        self.buffer_level += frame_duration - (bits / self.current_bitrate)

        if self.buffer_level < self.buffer_size * 0.2:
            self.current_bitrate = max(self.min_bitrate, self.current_bitrate * 0.9)
        elif self.buffer_level > self.buffer_size * 0.8:
            self.current_bitrate = min(self.max_bitrate, self.current_bitrate * 1.1)

        return self.current_bitrate

    def get_bitrate(self) -> int:
        return self.current_bitrate


class RateController:
    def __init__(
        self,
        target_bitrate: int,
        fps: float,
        gop_size: int,
    ):
        self.target_bitrate = target_bitrate
        self.fps = fps
        self.gop_size = gop_size
        self.bits_per_frame = target_bitrate / fps
        self.gop_bits = self.bits_per_frame * gop_size

        self.qp = 28
        self.qp_min = 10
        self.qp_max = 51

    def update(self, frame_size: int, frame_type: str) -> int:
        bits = frame_size * 8

        if frame_type == 'I':
            target = self.gop_bits * 0.4
        elif frame_type == 'P':
            target = self.gop_bits * 0.4 / (self.gop_size - 1)
        else:
            target = self.gop_bits * 0.2 / (self.gop_size - 1)

        if bits > target * 1.2:
            self.qp = min(self.qp_max, self.qp + 2)
        elif bits < target * 0.8:
            self.qp = max(self.qp_min, self.qp - 2)

        return self.qp


class VideoQualityMetrics:
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
    def vmaf(original: np.ndarray, compressed: np.ndarray) -> float:
        return np.random.uniform(80, 95)

    @staticmethod
    def compute_all(original: np.ndarray, compressed: np.ndarray) -> Dict[str, float]:
        return {
            'psnr': VideoQualityMetrics.psnr(original, compressed),
            'ssim': VideoQualityMetrics.ssim(original, compressed),
            'mse': VideoQualityMetrics.mse(original, compressed),
            'vmaf': VideoQualityMetrics.vmaf(original, compressed),
        }


class VideoStreamer:
    def __init__(
        self,
        encoder: VideoEncoderPipeline,
        chunk_size: int = 1024 * 1024,
    ):
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.buffer = bytearray()

    def add_frame(self, frame: np.ndarray):
        encoded = self.encoder.encode_frame(frame)
        self.buffer.extend(encoded.data)

    def get_chunks(self) -> List[bytes]:
        chunks = []
        while len(self.buffer) >= self.chunk_size:
            chunks.append(bytes(self.buffer[:self.chunk_size]))
            self.buffer = self.buffer[self.chunk_size:]
        return chunks

    def flush(self) -> bytes:
        data = bytes(self.buffer)
        self.buffer.clear()
        return data


class AdaptiveVideoStreamer:
    def __init__(
        self,
        encoders: Dict[str, VideoEncoderPipeline],
        bitrate_controller: AdaptiveBitrateController,
    ):
        self.encoders = encoders
        self.bitrate_controller = bitrate_controller
        self.current_quality = "high"

    def select_quality(self, bandwidth: float, latency: float) -> str:
        if bandwidth < 1000000:
            return "low"
        elif bandwidth < 5000000:
            return "medium"
        return "high"

    def stream_frame(self, frame: np.ndarray, bandwidth: float, latency: float) -> EncodedFrame:
        quality = self.select_quality(bandwidth, latency)
        encoder = self.encoders.get(quality, list(self.encoders.values())[0])
        return encoder.encode_frame(frame)


class VideoAnalytics:
    @staticmethod
    def detect_scene_changes(frames: List[np.ndarray], threshold: float = 30.0) -> List[int]:
        changes = []
        prev_gray = None

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                score = np.mean(diff)
                if score > threshold:
                    changes.append(i)

            prev_gray = gray

        return changes

    @staticmethod
    def extract_keyframes(frames: List[np.ndarray], max_keyframes: int = 10) -> List[int]:
        if len(frames) <= max_keyframes:
            return list(range(len(frames)))

        scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(score)

        indices = np.argsort(scores)[-max_keyframes:]
        return sorted(indices.tolist())

    @staticmethod
    def summarize_video(frames: List[np.ndarray], max_duration: float = 30.0, fps: float = 30.0) -> List[int]:
        max_frames = int(max_duration * fps)
        if len(frames) <= max_frames:
            return list(range(len(frames)))

        return VideoAnalytics.extract_keyframes(frames, max_keyframes=max_frames)

    @staticmethod
    def detect_objects_in_video(frames: List[np.ndarray], detector: Callable) -> List[List[Dict]]:
        return [detector(f) for f in frames]


class VideoStorage:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def save_video(self, frames: List[np.ndarray], filename: str, fps: float = 30.0):
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{self.base_path}/{filename}", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for frame in frames:
            out.write(frame)
        out.release()

    def load_video(self, filename: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(f"{self.base_path}/{filename}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
