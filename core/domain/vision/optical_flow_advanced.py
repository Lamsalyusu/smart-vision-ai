from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math


class OpticalFlowAlgorithm(Enum):
    FARNEBACK = "farneback"
    LUCAS_KANADE = "lucas_kanade"
    DEEPFLOW = "deepflow"
    PWC_NET = "pwc_net"
    RAFT = "raft"
    FLOWNET = "flownet"
    LITEFLOWNET = "liteflownet"
    FLOWNET2 = "flownet2"
    SEL_FLOW = "sel_flow"
    HD3 = "hd3"


@dataclass(frozen=True, slots=True)
class FlowField:
    u: np.ndarray
    v: np.ndarray
    confidence: Optional[np.ndarray] = None
    timestamp: float = 0.0

    @property
    def magnitude(self) -> np.ndarray:
        return np.sqrt(self.u**2 + self.v**2)

    @property
    def angle(self) -> np.ndarray:
        return np.arctan2(self.v, self.u)

    def warp_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        flow_map = np.stack([self.u, self.v], axis=-1)
        grid = np.mgrid[0:h, 0:w].transpose(1, 2, 0).astype(np.float32)
        map_x = grid[:, :, 1] + self.u
        map_y = grid[:, :, 0] + self.v
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


@dataclass(frozen=True, slots=True)
class SparseFlow:
    points: np.ndarray
    flow: np.ndarray
    status: np.ndarray
    error: Optional[np.ndarray] = None


class BaseOpticalFlow(ABC):
    @abstractmethod
    def compute(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        pass

    @abstractmethod
    def compute_sparse(self, prev: np.ndarray, curr: np.ndarray, points: np.ndarray) -> SparseFlow:
        pass


class FarnebackFlow(BaseOpticalFlow):
    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.1,
        flags: int = 0,
    ):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags

    def compute(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        if len(prev.shape) == 3:
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = prev

        if len(curr.shape) == 3:
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
        else:
            curr_gray = curr

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma, self.flags
        )

        u = flow[:, :, 0]
        v = flow[:, :, 1]
        return FlowField(u=u, v=v)

    def compute_sparse(self, prev: np.ndarray, curr: np.ndarray, points: np.ndarray) -> SparseFlow:
        if len(prev.shape) == 3:
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = prev

        if len(curr.shape) == 3:
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
        else:
            curr_gray = curr

        points = points.reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prev, curr, points, None,
            winSize=(15, 15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        flow = next_pts - points
        return SparseFlow(
            points=points.reshape(-1, 2),
            flow=flow.reshape(-1, 2),
            status=status.flatten(),
            error=error.flatten() if error is not None else None,
        )


class LucasKanadeFlow(BaseOpticalFlow):
    def __init__(
        self,
        win_size: Tuple[int, int] = (15, 15),
        max_level: int = 3,
        criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    ):
        self.win_size = win_size
        self.max_level = max_level
        self.criteria = criteria

    def compute(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        if len(prev.shape) == 3:
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = prev

        if len(curr.shape) == 3:
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
        else:
            curr_gray = curr

        h, w = prev_gray.shape
        points = np.mgrid[0:h:10, 0:w:10].reshape(2, -1).T.astype(np.float32)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, curr,
            points.reshape(-1, 1, 2).astype(np.float32),
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria
        )

        flow_field = np.zeros((h, w, 2), dtype=np.float32)
        for i, (pt, npt, st) in enumerate(zip(points, next_pts, status)):
            if st:
                y, x = int(pt[1]), int(pt[0])
                dy, dx = npt - pt
                if 0 <= y < curr.shape[0] and 0 <= x < curr.shape[1]:
                    flow_field[y, x] = [dx, dy]

        u = flow_field[:, :, 0]
        v = flow_field[:, :, 1]
        return FlowField(u=u, v=v)

    def compute_sparse(self, prev: np.ndarray, curr: np.ndarray, points: np.ndarray) -> SparseFlow:
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prev, curr,
            points.reshape(-1, 1, 2).astype(np.float32),
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria
        )

        return SparseFlow(
            points=points,
            flow=next_pts.reshape(-1, 2) - points,
            status=status.flatten(),
            error=error.flatten() if error is not None else None,
        )


class DeepFlow(BaseOpticalFlow):
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def compute(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._compute_deep(prev, curr)
        else:
            return self._compute_classical(prev, curr)

    def _compute_classical(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        fb = FarnebackFlow()
        return fb.compute(prev, curr)

    def _compute_deep(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        return self._compute_classical(prev, curr)

    def compute_sparse(self, prev: np.ndarray, curr: np.ndarray, points: np.ndarray) -> SparseFlow:
        lk = LucasKanadeFlow()
        return lk.compute_sparse(prev, curr, points)


class PWCNetFlow(BaseOpticalFlow):
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class PWCNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(6, 16, 7, padding=3)
                    self.conv2 = nn.Conv2d(16, 32, 5, padding=2, stride=2)
                    self.conv3 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
                    self.conv4 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
                    self.conv5 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
                    self.conv6 = nn.Conv2d(256, 512, 3, padding=1, stride=2)

                    self.dc_conv1 = nn.Conv2d(512, 512, 3, padding=1)
                    self.dc_conv2 = nn.Conv2d(512, 512, 3, padding=1)
                    self.dc_conv3 = nn.Conv2d(512, 512, 3, padding=1)
                    self.dc_conv4 = nn.Conv2d(512, 512, 3, padding=1)
                    self.dc_conv5 = nn.Conv2d(512, 512, 3, padding=1)
                    self.dc_conv6 = nn.Conv2d(512, 512, 3, padding=1)

                    self.flow_conv = nn.Conv2d(512, 2, 3, padding=1)

                def forward(self, x1, x2):
                    x = torch.cat([x1, x2], dim=1)
                    c1 = F.relu(self.conv1(x))
                    c2 = F.relu(self.conv2(c1))
                    c3 = F.relu(self.conv3(c2))
                    c4 = F.relu(self.conv3(c3))
                    c5 = F.relu(self.conv4(c4))
                    c6 = F.relu(self.conv4(c5))

                    dc6 = F.relu(self.dc_conv1(c6))
                    dc5 = F.relu(self.dc_conv2(torch.cat([dc6, c5], dim=1)))
                    dc4 = F.relu(self.dc_conv3(torch.cat([dc5, c4], dim=1)))
                    dc3 = F.relu(self.dc_conv4(torch.cat([dc4, c3], dim=1)))
                    dc2 = F.relu(self.dc_conv5(torch.cat([dc3, c2], dim=1)))
                    dc1 = F.relu(self.dc_conv6(torch.cat([dc2, c1], dim=1)))

                    flow = self.flow_conv(dc1)
                    flow = F.interpolate(flow, scale_factor=4, mode='bilinear', align_corners=False)
                    return flow

            self._model = PWCNet()
            self._model.eval()
            self._initialized = True

        except ImportError:
            self._initialized = False

    def compute(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._compute_deep(prev, curr)
        else:
            return self._compute_classical(prev, curr)

    def _compute_classical(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        fb = FarnebackFlow()
        return fb.compute(prev, curr)

    def _compute_deep(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        return self._compute_classical(prev, curr)

    def compute_sparse(self, prev: np.ndarray, curr: np.ndarray, points: np.ndarray) -> SparseFlow:
        lk = LucasKanadeFlow()
        return lk.compute_sparse(prev, curr, points)


class RAFTFlow(BaseOpticalFlow):
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        iters: int = 12,
    ):
        self.model_path = model_path
        self.device = device
        self.iters = iters
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class RAFT(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Conv2d(3, 64, 7, padding=3),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1),
                        nn.ReLU(),
                    )
                    self.context_encoder = nn.Sequential(
                        nn.Conv2d(3, 64, 7, padding=3),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1),
                        nn.ReLU(),
                    )
                    self.flow_encoder = nn.Sequential(
                        nn.Conv2d(4, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, 3, padding=1),
                        nn.ReLU(),
                    )
                    self.flow_head = nn.Sequential(
                        nn.Conv2d(128, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 2, 3, padding=1),
                    )

                def forward(self, x1, x2, iters=12):
                    fmap1 = self.encoder(x1)
                    fmap2 = self.encoder(x2)

                    corr = self._corr_block(fmap1, fmap2)

                    net = self.context_encoder(x1)
                    net = torch.tanh(net)

                    coords0 = self._make_coords_grid(x1.shape[2:])
                    coords1 = coords0.clone()

                    for i in range(iters):
                        corr = self._corr_lookup(corr, coords1)
                        flow = coords1 - coords0
                        delta = self.flow_head(torch.cat([fmap1, corr, flow], dim=1))
                        coords1 = coords1 + delta

                    flow = coords1 - coords0
                    return flow

                def _corr_block(self, fmap1, fmap2):
                    return None

                def _corr_lookup(self, corr, coords):
                    return None

                def _make_coords_grid(self, shape):
                    return None

            self._model = RAFT()
            self._model.eval()
            self._initialized = True

        except ImportError:
            self._initialized = False

    def compute(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._compute_deep(prev, curr)
        else:
            return self._compute_classical(prev, curr)

    def _compute_classical(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        fb = FarnebackFlow()
        return fb.compute(prev, curr)

    def _compute_deep(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        return self._compute_classical(prev, curr)

    def compute_sparse(self, prev: np.ndarray, curr: np.ndarray, points: np.ndarray) -> SparseFlow:
        lk = LucasKanadeFlow()
        return lk.compute_sparse(prev, curr, points)


class FlowNetFlow(BaseOpticalFlow):
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class FlowNetS(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(6, 64, 7, padding=3, stride=2)
                    self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
                    self.conv3 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
                    self.conv4 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
                    self.conv5 = nn.Conv2d(512, 512, 3, padding=1, stride=2)
                    self.conv6 = nn.Conv2d(512, 1024, 3, padding=1, stride=2)

                    self.deconv5 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
                    self.deconv4 = nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1)
                    self.deconv3 = nn.ConvTranspose2d(768, 128, 4, stride=2, padding=1)
                    self.deconv2 = nn.ConvTranspose2d(384, 64, 4, stride=2, padding=1)
                    self.deconv1 = nn.ConvTranspose2d(192, 2, 4, stride=2, padding=1)

                    self.predict_flow6 = nn.Conv2d(1024, 2, 3, padding=1)
                    self.predict_flow5 = nn.Conv2d(1024, 2, 3, padding=1)
                    self.predict_flow4 = nn.Conv2d(768, 2, 3, padding=1)
                    self.predict_flow3 = nn.Conv2d(384, 2, 3, padding=1)
                    self.predict_flow2 = nn.Conv2d(192, 2, 3, padding=1)

                def forward(self, x1, x2):
                    x = torch.cat([x1, x2], dim=1)
                    return self._forward(x)

                def _forward(self, x):
                    c1 = F.relu(self.conv1(x))
                    c2 = F.relu(self.conv2(c1))
                    c3 = F.relu(self.conv3(c2))
                    c4 = F.relu(self.conv4(c3))
                    c5 = F.relu(self.conv5(c4))
                    c6 = F.relu(self.conv6(c5))

                    flow6 = self.predict_flow6(c6)
                    up5 = F.relu(self.deconv5(c6))
                    concat5 = torch.cat([up5, c5, flow6], dim=1)
                    flow5 = self.predict_flow5(concat5)

                    up4 = F.relu(self.deconv4(concat5))
                    concat4 = torch.cat([up4, c4, flow5], dim=1)
                    flow4 = self.predict_flow4(concat4)

                    up3 = F.relu(self.deconv3(concat4))
                    concat3 = torch.cat([up3, c3, flow4], dim=1)
                    flow3 = self.predict_flow3(concat3)

                    up2 = F.relu(self.deconv2(concat3))
                    concat2 = torch.cat([up2, c2, flow3], dim=1)
                    flow2 = self.predict_flow2(concat2)

                    up1 = F.relu(self.deconv1(concat2))
                    return flow2

            self._model = FlowNetS()
            self._model.eval()
            self._initialized = True

        except ImportError:
            self._initialized = False

    def compute(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._compute_deep(prev, curr)
        else:
            return self._compute_classical(prev, curr)

    def _compute_classical(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        fb = FarnebackFlow()
        return fb.compute(prev, curr)

    def _compute_deep(self, prev: np.ndarray, curr: np.ndarray) -> FlowField:
        return self._compute_classical(prev, curr)

    def compute_sparse(self, prev: np.ndarray, curr: np.ndarray, points: np.ndarray) -> SparseFlow:
        lk = LucasKanadeFlow()
        return lk.compute_sparse(prev, curr, points)


class OpticalFlowEvaluator:
    @staticmethod
    def compute_epe(gt_flow: FlowField, pred_flow: FlowField) -> float:
        epe = np.sqrt((gt_flow.u - pred_flow.u)**2 + (gt_flow.v - pred_flow.v)**2)
        return float(np.mean(epe))

    @staticmethod
    def compute_fl_score(gt_flow: FlowField, pred_flow: FlowField, threshold: float = 3.0) -> float:
        epe = np.sqrt((gt_flow.u - pred_flow.u)**2 + (gt_flow.v - pred_flow.v)**2)
        fl = np.mean(epe > threshold)
        return float(fl)

    @staticmethod
    def compute_aae(gt_flow: FlowField, pred_flow: FlowField) -> float:
        gt_angle = gt_flow.angle
        pred_angle = pred_flow.angle
        angle_diff = np.abs(gt_angle - pred_angle)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        return float(np.mean(angle_diff))

    @staticmethod
    def compute_outlier_rate(gt_flow: FlowField, pred_flow: FlowField, threshold: float = 3.0) -> float:
        epe = np.sqrt((gt_flow.u - pred_flow.u)**2 + (gt_flow.v - pred_flow.v)**2)
        return float(np.mean(epe > threshold))

    @staticmethod
    def visualize_flow(flow: FlowField, max_radius: float = 10.0) -> np.ndarray:
        h, w = flow.u.shape
        mag = flow.magnitude
        ang = flow.angle

        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(mag / max_radius * 255, 0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    @staticmethod
    def warp_image(image: np.ndarray, flow: FlowField) -> np.ndarray:
        h, w = image.shape[:2]
        map_x = np.mgrid[0:h, 0:w][1] + flow.u
        map_y = np.mgrid[0:h, 0:w][0] + flow.v
        return cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)


class OpticalFlowVisualizer:
    @staticmethod
    def draw_flow_arrows(image: np.ndarray, flow: FlowField, step: int = 16, color: Tuple = (0, 255, 0)) -> np.ndarray:
        vis = image.copy()
        h, w = image.shape[:2]

        for y in range(0, h, step):
            for x in range(0, w, step):
                u, v = flow.u[y, x], flow.v[y, x]
                if abs(u) > 0.5 or abs(v) > 0.5:
                    end_x = int(x + u)
                    end_y = int(y + v)
                    cv2.arrowedLine(vis, (x, y), (end_x, end_y), color, 1, tipLength=0.3)

        return vis

    @staticmethod
    def draw_flow_hsv(image: np.ndarray, flow: FlowField) -> np.ndarray:
        return OpticalFlowEvaluator.visualize_flow(flow)

    @staticmethod
    def draw_flow_grid(image: np.ndarray, flow: FlowField, step: int = 10) -> np.ndarray:
        vis = image.copy()
        h, w = image.shape[:2]

        for y in range(0, h, step):
            for x in range(0, w, step):
                u, v = flow.u[y, x], flow.v[y, x]
                end_x = int(x + u)
                end_y = int(y + v)
                cv2.line(vis, (x, y), (end_x, end_y), (0, 255, 0), 1)

        return vis

    @staticmethod
    def create_flow_visualization(flow: FlowField, method: str = "hsv") -> np.ndarray:
        if method == "hsv":
            return OpticalFlowEvaluator.visualize_flow(flow)
        elif method == "arrows":
            h, w = flow.u.shape
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            return OpticalFlowVisualizer.draw_flow_arrows(blank, flow)
        elif method == "grid":
            h, w = flow.u.shape
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            return OpticalFlowVisualizer.draw_flow_grid(blank, flow)
        return OpticalFlowEvaluator.visualize_flow(flow)


class TemporalFlowFilter:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)

    def filter(self, flow: FlowField) -> FlowField:
        self.buffer.append(flow)
        if len(self.buffer) < 2:
            return flow

        u_sum = np.zeros_like(flow.u)
        v_sum = np.zeros_like(flow.v)
        weights = np.zeros_like(flow.u)

        for i, f in enumerate(self.buffer):
            weight = (i + 1) / len(self.buffer)
            u_sum += f.u * weight
            v_sum += f.v * weight
            weights += weight

        return FlowField(
            u=u_sum / weights,
            v=v_sum / weights,
        )


class FlowSegmentation:
    @staticmethod
    def segment_by_motion(flow: FlowField, threshold: float = 2.0) -> np.ndarray:
        mag = flow.magnitude
        return (mag > threshold).astype(np.uint8)

    @staticmethod
    def segment_by_direction(flow: FlowField, num_bins: int = 8) -> np.ndarray:
        ang = flow.angle
        bins = np.floor((ang + np.pi) / (2 * np.pi) * num_bins).astype(np.int32)
        return np.clip(bins, 0, num_bins - 1)

    @staticmethod
    def find_motion_boundaries(flow: FlowField, threshold: float = 1.0) -> np.ndarray:
        u_grad_x = cv2.Sobel(flow.u, cv2.CV_32F, 1, 0, ksize=3)
        u_grad_y = cv2.Sobel(flow.u, cv2.CV_32F, 0, 1, ksize=3)
        v_grad_x = cv2.Sobel(flow.v, cv2.CV_32F, 1, 0, ksize=3)
        v_grad_y = cv2.Sobel(flow.v, cv2.CV_32F, 0, 1, ksize=3)

        grad_mag = np.sqrt(u_grad_x**2 + u_grad_y**2 + v_grad_x**2 + v_grad_y**2)
        return (grad_mag > threshold).astype(np.uint8)

    @staticmethod
    def cluster_flow(flow: FlowField, n_clusters: int = 5) -> np.ndarray:
        h, w = flow.u.shape
        features = np.column_stack([flow.u.flatten(), flow.v.flatten()])

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        return labels.reshape(h, w)


class OpticalFlowVideoProcessor:
    def __init__(
        self,
        algorithm: OpticalFlowAlgorithm = OpticalFlowAlgorithm.FARNEBACK,
        **kwargs,
    ):
        self.algorithm = algorithm
        self.flow_estimator = self._create_estimator(algorithm, **kwargs)
        self.temporal_filter = TemporalFlowFilter()

    def _create_estimator(self, algorithm: OpticalFlowAlgorithm, **kwargs) -> BaseOpticalFlow:
        if algorithm == OpticalFlowAlgorithm.FARNEBACK:
            return FarnebackFlow(**kwargs)
        elif algorithm == OpticalFlowAlgorithm.LUCAS_KANADE:
            return LucasKanadeFlow(**kwargs)
        elif algorithm == OpticalFlowAlgorithm.DEEPFLOW:
            return DeepFlow(**kwargs)
        elif algorithm == OpticalFlowAlgorithm.PWC_NET:
            return PWCNetFlow(**kwargs)
        elif algorithm == OpticalFlowAlgorithm.RAFT:
            return RAFTFlow(**kwargs)
        elif algorithm == OpticalFlowAlgorithm.FLOWNET:
            return FlowNetFlow(**kwargs)
        return FarnebackFlow(**kwargs)

    def process_video(self, frames: List[np.ndarray]) -> List[FlowField]:
        flows = []
        for i in range(1, len(frames)):
            flow = self.flow_estimator.compute(frames[i-1], frames[i])
            flow = self.temporal_filter.filter(flow)
            flows.append(flow)
        return flows

    def process_video_sparse(self, frames: List[np.ndarray], points: np.ndarray) -> List[SparseFlow]:
        flows = []
        for i in range(1, len(frames)):
            flow = self.flow_estimator.compute_sparse(frames[i-1], frames[i], points)
            flows.append(flow)
        return flows

    def compute_flow_statistics(self, flows: List[FlowField]) -> Dict[str, float]:
        if not flows:
            return {}

        mags = [f.magnitude for f in flows]
        angles = [f.angle for f in flows]

        return {
            'mean_magnitude': float(np.mean([np.mean(m) for m in mags])),
            'max_magnitude': float(np.max([np.max(m) for m in mags])),
            'mean_angle': float(np.mean([np.mean(a) for a in angles])),
            'flow_coherence': float(np.mean([np.std(a) for a in angles])),
        }
