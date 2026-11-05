import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MiDaS is pinned to a release tag rather than master. Its small model pulls a
# second TorchHub repo, which TorchHub would otherwise confirm interactively;
# under gunicorn there is no terminal, so both repos are trusted up front.
MIDAS_REPO = "intel-isl/MiDaS:v3_1"
TRUSTED_HUB_REPOS = ("intel-isl_MiDaS", "rwightman_gen-efficientnet-pytorch")


def _trust_hub_repos(hub_dir):
    os.makedirs(hub_dir, exist_ok=True)
    path = os.path.join(hub_dir, "trusted_list")
    existing = set()
    if os.path.exists(path):
        with open(path) as handle:
            existing = {line.strip() for line in handle}
    missing = [repo for repo in TRUSTED_HUB_REPOS if repo not in existing]
    if missing:
        with open(path, "a") as handle:
            handle.writelines(f"{repo}\n" for repo in missing)


def load_midas(torch):
    """Load MiDaS-small and its transform from TorchHub (or its local cache)."""
    _trust_hub_repos(torch.hub.get_dir())
    model = torch.hub.load(MIDAS_REPO, "MiDaS_small", trust_repo=True)
    transforms = torch.hub.load(MIDAS_REPO, "transforms", trust_repo=True)
    return model, transforms.small_transform


class DepthUnavailableError(RuntimeError):
    """Raised when no depth model is loaded, so no depth map can be produced.

    Depth is optional: it needs PyTorch and the MiDaS weights from TorchHub.
    Callers should skip depth-derived values rather than substitute guesses.
    """


class DepthEstimator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance.model = None
            cls._instance.transform = None
            cls._instance.unavailable_reason = None
        return cls._instance

    def _initialize(self):
        if self._initialized:
            return

        self._initialized = True
        try:
            import torch
        except ImportError:
            self.unavailable_reason = "PyTorch is not installed"
            logger.warning("Depth estimation disabled: %s", self.unavailable_reason)
            return

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model, self.transform = load_midas(torch)
            self.model.to(self.device)
            self.model.eval()
            logger.info("MiDaS monocular depth model loaded successfully on %s", self.device)
        except Exception as exc:
            self.unavailable_reason = f"MiDaS model could not be loaded: {exc}"
            logger.warning("Depth estimation disabled: %s", self.unavailable_reason)
            self.model = None
            self.transform = None

    def estimate_depth(self, image_bytes):
        self._initialize()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        if self.model is None or self.transform is None:
            raise DepthUnavailableError(self.unavailable_reason or "Depth model is not loaded")

        torch = self._torch
        input_batch = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # MiDaS output is relative inverse depth; normalise to [0, 1].
        depth_map = prediction.cpu().numpy()
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 0:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map)
        return depth_map, img_rgb

    def generate_point_cloud(self, image_bytes, max_points=16384, depth_scale=2.0):
        depth_map, img_rgb = self.estimate_depth(image_bytes)

        h, w = depth_map.shape
        total_pixels = h * w
        # Round the sampling step up so the cloud never exceeds the point budget.
        step = max(1, int(np.ceil(np.sqrt(total_pixels / max_points))))

        positions = []
        colors = []

        for y in range(0, h, step):
            for x in range(0, w, step):
                depth = float(depth_map[y, x])
                if depth < 0.05:
                    continue

                px = (x / w - 0.5) * 2.0
                py = -(y / h - 0.5) * 2.0
                pz = depth * depth_scale

                positions.extend([px, py, pz])
                r, g, b = img_rgb[y, x]
                colors.extend([r / 255.0, g / 255.0, b / 255.0])

        point_count = len(positions) // 3

        return {
            "positions": positions,
            "colors": colors,
            "point_count": point_count,
            "original_width": w,
            "original_height": h,
            "step": step,
        }


depth_estimator = DepthEstimator()
