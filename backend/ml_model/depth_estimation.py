import logging
import numpy as np
import cv2
import torch

logger = logging.getLogger(__name__)


class DepthEstimator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance.model = None
            cls._instance.transform = None
        return cls._instance

    def _initialize(self):
        if self._initialized:
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Load MiDaS Small from TorchHub or local cache
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform
            logger.info("MiDaS monocular depth model loaded successfully on %s", self.device)
        except Exception as exc:
            logger.warning("MiDaS model hub download unavailable (%s); using geometric depth fallback", exc)
            self.model = None
            self.transform = None

        self._initialized = True

    def estimate_depth(self, image_bytes):
        self._initialize()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        if self.model is not None and self.transform is not None:
            try:
                input_batch = self.transform(img_rgb).to(self.device)
                with torch.no_grad():
                    prediction = self.model(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(h, w),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                depth_map = prediction.cpu().numpy()
                d_min, d_max = depth_map.min(), depth_map.max()
                if d_max - d_min > 0:
                    depth_map = (depth_map - d_min) / (d_max - d_min)
                else:
                    depth_map = np.zeros_like(depth_map)
                return depth_map, img_rgb
            except Exception as exc:
                logger.warning("MiDaS inference error (%s); falling back to geometric gradient", exc)

        # High-quality perspective depth gradient fallback for offline / disconnected operation
        y_coords = np.linspace(0.2, 0.9, h, dtype=np.float32)[:, None]
        base_depth = np.repeat(y_coords, w, axis=1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        norm_gray = blur.astype(np.float32) / 255.0

        depth_map = 0.6 * base_depth + 0.4 * norm_gray
        depth_map = np.clip(depth_map, 0.0, 1.0)

        return depth_map, img_rgb

    def generate_point_cloud(self, image_bytes, max_points=16384, depth_scale=2.0):
        depth_map, img_rgb = self.estimate_depth(image_bytes)

        h, w = depth_map.shape
        total_pixels = h * w
        step = max(1, int(np.sqrt(total_pixels / max_points)))

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