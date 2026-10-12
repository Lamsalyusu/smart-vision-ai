import numpy as np


class CameraIntrinsics:
    """Pinhole camera intrinsic calibration model and coordinate projector."""

    def __init__(self, fov_degrees=60.0, width=640, height=480):
        self.width = width
        self.height = height
        self.fov_rad = np.radians(fov_degrees)
        self.fx = (width / 2.0) / np.tan(self.fov_rad / 2.0)
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0

        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        self.K_inv = np.linalg.inv(self.K)

    def back_project(self, u, v, z_metric):
        """Unproject pixel (u, v) and metric depth Z to 3D camera coordinate frame (X, Y, Z)."""
        pixel_homog = np.array([u, v, 1.0], dtype=np.float32)
        ray = np.dot(self.K_inv, pixel_homog)
        x = ray[0] * z_metric
        y = -ray[1] * z_metric  # Invert Y so camera frame points upwards
        return float(x), float(y), float(z_metric)

    def update_resolution(self, width, height):
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            self.fx = (width / 2.0) / np.tan(self.fov_rad / 2.0)
            self.fy = self.fx
            self.cx = width / 2.0
            self.cy = height / 2.0
            self.K[0, 0] = self.fx
            self.K[1, 1] = self.fy
            self.K[0, 2] = self.cx
            self.K[1, 2] = self.cy
            self.K_inv = np.linalg.inv(self.K)


class Spatial3DFusion:
    """Fuses 2D bounding detections with monocular depth maps for metric 3D perception."""

    def __init__(self, fov_degrees=60.0, min_depth_m=0.35, max_depth_m=4.5):
        self.intrinsics = CameraIntrinsics(fov_degrees=fov_degrees)
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

    def _extract_robust_depth(self, roi):
        """Compute robust depth value using interquartile range (IQR) outlier filtering."""
        if roi.size == 0:
            return 0.5

        flat = roi.flatten()
        q25, q75 = np.percentile(flat, [25, 75])
        iqr = q75 - q25
        valid = flat[(flat >= q25 - 1.5 * iqr) & (flat <= q75 + 1.5 * iqr)]

        if valid.size > 0:
            return float(np.median(valid))
        return float(np.median(flat))

    def estimate_spatial_metrics(self, detections, depth_map, image_shape):
        """
        Estimate 3D camera coordinates (X, Y, Z), distance, and physical dimensions.
        detections: list of dicts with 'bbox': [x1, y1, x2, y2]
        depth_map: 2D numpy array normalized in [0, 1]
        image_shape: tuple of (height, width)
        """
        if depth_map is None or not detections:
            return detections

        img_h, img_w = image_shape[:2]
        self.intrinsics.update_resolution(img_w, img_h)

        depth_h, depth_w = depth_map.shape[:2]
        scale_x = depth_w / float(img_w)
        scale_y = depth_h / float(img_h)

        enriched = []
        for det in detections:
            det_copy = dict(det)
            x1, y1, x2, y2 = det["bbox"]

            dx1 = int(np.clip(x1 * scale_x, 0, depth_w - 1))
            dy1 = int(np.clip(y1 * scale_y, 0, depth_h - 1))
            dx2 = int(np.clip(x2 * scale_x, dx1 + 1, depth_w))
            dy2 = int(np.clip(y2 * scale_y, dy1 + 1, depth_h))

            # Sample central 60% ROI to prevent boundary contamination
            roi_w = max(1, dx2 - dx1)
            roi_h = max(1, dy2 - dy1)
            crop_x1 = dx1 + int(0.2 * roi_w)
            crop_y1 = dy1 + int(0.2 * roi_h)
            crop_x2 = max(crop_x1 + 1, dx2 - int(0.2 * roi_w))
            crop_y2 = max(crop_y1 + 1, dy2 - int(0.2 * roi_h))

            roi = depth_map[crop_y1:crop_y2, crop_x1:crop_x2]
            median_disp = self._extract_robust_depth(roi)

            # Convert normalized relative disparity to approximate physical distance
            z_metric = self.min_depth_m + (1.0 - np.clip(median_disp, 0.0, 1.0)) * (
                self.max_depth_m - self.min_depth_m
            )

            # Unproject centroid to 3D Euclidean coordinates
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            x_m, y_m, z_m = self.intrinsics.back_project(u, v, z_metric)

            # Estimate metric dimensions
            box_w_px = max(1.0, x2 - x1)
            box_h_px = max(1.0, y2 - y1)
            w_metric = (box_w_px * z_metric) / self.intrinsics.fx
            h_metric = (box_h_px * z_metric) / self.intrinsics.fy

            distance_m = np.sqrt(x_m**2 + y_m**2 + z_m**2)
            azimuth_deg = np.degrees(np.arctan2(x_m, z_m))

            det_copy["spatial_3d"] = {
                "x_m": round(float(x_m), 2),
                "y_m": round(float(y_m), 2),
                "z_m": round(float(z_m), 2),
                "distance_m": round(float(distance_m), 2),
                "width_m": round(float(w_metric), 2),
                "height_m": round(float(h_metric), 2),
                "azimuth_deg": round(float(azimuth_deg), 1),
            }

            enriched.append(det_copy)

        return enriched


spatial_fuser = Spatial3DFusion()
