import numpy as np


class Spatial3DFusion:
    """Fuses 2D bounding boxes with monocular depth maps for metric 3D estimation."""

    def __init__(self, fov_degrees=60.0, min_depth_m=0.35, max_depth_m=4.5):
        self.fov_rad = np.radians(fov_degrees)
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

    def estimate_spatial_metrics(self, detections, depth_map, image_shape):
        """
        Estimate 3D camera coordinates (X, Y, Z) and physical dimensions for each detection.

        detections: list of dicts with 'bbox': [x1, y1, x2, y2]
        depth_map: 2D numpy array normalized in [0, 1] (MiDaS output)
        image_shape: tuple of (height, width)
        """
        if depth_map is None or not detections:
            return detections

        img_h, img_w = image_shape[:2]
        depth_h, depth_w = depth_map.shape[:2]

        # Camera intrinsic matrix parameters
        fx = (img_w / 2.0) / np.tan(self.fov_rad / 2.0)
        fy = fx
        cx = img_w / 2.0
        cy = img_h / 2.0

        scale_x = depth_w / float(img_w)
        scale_y = depth_h / float(img_h)

        enriched = []
        for det in detections:
            det_copy = dict(det)
            x1, y1, x2, y2 = det["bbox"]

            # Map coordinates to depth map grid
            dx1 = int(np.clip(x1 * scale_x, 0, depth_w - 1))
            dy1 = int(np.clip(y1 * scale_y, 0, depth_h - 1))
            dx2 = int(np.clip(x2 * scale_x, dx1 + 1, depth_w))
            dy2 = int(np.clip(y2 * scale_y, dy1 + 1, depth_h))

            # Sample the central 60% region of the box to avoid background boundary bleed
            roi_w = max(1, dx2 - dx1)
            roi_h = max(1, dy2 - dy1)
            margin_x = int(0.2 * roi_w)
            margin_y = int(0.2 * roi_h)

            crop_x1 = dx1 + margin_x
            crop_y1 = dy1 + margin_y
            crop_x2 = max(crop_x1 + 1, dx2 - margin_x)
            crop_y2 = max(crop_y1 + 1, dy2 - margin_y)

            roi = depth_map[crop_y1:crop_y2, crop_x1:crop_x2]

            if roi.size > 0:
                # Use robust median depth value
                median_disp = float(np.median(roi))
            else:
                median_disp = 0.5

            # Map relative disparity [0, 1] to approximate metric distance [min_depth, max_depth]
            # Higher disparity = closer to camera
            z_metric = self.min_depth_m + (1.0 - np.clip(median_disp, 0.0, 1.0)) * (
                self.max_depth_m - self.min_depth_m
            )

            # Back-project 2D box centroid to 3D camera coordinates
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0

            x_metric = (u - cx) * z_metric / fx
            y_metric = -(v - cy) * z_metric / fy  # Inverted so +Y points upward

            # Estimate physical dimensions
            box_w_px = max(1.0, x2 - x1)
            box_h_px = max(1.0, y2 - y1)
            w_metric = box_w_px * z_metric / fx
            h_metric = box_h_px * z_metric / fy

            distance_m = np.sqrt(x_metric**2 + y_metric**2 + z_metric**2)

            det_copy["spatial_3d"] = {
                "x_m": round(float(x_metric), 2),
                "y_m": round(float(y_metric), 2),
                "z_m": round(float(z_metric), 2),
                "distance_m": round(float(distance_m), 2),
                "width_m": round(float(w_metric), 2),
                "height_m": round(float(h_metric), 2),
            }

            enriched.append(det_copy)

        return enriched


spatial_fuser = Spatial3DFusion()
