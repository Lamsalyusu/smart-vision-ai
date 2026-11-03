from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import math


@dataclass(frozen=True, slots=True)
class Point2D:
    x: float
    y: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Point2D":
        return cls(float(arr[0]), float(arr[1]))


@dataclass(frozen=True, slots=True)
class Point3D:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Point3D":
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass(frozen=True, slots=True)
class Line2D:
    a: float
    b: float
    c: float

    @classmethod
    def from_points(cls, p1: Point2D, p2: Point2D) -> "Line2D":
        a = p2.y - p1.y
        b = p1.x - p2.x
        c = p2.x * p1.y - p1.x * p2.y
        return cls(a, b, c)

    def distance(self, point: Point2D) -> float:
        return abs(self.a * point.x + self.b * point.y + self.c) / math.sqrt(self.a**2 + self.b**2)


@dataclass(frozen=True, slots=True)
class Circle2D:
    center: Point2D
    radius: float

    def contains(self, point: Point2D) -> bool:
        return (point.x - self.center.x)**2 + (point.y - self.center.y)**2 <= self.radius**2

    def intersect_line(self, line: Line2D) -> List[Point2D]:
        d = line.distance(self.center)
        if d > self.radius:
            return []
        elif abs(d - self.radius) < 1e-9:
            x = (line.b * (line.b * self.center.x - line.a * self.center.y) - line.a * line.c) / (line.a**2 + line.b**2)
            y = (line.a * (-line.b * self.center.x + line.a * self.center.y) - line.b * line.c) / (line.a**2 + line.b**2)
            return [Point2D(x, y)]
        else:
            x0 = (line.b * (line.b * self.center.x - line.a * self.center.y) - line.a * line.c) / (line.a**2 + line.b**2)
            y0 = (line.a * (-line.b * self.center.x + line.a * self.center.y) - line.b * line.c) / (line.a**2 + line.b**2)
            dx = math.sqrt(self.radius**2 - d**2) * line.b / math.sqrt(line.a**2 + line.b**2)
            dy = -math.sqrt(self.radius**2 - d**2) * line.a / math.sqrt(line.a**2 + line.b**2)
            return [Point2D(x0 + dx, y0 + dy), Point2D(x0 - dx, y0 - dy)]


@dataclass(frozen=True, slots=True)
class Polygon2D:
    vertices: List[Point2D]

    def area(self) -> float:
        area = 0.0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        return abs(area) / 2.0

    def centroid(self) -> Point2D:
        cx = cy = 0.0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            cross = self.vertices[i].x * self.vertices[j].y - self.vertices[j].x * self.vertices[i].y
            cx += (self.vertices[i].x + self.vertices[j].x) * cross
            cy += (self.vertices[i].y + self.vertices[j].y) * cross
        area = self.area() * 2
        if area == 0:
            return Point2D(0, 0)
        return Point2D(cx / (3 * area), cy / (3 * area))

    def contains(self, point: Point2D) -> bool:
        inside = False
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            if ((self.vertices[i].y > point.y) != (self.vertices[j].y > point.y)) and \
               (point.x < (self.vertices[j].x - self.vertices[i].x) * (point.y - self.vertices[i].y) / (self.vertices[j].y - self.vertices[i].y) + self.vertices[i].x):
                inside = not inside
        return inside


class GeometryEngine:
    @staticmethod
    def line_intersection(l1: Line2D, l2: Line2D) -> Optional[Point2D]:
        det = l1.a * l2.b - l2.a * l1.b
        if abs(det) < 1e-9:
            return None
        x = (l1.b * l2.c - l2.b * l1.c) / det
        y = (l2.a * l1.c - l1.a * l2.c) / det
        return Point2D(x, y)

    @staticmethod
    def point_line_distance(point: Point2D, line: Line2D) -> float:
        return line.distance(point)

    @staticmethod
    def project_point_on_line(point: Point2D, line: Line2D) -> Point2D:
        denom = line.a**2 + line.b**2
        if denom == 0:
            return point
        x = (line.b * (line.b * point.x - line.a * point.y) - line.a * line.c) / denom
        y = (line.a * (-line.b * point.x + line.a * point.y) - line.b * line.c) / denom
        return Point2D(x, y)

    @staticmethod
    def circle_intersection(c1: Circle2D, c2: Circle2D) -> List[Point2D]:
        dx = c2.center.x - c1.center.x
        dy = c2.center.y - c1.center.y
        d = math.sqrt(dx**2 + dy**2)

        if d > c1.radius + c2.radius:
            return []
        if d < abs(c1.radius - c2.radius):
            return []
        if d == 0 and c1.radius == c2.radius:
            return []

        a = (c1.radius**2 - c2.radius**2 + d**2) / (2 * d)
        h = math.sqrt(max(0, c1.radius**2 - a**2))

        x2 = c1.center.x + a * dx / d
        y2 = c1.center.y + a * dy / d

        rx = -dy * h / d
        ry = dx * h / d

        if h < 1e-9:
            return [Point2D(x2, y2)]
        return [Point2D(x2 + rx, y2 + ry), Point2D(x2 - rx, y2 - ry)]

    @staticmethod
    def polygon_clip(subject: Polygon2D, clip: Polygon2D) -> Polygon2D:
        def inside(p, edge_start, edge_end):
            return (edge_end.x - edge_start.x) * (p.y - edge_start.y) - (edge_end.y - edge_start.y) * (p.x - edge_start.x) >= 0

        def intersection(p1, p2, edge_start, edge_end):
            dc = Point2D(edge_start.x - edge_end.x, edge_start.y - edge_end.y)
            dp = Point2D(p1.x - p2.x, p1.y - p2.y)
            n1 = edge_start.x * edge_end.y - edge_start.y * edge_end.x
            n2 = p1.x * p2.y - p1.y * p2.x
            denom = dc.x * dp.y - dc.y * dp.x
            if denom == 0:
                return Point2D(0, 0)
            return Point2D((n1 * dp.x - n2 * dc.x) / denom, (n1 * dp.y - n2 * dc.y) / denom)

        output = subject.vertices[:]
        for i in range(len(clip.vertices)):
            edge_start = clip.vertices[i]
            edge_end = clip.vertices[(i + 1) % len(clip.vertices)]
            input_list = output
            output = []
            if not input_list:
                break
            s = input_list[-1]
            for e in input_list:
                if inside(e, edge_start, edge_end):
                    if not inside(s, edge_start, edge_end):
                        output.append(intersection(s, e, edge_start, edge_end))
                    output.append(e)
                elif inside(s, edge_start, edge_end):
                    output.append(intersection(s, e, edge_start, edge_end))
                s = e

        return Polygon2D(output)


class Transform2D:
    def __init__(self, matrix: np.ndarray = None):
        if matrix is None:
            self.matrix = np.eye(3, dtype=np.float64)
        else:
            self.matrix = matrix.astype(np.float64)

    @classmethod
    def translation(cls, tx: float, ty: float) -> "Transform2D":
        m = np.eye(3)
        m[0, 2] = tx
        m[1, 2] = ty
        return cls(m)

    @classmethod
    def rotation(cls, angle: float) -> "Transform2D":
        c, s = math.cos(angle), math.sin(angle)
        m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        return cls(m)

    @classmethod
    def scale(cls, sx: float, sy: float) -> "Transform2D":
        m = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        return cls(m)

    @classmethod
    def shear(cls, shx: float, shy: float) -> "Transform2D":
        m = np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]], dtype=np.float64)
        return cls(m)

    def apply(self, point: Point2D) -> Point2D:
        p = np.array([point.x, point.y, 1.0])
        result = self.matrix @ p
        return Point2D(result[0] / result[2], result[1] / result[2])

    def apply_array(self, points: np.ndarray) -> np.ndarray:
        ones = np.ones((points.shape[0], 1))
        homogeneous = np.hstack([points, ones])
        result = (self.matrix @ homogeneous.T).T
        return result[:, :2] / result[:, 2:3]

    def compose(self, other: "Transform2D") -> "Transform2D":
        return Transform2D(self.matrix @ other.matrix)

    def inverse(self) -> "Transform2D":
        return Transform2D(np.linalg.inv(self.matrix))

    def decompose(self) -> Tuple[float, float, float, float, float]:
        tx = self.matrix[0, 2]
        ty = self.matrix[1, 2]
        sx = math.sqrt(self.matrix[0, 0]**2 + self.matrix[1, 0]**2)
        sy = math.sqrt(self.matrix[0, 1]**2 + self.matrix[1, 1]**2)
        angle = math.atan2(self.matrix[1, 0], self.matrix[0, 0])
        return tx, ty, sx, sy, angle


class Transform3D:
    def __init__(self, matrix: np.ndarray = None):
        if matrix is None:
            self.matrix = np.eye(4, dtype=np.float64)
        else:
            self.matrix = matrix.astype(np.float64)

    @classmethod
    def translation(cls, tx: float, ty: float, tz: float) -> "Transform3D":
        m = np.eye(4)
        m[0, 3] = tx
        m[1, 3] = ty
        m[2, 3] = tz
        return cls(m)

    @classmethod
    def rotation_x(cls, angle: float) -> "Transform3D":
        c, s = math.cos(angle), math.sin(angle)
        m = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float64)
        return cls(m)

    @classmethod
    def rotation_y(cls, angle: float) -> "Transform3D":
        c, s = math.cos(angle), math.sin(angle)
        m = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float64)
        return cls(m)

    @classmethod
    def rotation_z(cls, angle: float) -> "Transform3D":
        c, s = math.cos(angle), math.sin(angle)
        m = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        return cls(m)

    @classmethod
    def rotation_axis(cls, axis: np.ndarray, angle: float) -> "Transform3D":
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c, s = math.cos(angle), math.sin(angle)
        C = 1 - c
        m = np.array([
            [x*x*C + c, x*y*C - z*s, x*z*C + y*s, 0],
            [y*x*C + z*s, y*y*C + c, y*z*C - x*s, 0],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        return cls(m)

    @classmethod
    def scale(cls, sx: float, sy: float, sz: float) -> "Transform3D":
        m = np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]], dtype=np.float64)
        return cls(m)

    def apply(self, point: Point3D) -> Point3D:
        p = np.array([point.x, point.y, point.z, 1.0])
        result = self.matrix @ p
        return Point3D(result[0] / result[3], result[1] / result[3], result[2] / result[3])

    def apply_array(self, points: np.ndarray) -> np.ndarray:
        ones = np.ones((points.shape[0], 1))
        homogeneous = np.hstack([points, ones])
        result = (self.matrix @ homogeneous.T).T
        return result[:, :3] / result[:, 3:4]

    def compose(self, other: "Transform3D") -> "Transform3D":
        return Transform3D(self.matrix @ other.matrix)

    def inverse(self) -> "Transform3D":
        return Transform3D(np.linalg.inv(self.matrix))

    def decompose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        translation = self.matrix[:3, 3]
        rotation = self.matrix[:3, :3]
        scale = np.array([
            np.linalg.norm(rotation[:, 0]),
            np.linalg.norm(rotation[:, 1]),
            np.linalg.norm(rotation[:, 2])
        ])
        rotation = rotation / scale
        return translation, rotation, scale


class CameraModel:
    def __init__(
        self,
        fx: float, fy: float,
        cx: float, cy: float,
        width: int, height: int,
        dist_coeffs: np.ndarray = None,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)

    @property
    def intrinsics(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def project(self, point_3d: Point3D) -> Point2D:
        x, y, z = point_3d.x, point_3d.y, point_3d.z
        if z == 0:
            return Point2D(self.cx, self.cy)
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return Point2D(u, v)

    def project_array(self, points_3d: np.ndarray) -> np.ndarray:
        z = points_3d[:, 2]
        z[z == 0] = 1e-6
        u = self.fx * points_3d[:, 0] / z + self.cx
        v = self.fy * points_3d[:, 1] / z + self.cy
        return np.column_stack([u, v])

    def unproject(self, point_2d: Point2D, depth: float) -> Point3D:
        x = (point_2d.x - self.cx) * depth / self.fx
        y = (point_2d.y - self.cy) * depth / self.fy
        return Point3D(x, y, depth)

    def undistort(self, point_2d: Point2D) -> Point2D:
        if np.all(self.dist_coeffs == 0):
            return point_2d
        u, v = point_2d.x, point_2d.y
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        k1, k2, p1, p2, k3 = self.dist_coeffs[:5]
        radial = 1 + k1*r2 + k2*r4 + k3*r6
        x_undist = x * radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_undist = y * radial + p1*(r2 + 2*y*y) + 2*p2*x*y
        return Point2D(x_undist * self.fx + self.cx, y_undist * self.fy + self.cy)

    def distort(self, point_2d: Point2D) -> Point2D:
        return self.undistort(point_2d)


class EpipolarGeometry:
    @staticmethod
    def fundamental_matrix(pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
        return F, mask

    @staticmethod
    def essential_matrix(pts1: np.ndarray, pts2: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E, mask = cv2.findEssentialMat(pts1, pts2, K1, cv2.RANSAC, 0.999, 1.0)
        return E, mask

    @staticmethod
    def decompose_essential(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        U, _, Vt = np.linalg.svd(E)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        solutions = []
        for R in [U @ W @ Vt, U @ W.T @ Vt]:
            if np.linalg.det(R) < 0:
                R = -R
            t = U[:, 2]
            solutions.append((R, t))
            solutions.append((R, -t))
        return solutions

    @staticmethod
    def triangulate(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T

    @staticmethod
    def rectify(K1: np.ndarray, K2: np.ndarray, R: np.ndarray, t: np.ndarray, img_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, None, K2, None, img_shape, R, t, alpha=0)
        return R1, R2, P1, P2


class Homography:
    @staticmethod
    def compute(pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        return H, mask

    @staticmethod
    def apply(H: np.ndarray, points: np.ndarray) -> np.ndarray:
        ones = np.ones((points.shape[0], 1))
        homogeneous = np.hstack([points, ones])
        result = (H @ homogeneous.T).T
        return result[:, :2] / result[:, 2:3]

    @staticmethod
    def decompose(H: np.ndarray, K: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
        return list(zip(Rs, Ts, Ns))[:num]


class BundleAdjustment:
    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(
        self,
        cameras: List[Dict],
        points_3d: Dict[int, np.ndarray],
        observations: List[Tuple[int, int, np.ndarray]],
        camera_params: int = 6,
        point_params: int = 3,
    ) -> Tuple[Dict, Dict]:
        n_cameras = len(cameras)
        n_points = len(points_3d)
        n_obs = len(observations)

        camera_indices = np.array([obs[0] for obs in observations])
        point_indices = np.array([obs[1] for obs in observations])
        points_2d = np.array([obs[2] for obs in observations])

        camera_params_arr = np.zeros((n_cameras, camera_params))
        for i, cam in enumerate(cameras):
            camera_params_arr[i] = cam['params']

        point_params_arr = np.zeros((n_points, point_params))
        for i, (pid, pt) in enumerate(points_3d.items()):
            point_params_arr[i] = pt

        for iteration in range(self.max_iterations):
            residuals = self._compute_residuals(
                camera_params_arr, point_params_arr,
                camera_indices, point_indices, points_2d, cameras
            )

            if np.linalg.norm(residuals) < self.tolerance:
                break

            J = self._compute_jacobian(
                camera_params_arr, point_params_arr,
                camera_indices, point_indices, cameras
            )

            try:
                dx = np.linalg.solve(J.T @ J + 1e-6 * np.eye(J.shape[1]), -J.T @ residuals)
            except np.linalg.LinAlgError:
                break

            camera_params_arr = camera_params_arr + dx[:n_cameras * camera_params].reshape(n_cameras, camera_params)
            point_params_arr = point_params_arr + dx[n_cameras * camera_params:].reshape(n_points, point_params)

        opt_cameras = {i: camera_params_arr[i] for i in range(n_cameras)}
        opt_points = {pid: point_params_arr[i] for i, pid in enumerate(points_3d.keys())}
        return opt_cameras, opt_points

    def _compute_residuals(self, camera_params, point_params, cam_indices, pt_indices, points_2d, cameras):
        residuals = []
        for i in range(len(cam_indices)):
            cam_idx = cam_indices[i]
            pt_idx = pt_indices[i]
            cam = cameras[cam_idx]
            K = cam['intrinsics']
            R = camera_params[cam_idx, :3].reshape(3, 3)
            t = camera_params[cam_idx, 3:]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            pt_3d = np.append(point_params[pt_idx], 1)
            proj = P @ pt_3d
            proj = proj[:2] / proj[2]
            residuals.append(proj - points_2d[i])
        return np.array(residuals).flatten()

    def _compute_jacobian(self, camera_params, point_params, cam_indices, pt_indices, cameras):
        n_cameras = len(cameras)
        n_points = point_params.shape[0]
        n_obs = len(cam_indices)
        n_cam_params = 6
        n_pt_params = 3

        J = np.zeros((2 * n_obs, n_cameras * n_cam_params + n_points * n_pt_params))

        for i in range(n_obs):
            cam_idx = cam_indices[i]
            pt_idx = pt_indices[i]
            pt_3d = np.append(point_params[pt_idx], 1)
            K = cameras[cam_idx]['intrinsics']
            R = camera_params[cam_idx, :3].reshape(3, 3)
            t = camera_params[cam_idx, 3:]
            P = K @ np.hstack([R, t.reshape(3, 1)])

            proj = P @ pt_3d
            z = proj[2]
            proj_2d = proj[:2] / z

            dP_dR = np.zeros((3, 4, 3))
            dP_dt = np.zeros((3, 4, 3))

            for j in range(3):
                for k in range(3):
                    dP_dR[:, j, k] = K[j, :] @ R[:, k]

            for j in range(3):
                dP_dt[j, :, j] = K[j, j]

            dproj_dpt = (P[:2] - proj_2d.reshape(2, 1) * P[2:3]) / z

            pt_params_start = n_cameras * n_cam_params + pt_idx * n_pt_params
            J[2*i:2*i+2, pt_params_start:pt_params_start+n_pt_params] = dproj_dpt @ R

            cam_params_start = cam_idx * n_cam_params
            J[2*i:2*i+2, cam_params_start:cam_params_start+3] = dproj_dpt @ (dP_dR @ pt_3d)[:, :3]
            J[2*i:2*i+2, cam_params_start+3:cam_params_start+6] = dproj_dpt @ dP_dt

        return J
