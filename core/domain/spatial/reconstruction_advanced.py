from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time


class ReconstructionMethod(Enum):
    POISSON = "poisson"
    DIBR = "dibr"
    MARCHING_CUBES = "marching_cubes"
    DUAL_CONTOURING = "dual_contouring"
    VOXEL_CARVING = "voxel_carving"
    TSDF_FUSION = "tsdf_fusion"
    NEURAL_RADIANCE = "neural_radiance"
    GAUSSIAN_SPLATTING = "gaussian_splatting"


@dataclass(frozen=True, slots=True)
class CameraPose:
    camera_id: str
    rotation: np.ndarray
    translation: np.ndarray
    intrinsics: np.ndarray
    timestamp: float = 0.0

    @property
    def projection_matrix(self) -> np.ndarray:
        Rt = np.hstack([self.rotation, self.translation.reshape(3, 1)])
        return self.intrinsics @ Rt


@dataclass(frozen=True, slots=True)
class Point3D:
    position: np.ndarray
    color: np.ndarray
    normal: np.ndarray
    track_id: int
    observations: Dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None


class FeatureExtractor:
    def __init__(self, detector_type: str = "orb", n_features: int = 2000):
        self.detector_type = detector_type
        self.n_features = n_features
        self.detector = self._create_detector()

    def _create_detector(self):
        if self.detector_type == "orb":
            return cv2.ORB_create(nfeatures=self.n_features)
        elif self.detector_type == "sift":
            return cv2.SIFT_create(nfeatures=self.n_features)
        elif self.detector_type == "surf":
            return cv2.xfeatures2d.SURF_create()
        else:
            return cv2.ORB_create(nfeatures=self.n_features)

    def extract(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp, desc = self.detector.detectAndCompute(gray, None)
        return kp, desc

    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        if self.detector_type == "orb":
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(desc1, desc2)
        return sorted(matches, key=lambda x: x.distance)


class StructureFromMotion:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        min_matches: int = 50,
        ransac_threshold: float = 3.0,
    ):
        self.feature_extractor = feature_extractor
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.views: Dict[str, Dict] = {}
        self.tracks: Dict[int, Dict] = {}
        self.next_track_id = 0

    def add_view(self, image: np.ndarray, camera_id: str, pose: CameraPose = None):
        kp, desc = self.feature_extractor.extract(image)

        view = {
            'camera_id': camera_id,
            'image': image,
            'keypoints': kp,
            'descriptors': desc,
            'pose': pose,
            'points_2d': np.array([p.pt for p in kp]),
        }
        self.views[camera_id] = view

        if pose is None and len(self.views) > 1:
            self._estimate_pose(camera_id)

    def _estimate_pose(self, camera_id: str):
        new_view = self.views[camera_id]
        best_view = None
        best_matches = []

        for vid, view in self.views.items():
            if vid == camera_id or view['descriptors'] is None:
                continue
            matches = self.feature_extractor.match(view['descriptors'], new_view['descriptors'])
            if len(matches) > len(best_matches) and len(matches) >= self.min_matches:
                best_matches = matches
                best_view = view

        if best_view is None or len(best_matches) < self.min_matches:
            return

        pts1 = best_view['points_2d'][[m.queryIdx for m in best_matches]]
        pts2 = new_view['points_2d'][[m.trainIdx for m in best_matches]]

        E, mask = cv2.findEssentialMat(pts1, pts2, best_view['pose'].intrinsics if best_view['pose'] else np.eye(3), cv2.RANSAC, 0.999, self.ransac_threshold)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, best_view['pose'].intrinsics if best_view['pose'] else np.eye(3), mask=mask)

        if best_view['pose'] is not None:
            new_pose = CameraPose(
                camera_id=camera_id,
                rotation=best_view['pose'].rotation @ R,
                translation=best_view['pose'].rotation @ t + best_view['pose'].translation,
                intrinsics=new_view['pose'].intrinsics if new_view['pose'] else np.eye(3),
            )
            new_view['pose'] = new_pose

        self._triangulate_points(best_view, new_view, best_matches, mask.ravel())

    def _triangulate_points(self, view1: Dict, view2: Dict, matches: List, mask: np.ndarray):
        if view1['pose'] is None or view2['pose'] is None:
            return

        pts1 = view1['points_2d'][[m.queryIdx for m in matches]]
        pts2 = view2['points_2d'][[m.trainIdx for m in matches]]
        pts1 = pts1[mask == 1]
        pts2 = pts2[mask == 1]

        if len(pts1) < 4:
            return

        P1 = view1['pose'].projection_matrix
        P2 = view2['pose'].projection_matrix

        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]

        for i, pt in enumerate(points_3d.T):
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = {
                'position': pt,
                'observations': {
                    view1['camera_id']: pts1[i],
                    view2['camera_id']: pts2[i],
                },
                'color': view1['image'][int(pts1[i, 1]), int(pts1[i, 0])],
            }

    def get_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        positions = []
        colors = []
        for track in self.tracks.values():
            positions.append(track['position'])
            colors.append(track['color'])
        return np.array(positions), np.array(colors)

    def get_cameras(self) -> Dict[str, CameraPose]:
        return {vid: view['pose'] for vid, view in self.views.items() if view['pose'] is not None}


class MultiViewStereo:
    def __init__(
        self,
        patch_size: int = 11,
        window_size: int = 5,
        min_corr: float = 0.8,
    ):
        self.patch_size = patch_size
        self.window_size = window_size
        self.min_corr = min_corr

    def compute_depth(
        self,
        images: List[np.ndarray],
        cameras: Dict[str, CameraPose],
        reference: str,
    ) -> np.ndarray:
        ref_img = images[cameras[reference].camera_id] if isinstance(cameras[reference].camera_id, int) else images[0]
        h, w = ref_img.shape[:2]
        depth = np.zeros((h, w), dtype=np.float32)
        best_corr = np.zeros((h, w), dtype=np.float32)

        for cam_id, cam in cameras.items():
            if cam_id == reference:
                continue

            other_idx = cameras[cam_id].camera_id if isinstance(cam_id, str) else 0
            if isinstance(cam_id, str):
                other_idx = 0
                for i, c in enumerate(cameras.values()):
                    if c.camera_id == cam_id:
                        other_idx = i
                        break

            if other_idx >= len(images):
                continue

            other_img = images[other_idx]
            depth_map, corr_map = self._stereo_match(ref_img, other_img, cameras[reference], cam)

            mask = corr_map > self.min_corr
            mask &= (corr_map > best_corr)
            depth[mask] = depth_map[mask]
            best_corr[mask] = corr_map[mask]

        return depth

    def _stereo_match(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        cam1: CameraPose,
        cam2: CameraPose,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img1.shape[:2]
        depth = np.zeros((h, w), dtype=np.float32)
        corr = np.zeros((h, w), dtype=np.float32)

        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2

        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=9,
            P1=8 * 3 * 9**2,
            P2=32 * 3 * 9**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )

        disp = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

        fx = cam1.intrinsics[0, 0]
        baseline = np.linalg.norm(cam2.translation - cam1.translation)

        depth[disp > 0] = fx * baseline / disp[disp > 0]
        corr = (disp > 0).astype(np.float32)

        return depth, corr


class TSDFFusion:
    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (512, 512, 512),
        voxel_size: float = 0.01,
        truncation: float = 0.04,
    ):
        self.volume_size = np.array(volume_size)
        self.voxel_size = voxel_size
        self.truncation = truncation
        self.origin = np.array([0.0, 0.0, 0.0])
        self.tsdf = np.zeros(volume_size, dtype=np.float32)
        self.weights = np.zeros(volume_size, dtype=np.float32)
        self.colors = np.zeros((*volume_size, 3), dtype=np.uint8)

    def integrate(self, depth: np.ndarray, color: np.ndarray, pose: CameraPose):
        h, w = depth.shape
        fx, fy = pose.intrinsics[0, 0], pose.intrinsics[1, 1]
        cx, cy = pose.intrinsics[0, 2], pose.intrinsics[1, 2]

        for v in range(h):
            for u in range(w):
                d = depth[v, u]
                if d <= 0 or d > 5.0:
                    continue

                x = (u - cx) * d / fx
                y = (v - cy) * d / fy
                z = d
                point_cam = np.array([x, y, z, 1])
                point_world = pose.rotation @ point_cam[:3] + pose.translation

                voxel = np.floor((point_world - self.origin) / self.voxel_size).astype(int)
                if np.any(voxel < 0) or np.any(voxel >= self.volume_size):
                    continue

                voxel_center = (voxel + 0.5) * self.voxel_size + self.origin
                dist = np.linalg.norm(point_world - voxel_center)
                tsdf_val = dist / self.truncation
                tsdf_val = np.clip(tsdf_val, -1, 1)

                idx = tuple(voxel)
                w = self.weights[idx]
                new_w = w + 1
                self.tsdf[idx] = (self.tsdf[idx] * w + tsdf_val) / new_w
                self.weights[idx] = new_w

                if color is not None:
                    c = color[v, u]
                    self.colors[idx] = (self.colors[idx] * w + c) / new_w

    def extract_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        vertices = []
        faces = []
        for x in range(self.volume_size[0] - 1):
            for y in range(self.volume_size[1] - 1):
                for z in range(self.volume_size[2] - 1):
                    cube = [self.tsdf[x + dx, y + dy, z + dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
                    if max(cube) > 0 and min(cube) < 0:
                        self._marching_cubes(x, y, z, vertices, faces)
        return np.array(vertices), np.array(faces)

    def _marching_cubes(self, x: int, y: int, z: int, vertices: List, faces: List):
        edge_table = [0] * 256
        tri_table = [] * 256


class NeuralRadianceField:
    def __init__(
        self,
        num_layers: int = 8,
        hidden_dim: int = 256,
        num_freq: int = 10,
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_freq = num_freq
        self.network = self._build_network()

    def _build_network(self):
        return None

    def encode_position(self, x: np.ndarray) -> np.ndarray:
        encodings = [x]
        for i in range(self.num_freq):
            for fn in [np.sin, np.cos]:
                encodings.append(fn(2.0 ** i * np.pi * x))
        return np.concatenate(encodings, axis=-1)

    def query(self, points: np.ndarray, directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros((len(points), 1)), np.zeros((len(points), 3))

    def render_ray(self, origin: np.ndarray, direction: np.ndarray, near: float, far: float, num_samples: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        t_vals = np.linspace(near, far, num_samples)
        points = origin + t_vals[:, None] * direction
        densities, colors = self.query(points, np.tile(direction, (len(points), 1)))

        delta = np.diff(t_vals)
        delta = np.concatenate([delta, [1e10]])
        alpha = 1 - np.exp(-densities * delta)
        weights = alpha * np.cumprod(np.concatenate([[1], 1 - alpha + 1e-10]))[:-1]

        rgb = np.sum(weights[:, None] * colors, axis=0)
        depth = np.sum(weights * t_vals)
        return rgb, depth

    def train(self, images: List[np.ndarray], poses: List[CameraPose], intrinsics: np.ndarray, epochs: int = 1000):
        pass


class GaussianSplatting:
    def __init__(
        self,
        num_gaussians: int = 100000,
        sh_degree: int = 3,
    ):
        self.num_gaussians = num_gaussians
        self.sh_degree = sh_degree
        self.means = np.random.randn(num_gaussians, 3) * 0.1
        self.scales = np.ones((num_gaussians, 3)) * 0.01
        self.rotations = np.tile(np.array([1, 0, 0, 0]), (num_gaussians, 1))
        self.opacities = np.ones(num_gaussians) * 0.1
        self.sh_coeffs = np.random.randn(num_gaussians, (sh_degree + 1) ** 2, 3) * 0.01

    def render(self, camera: CameraPose, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        fx, fy = camera.intrinsics[0, 0], camera.intrinsics[1, 1]
        cx, cy = camera.intrinsics[0, 2], camera.intrinsics[1, 2]

        view_dir = camera.rotation @ np.array([0, 0, 1])
        camera_pos = camera.translation

        depths = np.linalg.norm(self.means - camera_pos, axis=1)
        sorted_idx = np.argsort(depths)

        image = np.zeros((height, width, 3), dtype=np.float32)
        depth_map = np.zeros((height, width), dtype=np.float32)
        alpha_accum = np.zeros((height, width), dtype=np.float32)

        for idx in sorted_idx:
            mean = self.means[idx]
            scale = self.scales[idx]
            rot = self.rotations[idx]
            opacity = self.opacities[idx]

            diff = mean - camera_pos
            proj = camera.intrinsics @ diff
            if proj[2] <= 0:
                continue

            u, v = proj[0] / proj[2], proj[1] / proj[2]
            u, v = int(u), int(v)

            if not (0 <= u < width and 0 <= v < height):
                continue

            alpha = opacity * (1 - alpha_accum[v, u])
            if alpha < 1e-4:
                continue

            image[v, u] = image[v, u] * (1 - alpha) + self._eval_sh(idx, view_dir) * alpha
            depth_map[v, u] = depth_map[v, u] * (1 - alpha) + depths[idx] * alpha
            alpha_accum[v, u] += alpha

            if alpha_accum[v, u] > 0.99:
                continue

        return image, depth_map

    def _eval_sh(self, idx: int, view_dir: np.ndarray) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])

    def train(self, images: List[np.ndarray], cameras: List[CameraPose], epochs: int = 1000):
        pass


class DenseReconstruction:
    def __init__(self):
        self.sfm = StructureFromMotion(FeatureExtractor())
        self.mvs = MultiViewStereo()
        self.tsdf = TSDFFusion()
        self.nerf = NeuralRadianceField()

    def reconstruct(self, images: List[np.ndarray], cameras: Dict[str, CameraPose]):
        for cam_id, cam in cameras.items():
            self.sfm.add_view(images[cam_id] if isinstance(cam_id, int) else images[0], cam_id, cam)

        for cam_id, cam in cameras.items():
            if cam_id in self.sfm.views and self.sfm.views[cam_id]['pose'] is not None:
                depth = self.mvs.compute_depth(images, self.sfm.get_cameras(), cam_id)
                self.tsdf.integrate(depth, images[cam_id] if isinstance(cam_id, int) else images[0], cam)

        vertices, faces = self.tsdf.extract_mesh()
        return Mesh(vertices, faces)


class NeuralRenderer:
    def __init__(self):
        self.nerf = NeuralRadianceField()

    def render_novel_view(self, camera: CameraPose, width: int, height: int) -> np.ndarray:
        image = np.zeros((height, width, 3))
        for v in range(height):
            for u in range(width):
                fx, fy = camera.intrinsics[0, 0], camera.intrinsics[1, 1]
                cx, cy = camera.intrinsics[0, 2], camera.intrinsics[1, 2]
                x = (u - cx) / fx
                y = (v - cy) / fy
                direction = camera.rotation @ np.array([x, y, 1])
                direction = direction / np.linalg.norm(direction)
                origin = camera.translation
                rgb, _ = self.nerf.render_ray(origin, direction, 0.1, 100)
                image[v, u] = rgb
        return image


class MeshProcessor:
    @staticmethod
    def simplify(mesh: Mesh, target_faces: int) -> Mesh:
        return mesh

    @staticmethod
    def smooth(mesh: Mesh, iterations: int = 10, lambda_factor: float = 0.5) -> Mesh:
        vertices = mesh.vertices.copy()
        faces = mesh.faces

        for _ in range(iterations):
            new_vertices = vertices.copy()
            for i in range(len(vertices)):
                neighbors = faces[np.any(faces == i, axis=1)]
                if len(neighbors) > 0:
                    neighbor_verts = np.unique(neighbors.flatten())
                    neighbor_verts = neighbor_verts[neighbor_verts != i]
                    if len(neighbor_verts) > 0:
                        avg = np.mean(vertices[neighbor_verts], axis=0)
                        new_vertices[i] = vertices[i] + lambda_factor * (avg - vertices[i])
            vertices = new_vertices

        return Mesh(vertices, faces, mesh.colors, mesh.normals, mesh.uvs)

    @staticmethod
    def compute_normals(mesh: Mesh) -> np.ndarray:
        vertices = mesh.vertices
        faces = mesh.faces
        normals = np.zeros_like(vertices)

        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            normals[face[0]] += normal
            normals[face[1]] += normal
            normals[face[2]] += normal

        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1
        normals = normals / norms[:, None]
        return normals

    @staticmethod
    def poisson_reconstruction(points: np.ndarray, normals: np.ndarray, depth: int = 8) -> Mesh:
        return Mesh(np.array([]), np.array([]))


@dataclass(frozen=True, slots=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None