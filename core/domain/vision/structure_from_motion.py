from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict
import heapq


class SfMMethod(Enum):
    INCREMENTAL = "incremental"
    GLOBAL = "global"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"


class FeatureMatcherType(Enum):
    EXHAUSTIVE = "exhaustive"
    VOCAB_TREE = "vocab_tree"
    SPATIAL = "spatial"
    SEQUENTIAL = "sequential"
    CLUSTER = "cluster"


class BundleAdjustmentType(Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    SLIDING_WINDOW = "sliding_window"
    HIERARCHICAL = "hierarchical"


@dataclass(frozen=True, slots=True)
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray
    pose: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None

    def project(self, point_3d: np.ndarray) -> np.ndarray:
        if self.pose is None:
            raise ValueError("Camera pose not set")
        point_cam = self.pose[:3, :3] @ point_3d + self.pose[:3, 3]
        if point_cam[2] <= 0:
            return np.array([-1, -1])
        u = self.intrinsics[0, 0] * point_cam[0] / point_cam[2] + self.intrinsics[0, 2]
        v = self.intrinsics[1, 1] * point_cam[1] / point_cam[2] + self.intrinsics[1, 2]
        return np.array([u, v])

    def unproject(self, point_2d: np.ndarray, depth: float) -> np.ndarray:
        if self.pose is None:
            raise ValueError("Camera pose not set")
        x = (point_2d[0] - self.intrinsics[0, 2]) * depth / self.intrinsics[0, 0]
        y = (point_2d[1] - self.intrinsics[1, 2]) * depth / self.intrinsics[1, 1]
        point_cam = np.array([x, y, depth])
        return self.pose[:3, :3].T @ (point_cam - self.pose[:3, 3])


@dataclass(frozen=True, slots=True)
class Image:
    id: int
    camera_id: int
    path: str
    keypoints: List[Any] = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class Match:
    image_id1: int
    image_id2: int
    keypoint_idx1: int
    keypoint_idx2: int
    distance: float


@dataclass(frozen=True, slots=True)
class Track:
    id: int
    observations: Dict[int, Tuple[int, np.ndarray]]
    point_3d: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    reprojection_error: float = 0.0


@dataclass(frozen=True, slots=True)
class Reconstruction:
    cameras: Dict[int, Camera]
    images: Dict[int, Image]
    tracks: Dict[int, Track]
    points_3d: Dict[int, np.ndarray]
    colors_3d: Dict[int, np.ndarray]


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image: np.ndarray) -> Tuple[List[Any], Optional[np.ndarray]]:
        pass


class SIFTExtractor(FeatureExtractor):
    def __init__(self, nfeatures: int = 0, n_octave_layers: int = 3):
        self.detector = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=n_octave_layers)

    def extract(self, image: np.ndarray) -> Tuple[List[Any], Optional[np.ndarray]]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp, desc = self.detector.detectAndCompute(gray, None)
        return kp if kp else [], desc


class ORBExtractor(FeatureExtractor):
    def __init__(self, nfeatures: int = 5000):
        self.detector = cv2.ORB_create(nfeatures=nfeatures)

    def extract(self, image: np.ndarray) -> Tuple[List[Any], Optional[np.ndarray]]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp, desc = self.detector.detectAndCompute(gray, None)
        return kp if kp else [], desc


class SuperPointExtractor(FeatureExtractor):
    def __init__(self, model_path: str = "superpoint.pth", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def extract(self, image: np.ndarray) -> Tuple[List[Any], Optional[np.ndarray]]:
        if self._model is None:
            try:
                import torch
                self._model = torch.jit.load(self.model_path)
                self._model.eval()
            except:
                return [], None

        import torch
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
        if self.device != "cpu" and torch.cuda.is_available():
            tensor = tensor.cuda()

        with torch.no_grad():
            pred = self._model(tensor)

        keypoints = pred['keypoints'][0].cpu().numpy()
        descriptors = pred['descriptors'][0].cpu().numpy()
        scores = pred['scores'][0].cpu().numpy()

        mask = scores > 0.005
        keypoints = keypoints[mask]
        descriptors = descriptors[mask]

        cv_kp = [cv2.KeyPoint(float(k[0]), float(k[1]), 1.0) for k in keypoints]
        return cv_kp, descriptors


class FeatureMatcher(ABC):
    @abstractmethod
    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> List[Match]:
        pass

    @abstractmethod
    def match_all(self, descriptors: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], List[Match]]:
        pass


class ExhaustiveMatcher(FeatureMatcher):
    def __init__(self, ratio_threshold: float = 0.8, cross_check: bool = True):
        self.ratio_threshold = ratio_threshold
        self.cross_check = cross_check
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)

    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> List[Match]:
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        if self.cross_check:
            matches = self.matcher.match(desc1, desc2)
            return [Match(0, 0, m.queryIdx, m.trainIdx, m.distance) for m in matches]
        else:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good = []
            for m, n in matches:
                if m.distance < self.ratio_threshold * n.distance:
                    good.append(Match(0, 0, m.queryIdx, m.trainIdx, m.distance))
            return good

    def match_all(self, descriptors: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], List[Match]]:
        results = {}
        ids = list(descriptors.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                matches = self.match(descriptors[ids[i]], descriptors[ids[j]])
                if matches:
                    results[(ids[i], ids[j])] = matches
        return results


class VocabTreeMatcher(FeatureMatcher):
    def __init__(self, vocab_tree_path: str = "vocab_tree.bin", num_images: int = 1000):
        self.vocab_tree_path = vocab_tree_path
        self.num_images = num_images
        self._tree = None

    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> List[Match]:
        return ExhaustiveMatcher().match(desc1, desc2)

    def match_all(self, descriptors: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], List[Match]]:
        if self._tree is None:
            self._build_tree(descriptors)
        return self._match_with_tree(descriptors)

    def _build_tree(self, descriptors: Dict[int, np.ndarray]):
        pass

    def _match_with_tree(self, descriptors: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], List[Match]]:
        return ExhaustiveMatcher().match_all(descriptors)


class SpatialMatcher(FeatureMatcher):
    def __init__(self, gps_data: Dict[int, Tuple[float, float, float]], max_distance: float = 100.0):
        self.gps_data = gps_data
        self.max_distance = max_distance

    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> List[Match]:
        return ExhaustiveMatcher().match(desc1, desc2)

    def match_all(self, descriptors: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], List[Match]]:
        results = {}
        ids = list(descriptors.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ids[i] in self.gps_data and ids[j] in self.gps_data:
                    pos1 = np.array(self.gps_data[ids[i]])
                    pos2 = np.array(self.gps_data[ids[j]])
                    if np.linalg.norm(pos1 - pos2) < self.max_distance:
                        matches = self.match(descriptors[ids[i]], descriptors[ids[j]])
                        if matches:
                            results[(ids[i], ids[j])] = matches
        return results


class SequentialMatcher(FeatureMatcher):
    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> List[Match]:
        return ExhaustiveMatcher().match(desc1, desc2)

    def match_all(self, descriptors: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], List[Match]]:
        results = {}
        ids = sorted(descriptors.keys())
        for i in range(len(ids)):
            for j in range(i + 1, min(i + self.window_size + 1, len(ids))):
                matches = self.match(descriptors[ids[i]], descriptors[ids[j]])
                if matches:
                    results[(ids[i], ids[j])] = matches
        return results


class ClusterMatcher(FeatureMatcher):
    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters

    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> List[Match]:
        return ExhaustiveMatcher().match(desc1, desc2)

    def match_all(self, descriptors: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], List[Match]]:
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return ExhaustiveMatcher().match_all(descriptors)

        all_desc = np.vstack(list(descriptors.values()))
        kmeans = KMeans(n_clusters=min(self.n_clusters, len(descriptors)))
        labels = kmeans.fit_predict(all_desc)

        results = {}
        ids = list(descriptors.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if labels[i] == labels[j]:
                    matches = self.match(descriptors[ids[i]], descriptors[ids[j]])
                    if matches:
                        results[(ids[i], ids[j])] = matches
        return results


class GeometricVerification:
    @staticmethod
    def verify_fundamental(
        kp1: List[Any], kp2: List[Any],
        matches: List[Match],
        threshold: float = 1.0
    ) -> Tuple[np.ndarray, List[Match]]:
        if len(matches) < 8:
            return np.eye(3), []

        pts1 = np.float32([kp1[m.keypoint_idx1].pt for m in matches])
        pts2 = np.float32([kp2[m.keypoint_idx2].pt for m in matches])

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, threshold, 0.99)
        inliers = [matches[i] for i in range(len(matches)) if mask[i]]
        return F, inliers

    @staticmethod
    def verify_essential(
        kp1: List[Any], kp2: List[Any],
        matches: List[Match],
        K1: np.ndarray, K2: np.ndarray,
        threshold: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Match]]:
        if len(matches) < 5:
            return np.eye(3), np.eye(3), np.zeros(3), []

        pts1 = np.float32([kp1[m.keypoint_idx1].pt for m in matches])
        pts2 = np.float32([kp2[m.keypoint_idx2].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, K1, cv2.RANSAC, 0.999, threshold)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K1, mask=mask)
        inliers = [matches[i] for i in range(len(matches)) if mask[i]]
        return E, R, t, inliers

    @staticmethod
    def verify_homography(
        kp1: List[Any], kp2: List[Any],
        matches: List[Match],
        threshold: float = 3.0
    ) -> Tuple[np.ndarray, List[Match]]:
        if len(matches) < 4:
            return np.eye(3), []

        pts1 = np.float32([kp1[m.keypoint_idx1].pt for m in matches])
        pts2 = np.float32([kp2[m.keypoint_idx2].pt for m in matches])

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)
        inliers = [matches[i] for i in range(len(matches)) if mask[i]]
        return H, inliers


class Triangulation:
    @staticmethod
    def linear_triangulation(
        P1: np.ndarray, P2: np.ndarray,
        pt1: np.ndarray, pt2: np.ndarray
    ) -> np.ndarray:
        A = np.array([
            pt1[0] * P1[2] - P1[0],
            pt1[1] * P1[2] - P1[1],
            pt2[0] * P2[2] - P2[0],
            pt2[1] * P2[2] - P2[1]
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return X[:3] / X[3] if X[3] != 0 else X[:3]

    @staticmethod
    def optimal_triangulation(
        P1: np.ndarray, P2: np.ndarray,
        pt1: np.ndarray, pt2: np.ndarray
    ) -> np.ndarray:
        return Triangulation.linear_triangulation(P1, P2, pt1, pt2)

    @staticmethod
    def triangulate_tracks(
        cameras: Dict[int, Camera],
        tracks: Dict[int, Track]
    ) -> Dict[int, np.ndarray]:
        points_3d = {}
        for track_id, track in tracks.items():
            if len(track.observations) < 2:
                continue

            obs_list = list(track.observations.items())
            P1 = cameras[obs_list[0][0]].pose @ np.hstack([cameras[obs_list[0][0]].intrinsics, np.zeros((3, 1))])
            best_point = None
            best_error = float('inf')

            for i in range(1, len(obs_list)):
                P2 = cameras[obs_list[i][0]].pose @ np.hstack([cameras[obs_list[i][0]].intrinsics, np.zeros((3, 1))])
                pt1 = np.array([obs_list[0][1][1][0], obs_list[0][1][1][1], 1])
                pt2 = np.array([obs_list[i][1][1][0], obs_list[i][1][1][1], 1])

                point = Triangulation.linear_triangulation(P1, P2, pt1, pt2)

                error = 0
                for img_id, (kp_idx, kp_pt) in track.observations.items():
                    proj = cameras[img_id].project(point)
                    error += np.linalg.norm(proj - kp_pt)

                if error < best_error:
                    best_error = error
                    best_point = point

            if best_point is not None:
                points_3d[track_id] = best_point

        return points_3d


class BundleAdjustment:
    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        loss_function: str = "huber",
        loss_scale: float = 1.0
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.loss_function = loss_function
        self.loss_scale = loss_scale

    def adjust(
        self,
        cameras: Dict[int, Camera],
        tracks: Dict[int, Track],
        points_3d: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, Camera], Dict[int, np.ndarray]]:
        try:
            import ceres
            return self._adjust_ceres(cameras, tracks, points_3d)
        except ImportError:
            return self._adjust_scipy(cameras, tracks, points_3d)

    def _adjust_ceres(
        self,
        cameras: Dict[int, Camera],
        tracks: Dict[int, Track],
        points_3d: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, Camera], Dict[int, np.ndarray]]:
        return cameras, points_3d

    def _adjust_scipy(
        self,
        cameras: Dict[int, Camera],
        tracks: Dict[int, Track],
        points_3d: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, Camera], Dict[int, np.ndarray]]:
        from scipy.optimize import least_squares

        cam_ids = list(cameras.keys())
        point_ids = list(points_3d.keys())

        cam_params = []
        for cam_id in cam_ids:
            cam = cameras[cam_id]
            rvec = cv2.Rodrigues(cam.pose[:3, :3])[0].flatten()
            tvec = cam.pose[:3, 3]
            cam_params.extend(rvec)
            cam_params.extend(tvec)

        point_params = []
        for pt_id in point_ids:
            point_params.extend(points_3d[pt_id])

        x0 = np.array(cam_params + point_params, dtype=np.float64)

        def residuals(x):
            n_cam_params = len(cam_params)
            cam_params_local = x[:n_cam_params]
            point_params_local = x[n_cam_params:]

            cams = {}
            idx = 0
            for cam_id in cam_ids:
                rvec = cam_params_local[idx:idx+3]
                tvec = cam_params_local[idx+3:idx+6]
                idx += 6
                R = cv2.Rodrigues(rvec)[0]
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = tvec
                cams[cam_id] = Camera(
                    id=cameras[cam_id].id,
                    model=cameras[cam_id].model,
                    width=cameras[cam_id].width,
                    height=cameras[cam_id].height,
                    params=cameras[cam_id].params,
                    pose=pose,
                    intrinsics=cameras[cam_id].intrinsics
                )

            pts = {}
            for i, pt_id in enumerate(point_ids):
                pts[pt_id] = point_params_local[i*3:(i+1)*3]

            res = []
            for track_id, track in tracks.items():
                if track_id not in pts:
                    continue
                point = pts[track_id]
                for img_id, (kp_idx, kp_pt) in track.observations.items():
                    if img_id not in cams:
                        continue
                    proj = cams[img_id].project(point)
                    res.extend([proj[0] - kp_pt[0], proj[1] - kp_pt[1]])

            return np.array(res)

        result = least_squares(residuals, x0, max_nfev=self.max_iterations, ftol=self.tolerance)
        x_opt = result.x

        cam_params_opt = x_opt[:len(cam_params)]
        point_params_opt = x_opt[len(cam_params):]

        new_cameras = {}
        idx = 0
        for cam_id in cam_ids:
            rvec = cam_params_opt[idx:idx+3]
            tvec = cam_params_opt[idx+3:idx+6]
            idx += 6
            R = cv2.Rodrigues(rvec)[0]
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = tvec
            new_cameras[cam_id] = Camera(
                id=cameras[cam_id].id,
                model=cameras[cam_id].model,
                width=cameras[cam_id].width,
                height=cameras[cam_id].height,
                params=cameras[cam_id].params,
                pose=pose,
                intrinsics=cameras[cam_id].intrinsics
            )

        new_points = {}
        for i, pt_id in enumerate(point_ids):
            new_points[pt_id] = point_params_opt[i*3:(i+1)*3]

        return new_cameras, new_points


class IncrementalSfM:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        feature_matcher: FeatureMatcher,
        min_triangulation_angle: float = 1.0,
        max_reprojection_error: float = 4.0,
        bundle_adjuster: Optional[BundleAdjustment] = None
    ):
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.min_triangulation_angle = min_triangulation_angle
        self.max_reprojection_error = max_reprojection_error
        self.bundle_adjuster = bundle_adjuster or BundleAdjustment()

        self.reconstruction = Reconstruction(
            cameras={}, images={}, tracks={}, points_3d={}, colors_3d={}
        )
        self.track_id_counter = 0

    def reconstruct(self, images: Dict[int, np.ndarray]) -> Reconstruction:
        self._extract_features(images)
        matches = self._match_features()
        self._initialize_reconstruction(matches)
        self._incremental_reconstruction(images, matches)
        self._bundle_adjustment()
        return self.reconstruction

    def _extract_features(self, images: Dict[int, np.ndarray]):
        for img_id, img in images.items():
            kp, desc = self.feature_extractor.extract(img)
            self.reconstruction.images[img_id] = Image(
                id=img_id, camera_id=0, path="", keypoints=kp, descriptors=desc
            )

    def _match_features(self) -> Dict[Tuple[int, int], List[Match]]:
        descriptors = {img_id: img.descriptors for img_id, img in self.reconstruction.images.items() if img.descriptors is not None}
        return self.feature_matcher.match_all(descriptors)

    def _initialize_reconstruction(self, matches: Dict[Tuple[int, int], List[Match]]):
        best_pair = max(matches.items(), key=lambda x: len(x[1]))[0] if matches else None
        if best_pair is None:
            return

        img1_id, img2_id = best_pair
        img1 = self.reconstruction.images[img1_id]
        img2 = self.reconstruction.images[img2_id]

        F, inliers = GeometricVerification.verify_fundamental(
            img1.keypoints, img2.keypoints, matches[best_pair]
        )

        if len(inliers) < 8:
            return

        K = np.eye(3)
        E, R, t, inliers = GeometricVerification.verify_essential(
            img1.keypoints, img2.keypoints, inliers, K, K
        )

        cam1 = Camera(id=0, model="pinhole", width=640, height=480, params=np.array([500, 500, 320, 240]), pose=np.eye(4), intrinsics=K)
        cam2 = Camera(id=1, model="pinhole", width=640, height=480, params=np.array([500, 500, 320, 240]), pose=np.eye(4), intrinsics=K)
        cam2.pose[:3, :3] = R
        cam2.pose[:3, 3] = t.flatten()

        self.reconstruction.cameras[0] = cam1
        self.reconstruction.cameras[1] = cam2

        for match in inliers:
            pt1 = np.array([img1.keypoints[match.keypoint_idx1].pt[0], img1.keypoints[match.keypoint_idx1].pt[1], 1])
            pt2 = np.array([img2.keypoints[match.keypoint_idx2].pt[0], img2.keypoints[match.keypoint_idx2].pt[1], 1])
            P1 = cam1.pose @ np.hstack([K, np.zeros((3, 1))])
            P2 = cam2.pose @ np.hstack([K, np.zeros((3, 1))])
            point_3d = Triangulation.linear_triangulation(P1, P2, pt1, pt2)

            track_id = self.track_id_counter
            self.track_id_counter += 1
            track = Track(
                id=track_id,
                observations={img1_id: (match.keypoint_idx1, pt1[:2]), img2_id: (match.keypoint_idx2, pt2[:2])},
                point_3d=point_3d
            )
            self.reconstruction.tracks[track_id] = track
            self.reconstruction.points_3d[track_id] = point_3d

    def _incremental_reconstruction(
        self,
        images: Dict[int, np.ndarray],
        matches: Dict[Tuple[int, int], List[Match]]
    ):
        registered = set(self.reconstruction.cameras.keys())
        unregistered = set(self.reconstruction.images.keys()) - registered

        while unregistered:
            best_img = None
            best_matches = []

            for img_id in unregistered:
                for reg_id in registered:
                    pair = (min(img_id, reg_id), max(img_id, reg_id))
                    if pair in matches:
                        if len(matches[pair]) > len(best_matches):
                            best_matches = matches[pair]
                            best_img = img_id
                            best_reg = reg_id

            if best_img is None or len(best_matches) < 10:
                break

            self._register_image(best_img, best_reg, best_matches)
            registered.add(best_img)
            unregistered.remove(best_img)

    def _register_image(
        self,
        img_id: int,
        reg_id: int,
        matches: List[Match]
    ):
        img = self.reconstruction.images[img_id]
        reg_cam = self.reconstruction.cameras[reg_id]

        obj_pts = []
        img_pts = []
        for match in matches:
            for track_id, track in self.reconstruction.tracks.items():
                if reg_id in track.observations and track.observations[reg_id][0] == match.keypoint_idx2:
                    obj_pts.append(track.point_3d)
                    img_pts.append(img.keypoints[match.keypoint_idx1].pt)
                    break

        if len(obj_pts) < 6:
            return

        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, reg_cam.intrinsics, None,
            reprojectionError=self.max_reprojection_error,
            confidence=0.99
        )

        if not success:
            return

        R = cv2.Rodrigues(rvec)[0]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.flatten()

        cam = Camera(
            id=len(self.reconstruction.cameras),
            model="pinhole",
            width=640,
            height=480,
            params=np.array([500, 500, 320, 240]),
            pose=pose,
            intrinsics=reg_cam.intrinsics
        )
        self.reconstruction.cameras[cam.id] = cam

        for i, match in enumerate(matches):
            if inliers is not None and i not in inliers:
                continue

            for track_id, track in self.reconstruction.tracks.items():
                if reg_id in track.observations and track.observations[reg_id][0] == match.keypoint_idx2:
                    track.observations[img_id] = (match.keypoint_idx1, img.keypoints[match.keypoint_idx1].pt)
                    break
            else:
                pt1 = np.array([img.keypoints[match.keypoint_idx1].pt[0], img.keypoints[match.keypoint_idx1].pt[1], 1])
                pt2 = np.array([self.reconstruction.images[reg_id].keypoints[match.keypoint_idx2].pt[0],
                               self.reconstruction.images[reg_id].keypoints[match.keypoint_idx2].pt[1], 1])
                P1 = cam.pose @ np.hstack([cam.intrinsics, np.zeros((3, 1))])
                P2 = reg_cam.pose @ np.hstack([reg_cam.intrinsics, np.zeros((3, 1))])
                point_3d = Triangulation.linear_triangulation(P1, P2, pt1, pt2)

                track_id = self.track_id_counter
                self.track_id_counter += 1
                track = Track(
                    id=track_id,
                    observations={img_id: (match.keypoint_idx1, pt1[:2]), reg_id: (match.keypoint_idx2, pt2[:2])},
                    point_3d=point_3d
                )
                self.reconstruction.tracks[track_id] = track
                self.reconstruction.points_3d[track_id] = point_3d

    def _bundle_adjustment(self):
        if self.bundle_adjuster:
            new_cameras, new_points = self.bundle_adjuster.adjust(
                self.reconstruction.cameras,
                self.reconstruction.tracks,
                self.reconstruction.points_3d
            )
            self.reconstruction.cameras = new_cameras
            self.reconstruction.points_3d = new_points


class GlobalSfM:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        feature_matcher: FeatureMatcher,
        rotation_averaging: str = "l1",
        translation_averaging: str = "l1"
    ):
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.rotation_averaging = rotation_averaging
        self.translation_averaging = translation_averaging

    def reconstruct(self, images: Dict[int, np.ndarray]) -> Reconstruction:
        pass


class SfMPipeline:
    def __init__(
        self,
        method: SfMMethod = SfMMethod.INCREMENTAL,
        feature_extractor: Optional[FeatureExtractor] = None,
        feature_matcher: Optional[FeatureMatcher] = None,
        **kwargs
    ):
        self.method = method
        self.feature_extractor = feature_extractor or SIFTExtractor()
        self.feature_matcher = feature_matcher or ExhaustiveMatcher()

        if method == SfMMethod.INCREMENTAL:
            self.pipeline = IncrementalSfM(self.feature_extractor, self.feature_matcher, **kwargs)
        elif method == SfMMethod.GLOBAL:
            self.pipeline = GlobalSfM(self.feature_extractor, self.feature_matcher, **kwargs)
        else:
            self.pipeline = IncrementalSfM(self.feature_extractor, self.feature_matcher, **kwargs)

    def reconstruct(self, images: Dict[int, np.ndarray]) -> Reconstruction:
        return self.pipeline.reconstruct(images)


class SfMVisualizer:
    @staticmethod
    def draw_reconstruction(
        reconstruction: Reconstruction,
        image_size: Tuple[int, int] = (1024, 768)
    ) -> np.ndarray:
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        for point_id, point in reconstruction.points_3d.items():
            x = int(point[0] * 100 + image_size[0] // 2)
            y = int(-point[1] * 100 + image_size[1] // 2)
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                color = reconstruction.colors_3d.get(point_id, (255, 255, 255))
                cv2.circle(vis, (x, y), 1, color.tolist() if isinstance(color, np.ndarray) else (255, 255, 255), -1)

        for cam_id, cam in reconstruction.cameras.items():
            if cam.pose is not None:
                t = cam.pose[:3, 3]
                x = int(t[0] * 100 + image_size[0] // 2)
                y = int(-t[1] * 100 + image_size[1] // 2)
                if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                    cv2.drawMarker(vis, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

        return vis

    @staticmethod
    def draw_camera_frustum(
        image: np.ndarray,
        camera: Camera,
        scale: float = 1.0,
        color: Tuple[int, int, int] = (255, 0, 0)
    ) -> np.ndarray:
        if camera.pose is None or camera.intrinsics is None:
            return image

        vis = image.copy()
        h, w = camera.height, camera.width

        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ], dtype=np.float32)

        corners_cam = (np.linalg.inv(camera.intrinsics) @ corners.T).T * scale
        corners_world = (camera.pose[:3, :3] @ corners_cam.T + camera.pose[:3, 3:4]).T

        for i in range(4):
            pt1 = tuple(corners_world[i][:2].astype(int))
            pt2 = tuple(corners_world[(i+1)%4][:2].astype(int))
            cv2.line(vis, pt1, pt2, color, 2)

        center = tuple(camera.pose[:3, 3][:2].astype(int))
        for corner in corners_world:
            cv2.line(vis, center, tuple(corner[:2].astype(int)), color, 1)

        return vis


class SfMMetrics:
    @staticmethod
    def compute_reprojection_error(
        reconstruction: Reconstruction
    ) -> Tuple[float, Dict[int, float]]:
        errors = {}
        total_error = 0.0
        total_obs = 0

        for track_id, track in reconstruction.tracks.items():
            if track_id not in reconstruction.points_3d:
                continue
            point = reconstruction.points_3d[track_id]
            track_error = 0.0
            n_obs = 0

            for img_id, (kp_idx, kp_pt) in track.observations.items():
                if img_id not in reconstruction.cameras:
                    continue
                cam = reconstruction.cameras[img_id]
                proj = cam.project(point)
                track_error += np.linalg.norm(proj - kp_pt)
                n_obs += 1

            if n_obs > 0:
                errors[track_id] = track_error / n_obs
                total_error += track_error
                total_obs += n_obs

        return total_error / total_obs if total_obs > 0 else 0.0, errors

    @staticmethod
    def compute_track_length_distribution(reconstruction: Reconstruction) -> Dict[int, int]:
        dist = defaultdict(int)
        for track in reconstruction.tracks.values():
            dist[len(track.observations)] += 1
        return dict(dist)

    @staticmethod
    def compute_camera_coverage(reconstruction: Reconstruction) -> float:
        if not reconstruction.cameras:
            return 0.0
        all_tracks = set()
        for track in reconstruction.tracks.values():
            all_tracks.update(track.observations.keys())
        return len(all_tracks) / len(reconstruction.cameras) if reconstruction.cameras else 0.0

    @staticmethod
    def compute_point_cloud_density(reconstruction: Reconstruction, voxel_size: float = 0.1) -> float:
        if not reconstruction.points_3d:
            return 0.0
        points = np.array(list(reconstruction.points_3d.values()))
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        volume = np.prod(max_coords - min_coords)
        return len(points) / volume if volume > 0 else 0.0


class SfMExporter:
    @staticmethod
    def export_ply(reconstruction: Reconstruction, path: str):
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(reconstruction.points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for pt_id, point in reconstruction.points_3d.items():
                color = reconstruction.colors_3d.get(pt_id, np.array([255, 255, 255]))
                f.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")

    @staticmethod
    def export_colmap(
        reconstruction: Reconstruction,
        output_dir: str
    ):
        import os
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "cameras.txt"), 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            for cam_id, cam in reconstruction.cameras.items():
                params_str = " ".join(map(str, cam.params))
                f.write(f"{cam_id} {cam.model.upper()} {cam.width} {cam.height} {params_str}\n")

        with open(os.path.join(output_dir, "images.txt"), 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for img_id, img in reconstruction.images.items():
                if img_id not in reconstruction.cameras:
                    continue
                cam = reconstruction.cameras[img_id]
                if cam.pose is not None:
                    R = cam.pose[:3, :3]
                    t = cam.pose[:3, 3]
                    q = cv2.Rodrigues(R)[0].flatten()
                    q = np.array([math.cos(np.linalg.norm(q)/2), *q/np.linalg.norm(q)*math.sin(np.linalg.norm(q)/2)]) if np.linalg.norm(q) > 0 else np.array([1, 0, 0, 0])
                    f.write(f"{img_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cam.id} image_{img_id}.jpg\n")
                    f.write("\n")

        with open(os.path.join(output_dir, "points3D.txt"), 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            for pt_id, point in reconstruction.points_3d.items():
                color = reconstruction.colors_3d.get(pt_id, np.array([255, 255, 255]))
                track = reconstruction.tracks.get(pt_id)
                track_str = ""
                if track:
                    for img_id, (kp_idx, _) in track.observations.items():
                        track_str += f" {img_id} {kp_idx}"
                f.write(f"{pt_id} {point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])} 0.0{track_str}\n")


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
