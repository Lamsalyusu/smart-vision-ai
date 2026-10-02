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


class CalibrationPattern(Enum):
    CHECKERBOARD = "checkerboard"
    CHARUCO = "charuco"
    APRILTAG = "apriltag"
    CIRCLE_GRID = "circle_grid"
    CUSTOM = "custom"


class CalibrationMethod(Enum):
    ZHANG = "zhang"
    TSAI = "tsai"
    BOUGUET = "bouguet"
    HEIKKILA = "heikkila"
    OPENCV = "opencv"
    KANNALA_BRANDT = "kannala_brandt"
    FISHEYE = "fisheye"
    OMNIDIRECTIONAL = "omnidirectional"


@dataclass(frozen=True, slots=True)
class CameraModel:
    model_type: str
    intrinsics: np.ndarray
    distortion: np.ndarray
    resolution: Tuple[int, int]
    serial_number: str = ""

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        points_cam = points_3d / points_cam[:, 2:3] if points_3d.shape[0] > 0 else points_3d
        u = self.intrinsics[0, 0] * points_cam[:, 0] + self.intrinsics[0, 2]
        v = self.intrinsics[1, 1] * points_cam[:, 1] + self.intrinsics[1, 2]
        return np.column_stack([u, v])

    def unproject(self, points_2d: np.ndarray, depth: np.ndarray) -> np.ndarray:
        x = (points_2d[:, 0] - self.intrinsics[0, 2]) * depth / self.intrinsics[0, 0]
        y = (points_2d[:, 1] - self.intrinsics[1, 2]) * depth / self.intrinsics[1, 1]
        return np.column_stack([x, y, depth])


@dataclass(frozen=True, slots=True)
class StereoPair:
    camera_id_1: str
    camera_id_2: str
    rotation: np.ndarray
    translation: np.ndarray
    essential_matrix: np.ndarray
    fundamental_matrix: np.ndarray
    rectification: Optional[Dict] = None


@dataclass(frozen=True, slots=True)
class MultiCameraCalibration:
    cameras: Dict[str, CameraModel]
    stereo_pairs: Dict[Tuple[str, str], StereoPair]
    global_optimization: bool = False
    total_reprojection_error: float = 0.0


class IntrinsicCalibrator:
    def __init__(
        self,
        pattern: CalibrationPattern = CalibrationPattern.CHECKERBOARD,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 25.0,
        method: CalibrationMethod = CalibrationMethod.OPENCV
    ):
        self.pattern = pattern
        self.board_size = board_size
        self.square_size = square_size
        self.method = method

        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    def calibrate(
        self,
        images: List[np.ndarray],
        camera_id: str = "camera_0"
    ) -> CameraModel:
        objpoints = []
        imgpoints = []
        image_size = None

        for img in images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            if image_size is None:
                image_size = gray.shape[::-1]

            found, corners = self._find_corners(gray)

            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                objpoints.append(self.objp)
                imgpoints.append(corners)

        if len(objpoints) < 10:
            raise ValueError("Need at least 10 valid images for calibration")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None,
            flags=self._get_calibration_flags()
        )

        return CameraModel(
            model_type="pinhole" if self.method != CalibrationMethod.FISHEYE else "fisheye",
            intrinsics=mtx,
            distortion=dist.flatten(),
            resolution=image_size
        )

    def _find_corners(self, gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        if self.pattern == CalibrationPattern.CHECKERBOARD:
            return cv2.findChessboardCorners(gray, self.board_size, None)
        elif self.pattern == CalibrationPattern.CHARUCO:
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            board = cv2.aruco.CharucoBoard_create(self.board_size[0], self.board_size[1],
                                                 self.square_size, self.square_size * 0.7, dictionary)
            charuco_corners, charuco_ids, _, _ = cv2.aruco.detectCharucoDiamond(gray, board, dictionary)
            if charuco_corners is not None and len(charuco_corners) > 0:
                return True, charuco_corners
        elif self.pattern == CalibrationPattern.CIRCLE_GRID:
            return cv2.findCirclesGrid(gray, self.board_size, None, cv2.CALIB_CB_SYMMETRIC_GRID)

        return False, None

    def _get_calibration_flags(self) -> int:
        if self.method == CalibrationMethod.FISHEYE:
            return cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
        return cv2.CALIB_RATIONAL_MODEL


class ExtrinsicCalibrator:
    def __init__(
        self,
        pattern: CalibrationPattern = CalibrationPattern.CHECKERBOARD,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 25.0
    ):
        self.pattern = pattern
        self.board_size = board_size
        self.square_size = square_size

        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    def calibrate_stereo(
        self,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
        left_intrinsics: CameraModel,
        right_intrinsics: CameraModel
    ) -> StereoPair:
        if len(left_images) != len(right_images):
            raise ValueError("Left and right image lists must have same length")

        left_objpoints = []
        right_objpoints = []
        left_imgpoints = []
        right_imgpoints = []
        image_size = None

        for left_img, right_img in zip(left_images, right_images):
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY) if len(left_img.shape) == 3 else left_img
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY) if len(right_img.shape) == 3 else right_img

            if image_size is None:
                image_size = left_gray.shape[::-1]

            left_found, left_corners = self._find_corners(left_gray)
            right_found, right_corners = self._find_corners(right_gray)

            if left_found and right_found:
                left_corners = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1),
                                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                right_corners = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1),
                                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

                left_objpoints.append(self.objp)
                right_objpoints.append(self.objp)
                left_imgpoints.append(left_corners)
                right_imgpoints.append(right_corners)

        if len(left_objpoints) < 10:
            raise ValueError("Need at least 10 valid stereo pairs")

        ret, left_mtx, left_dist, right_mtx, right_dist, R, T, E, F = cv2.stereoCalibrate(
            left_objpoints, left_imgpoints, right_imgpoints,
            left_intrinsics.intrinsics, left_intrinsics.distortion,
            right_intrinsics.intrinsics, right_intrinsics.distortion,
            image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL
        )

        return StereoPair(
            camera_id_1="left",
            camera_id_2="right",
            rotation=R,
            translation=T.flatten(),
            essential_matrix=E,
            fundamental_matrix=F
        )

    def _find_corners(self, gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        return cv2.findChessboardCorners(gray, self.board_size, None)


class MultiCameraCalibrator:
    def __init__(
        self,
        pattern: CalibrationPattern = CalibrationPattern.CHECKERBOARD,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 25.0
    ):
        self.pattern = pattern
        self.board_size = board_size
        self.square_size = square_size

        self.intrinsic_calibrator = IntrinsicCalibrator(pattern, board_size, square_size)
        self.extrinsic_calibrator = ExtrinsicCalibrator(pattern, board_size, square_size)

        self.cameras: Dict[str, CameraModel] = {}
        self.stereo_pairs: Dict[Tuple[str, str], StereoPair] = {}

    def add_camera(self, camera_id: str, images: List[np.ndarray]):
        camera_model = self.intrinsic_calibrator.calibrate(images, camera_id)
        self.cameras[camera_id] = camera_model

    def calibrate_stereo_pair(self, camera_id_1: str, camera_id_2: str, images_1: List[np.ndarray], images_2: List[np.ndarray]):
        if camera_id_1 not in self.cameras or camera_id_2 not in self.cameras:
            raise ValueError("Both cameras must be added first")

        stereo_pair = self.extrinsic_calibrator.calibrate_stereo(
            images_1, images_2,
            self.cameras[camera_id_1], self.cameras[camera_id_2]
        )

        self.stereo_pairs[(camera_id_1, camera_id_2)] = stereo_pair

    def global_optimization(self, all_images: Dict[str, List[np.ndarray]]) -> MultiCameraCalibration:
        pass


class HandEyeCalibrator:
    def __init__(self, method: str = "tsai"):
        self.method = method
        self.robot_poses: List[np.ndarray] = []
        self.camera_poses: List[np.ndarray] = []

    def add_correspondence(self, robot_pose: np.ndarray, camera_pose: np.ndarray):
        self.robot_poses.append(robot_pose)
        self.camera_poses.append(camera_pose)

    def calibrate(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.robot_poses) < 4:
            raise ValueError("Need at least 4 robot-camera pose pairs")

        R_gripper2base = [pose[:3, :3] for pose in self.robot_poses]
        t_gripper2base = [pose[:3, 3] for pose in self.robot_poses]
        R_target2cam = [pose[:3, :3] for pose in self.camera_poses]
        t_target2cam = [pose[:3, 3] for pose in self.camera_poses]

        if self.method == "tsai":
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                cv2.CALIB_HAND_EYE_TSAI
            )
        elif self.method == "park":
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                cv2.CALIB_HAND_EYE_PARK
            )
        elif self.method == "horaud":
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                cv2.CALIB_HAND_EYE_HORAUD
            )
        elif self.method == "andreff":
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                cv2.CALIB_HAND_EYE_ANDREFF
            )
        elif self.method == "daniilidis":
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                cv2.CALIB_HAND_EYE_DANIILIDIS
            )
        else:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                cv2.CALIB_HAND_EYE_TSAI
            )

        return R_cam2gripper, t_cam2gripper.flatten()


class LidarCameraCalibrator:
    def __init__(self):
        pass

    def calibrate(
        self,
        lidar_points: List[np.ndarray],
        camera_images: List[np.ndarray],
        camera_model: CameraModel,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 25.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        return np.eye(3), np.zeros(3)


class RadarCameraCalibrator:
    def __init__(self):
        pass

    def calibrate(
        self,
        radar_data: List[np.ndarray],
        camera_images: List[np.ndarray],
        camera_model: CameraModel
    ) -> Tuple[np.ndarray, np.ndarray]:
        return np.eye(3), np.zeros(3)


class IMUCalibrator:
    def __init__(self):
        self.accel_samples: List[np.ndarray] = []
        self.gyro_samples: List[np.ndarray] = []
        self.gravity = 9.81

    def add_imu_sample(self, accel: np.ndarray, gyro: np.ndarray):
        self.accel_samples.append(accel)
        self.gyro_samples.append(gyro)

    def calibrate_accel(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.accel_samples) < 100:
            raise ValueError("Need at least 100 samples")

        accel_data = np.array(self.accel_samples)

        scale = np.zeros(3)
        bias = np.zeros(3)

        for i in range(3):
            max_val = np.max(accel_data[:, i])
            min_val = np.min(accel_data[:, i])
            scale[i] = self.gravity * 2 / (max_val - min_val)
            bias[i] = -(max_val + min_val) / 2 * scale[i]

        return scale, bias

    def calibrate_gyro(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.gyro_samples) < 100:
            raise ValueError("Need at least 100 samples")

        gyro_data = np.array(self.gyro_samples)

        bias = np.mean(gyro_data, axis=0)
        noise = np.std(gyro_data, axis=0)

        return bias, noise


class CalibrationQualityChecker:
    @staticmethod
    def check_reprojection_error(
        camera_model: CameraModel,
        objpoints: List[np.ndarray],
        imgpoints: List[np.ndarray],
        rvecs: List[np.ndarray],
        tvecs: List[np.ndarray]
    ) -> Tuple[float, List[float]]:
        total_error = 0.0
        per_view_errors = []

        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                             camera_model.intrinsics, camera_model.distortion)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            per_view_errors.append(float(error))
            total_error += error

        return total_error / len(objpoints), per_view_errors

    @staticmethod
    def check_stereo_rectification(
        left_mtx: np.ndarray, right_mtx: np.ndarray,
        R: np.ndarray, T: np.ndarray,
        image_size: Tuple[int, int]
    ) -> float:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            left_mtx, np.zeros(5), right_mtx, np.zeros(5),
            image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY
        )

        return float(np.abs(np.sum(P1[:, 2] - P2[:, 2])))

    @staticmethod
    def check_handeye_accuracy(
        R_cam2gripper: np.ndarray, t_cam2gripper: np.ndarray,
        robot_poses: List[np.ndarray], camera_poses: List[np.ndarray]
    ) -> float:
        errors = []
        for i in range(len(robot_poses)):
            R_est = robot_poses[i][:3, :3] @ R_cam2gripper
            t_est = robot_poses[i][:3, :3] @ t_cam2gripper + robot_poses[i][:3, 3]

            R_gt = camera_poses[i][:3, :3]
            t_gt = camera_poses[i][:3, 3]

            rot_err = cv2.Rodrigues(R_est @ R_gt.T)[0]
            trans_err = np.linalg.norm(t_est - t_gt)

            errors.append(np.linalg.norm(rot_err) + trans_err)

        return np.mean(errors)


class MultiCameraVisualizer:
    @staticmethod
    def draw_camera_frustums(
        calibration: MultiCameraCalibration,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        positions = []
        for cam_id, cam in calibration.cameras.items():
            R = np.eye(3)
            t = np.zeros(3)
            pos = -R.T @ t
            positions.append(pos[:2])

        if not positions:
            return vis

        pos_array = np.array(positions)
        min_coords = np.min(pos_array, axis=0)
        max_coords = np.max(pos_array, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        for i, (cam_id, cam) in enumerate(calibration.cameras.items()):
            pos = positions[i]
            x = int((pos[0] - min_coords[0]) * scale + image_size[0] * 0.1)
            y = int(image_size[1] - (pos[1] - min_coords[1]) * scale - image_size[1] * 0.1)

            cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(vis, cam_id, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for (cam1, cam2), stereo in calibration.stereo_pairs.items():
            if cam1 in calibration.cameras and cam2 in calibration.cameras:
                idx1 = list(calibration.cameras.keys()).index(cam1)
                idx2 = list(calibration.cameras.keys()).index(cam2)
                if idx1 < len(positions) and idx2 < len(positions):
                    p1 = positions[idx1]
                    p2 = positions[idx2]
                    x1 = int((p1[0] - min_coords[0]) * scale + image_size[0] * 0.1)
                    y1 = int(image_size[1] - (p1[1] - min_coords[1]) * scale - image_size[1] * 0.1)
                    x2 = int((p2[0] - min_coords[0]) * scale + image_size[0] * 0.1)
                    y2 = int(image_size[1] - (p2[1] - min_coords[1]) * scale - image_size[1] * 0.1)
                    cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return vis

    @staticmethod
    def draw_reprojection_errors(
        errors: List[float],
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not errors:
            return vis

        max_error = max(errors)
        min_error = min(errors)

        for i, error in enumerate(errors):
            x = int(50 + i * (image_size[0] - 100) / len(errors))
            height = int((error - min_error) / (max_error - min_error + 1e-6) * (image_size[1] - 100))
            y = image_size[1] - 50 - height

            color = (0, 255, 0) if error < np.mean(errors) else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + 20, image_size[1] - 50), color, -1)

        return vis


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet