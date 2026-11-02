from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict
import cv2
import yaml


class CalibrationTarget(Enum):
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
    FISHEYE = "fisheye"
    OMNIDIRECTIONAL = "omnidirectional"
    STEREO = "stereo"
    HAND_EYE = "hand_eye"
    LIDAR_CAMERA = "lidar_camera"
    RADAR_CAMERA = "radar_camera"
    IMU_CAMERA = "imu_camera"


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    width: int = 640
    height: int = 480

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0, 0, 1]], dtype=np.float32)

    @property
    def D(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)

    @classmethod
    def from_K_D(cls, K: np.ndarray, D: np.ndarray, width: int, height: int) -> 'CameraIntrinsics':
        return cls(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            k1=D[0] if len(D) > 0 else 0,
            k2=D[1] if len(D) > 1 else 0,
            p1=D[2] if len(D) > 2 else 0,
            p2=D[3] if len(D) > 3 else 0,
            k3=D[4] if len(D) > 4 else 0,
            width=width, height=height
        )


@dataclass(frozen=True, slots=True)
class Extrinsics:
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def transform(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    @classmethod
    def from_transform(cls, T: np.ndarray) -> 'Extrinsics':
        return cls(T[:3, :3], T[:3, 3])

    def inverse(self) -> 'Extrinsics':
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation
        return Extrinsics(R_inv, t_inv)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    reprojection_error: float
    per_view_errors: List[float]
    success: bool
    method: CalibrationMethod
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class StereoCalibrationResult:
    left_intrinsics: CameraIntrinsics
    right_intrinsics: CameraIntrinsics
    rotation: np.ndarray
    translation: np.ndarray
    essential_matrix: np.ndarray
    fundamental_matrix: np.ndarray
    reprojection_error: float
    success: bool


@dataclass(frozen=True, slots=True)
class HandEyeCalibrationResult:
    robot_to_camera: Extrinsics
    camera_to_robot: Extrinsics
    error: float
    success: bool


class BaseCalibrator(ABC):
    def __init__(self, method: CalibrationMethod):
        self.method = method

    @abstractmethod
    def calibrate(self, images: List[np.ndarray], **kwargs) -> CalibrationResult:
        pass


class CameraCalibrator(BaseCalibrator):
    def __init__(
        self,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 25.0,
        target_type: CalibrationTarget = CalibrationTarget.CHECKERBOARD,
        flags: int = cv2.CALIB_RATIONAL_MODEL
    ):
        super().__init__(CalibrationMethod.ZHANG)
        self.board_size = board_size
        self.square_size = square_size
        self.target_type = target_type
        self.flags = flags

        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

        self.objpoints: List[np.ndarray] = []
        self.imgpoints: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None

    def add_image(self, image: np.ndarray) -> bool:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        if self.image_size is None:
            self.image_size = gray.shape[::-1]

        found = False
        corners = None

        if self.target_type == CalibrationTarget.CHECKERBOARD:
            found, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        elif self.target_type == CalibrationTarget.CHARUCO:
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            board = cv2.aruco.CharucoBoard_create(self.board_size[0], self.board_size[1], self.square_size, self.square_size * 0.7, dictionary)
            charuco_corners, charuco_ids, _, _ = cv2.aruco.detectCharucoDiamond(gray, board, dictionary)
            if charuco_corners is not None and len(charuco_corners) > 0:
                found = True
                corners = charuco_corners
        elif self.target_type == CalibrationTarget.CIRCLE_GRID:
            found, corners = cv2.findCirclesGrid(gray, self.board_size, None, cv2.CALIB_CB_SYMMETRIC_GRID)

        if found:
            if self.target_type != CalibrationTarget.CHARUCO:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)
            return True

        return False

    def calibrate(self, images: List[np.ndarray] = None, **kwargs) -> CalibrationResult:
        if images:
            for img in images:
                self.add_image(img)

        if len(self.objpoints) < 10:
            return CalibrationResult(
                np.eye(3), np.zeros(5), [], [], float('inf'), [], False, self.method
            )

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None,
            flags=self.flags
        )

        per_view_errors = []
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            per_view_errors.append(float(error))

        return CalibrationResult(
            camera_matrix=mtx,
            dist_coeffs=dist,
            rvecs=rvecs,
            tvecs=tvecs,
            reprojection_error=ret,
            per_view_errors=per_view_errors,
            success=True,
            method=self.method
        )

    def undistort(self, image: np.ndarray, result: CalibrationResult) -> np.ndarray:
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(result.camera_matrix, result.dist_coeffs, (w, h), 1, (w, h))
        dst = cv2.undistort(image, result.camera_matrix, result.dist_coeffs, None, newcameramtx)
        x, y, w, h = roi
        return dst[y:y+h, x:x+w]


class FisheyeCalibrator(BaseCalibrator):
    def __init__(
        self,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 25.0,
        flags: int = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
    ):
        super().__init__(CalibrationMethod.FISHEYE)
        self.board_size = board_size
        self.square_size = square_size
        self.flags = flags

        self.objp = np.zeros((1, board_size[0] * board_size[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

        self.objpoints: List[np.ndarray] = []
        self.imgpoints: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None

    def add_image(self, image: np.ndarray) -> bool:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        if self.image_size is None:
            self.image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, self.board_size, None)

        if found:
            corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)
            return True

        return False

    def calibrate(self, images: List[np.ndarray] = None, **kwargs) -> CalibrationResult:
        if images:
            for img in images:
                self.add_image(img)

        if len(self.objpoints) < 10:
            return CalibrationResult(
                np.eye(3), np.zeros(4), [], [], float('inf'), [], False, self.method
            )

        N_OK = len(self.objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

        ret, _, _, _, _ = cv2.fisheye.calibrate(
            self.objpoints, self.imgpoints, self.image_size,
            K, D, rvecs, tvecs,
            self.flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

        per_view_errors = []
        for i in range(N_OK):
            imgpoints2, _ = cv2.fisheye.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], K, D)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            per_view_errors.append(float(error))

        return CalibrationResult(
            camera_matrix=K,
            dist_coeffs=D.flatten(),
            rvecs=[r.flatten() for r in rvecs],
            tvecs=[t.flatten() for t in tvecs],
            reprojection_error=ret,
            per_view_errors=per_view_errors,
            success=True,
            method=self.method
        )


class StereoCalibrator(BaseCalibrator):
    def __init__(
        self,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 25.0,
        flags: int = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL
    ):
        super().__init__(CalibrationMethod.STEREO)
        self.board_size = board_size
        self.square_size = square_size
        self.flags = flags

        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

        self.left_objpoints: List[np.ndarray] = []
        self.left_imgpoints: List[np.ndarray] = []
        self.right_objpoints: List[np.ndarray] = []
        self.right_imgpoints: List[np.ndarray] = []

        self.left_image_size: Optional[Tuple[int, int]] = None
        self.right_image_size: Optional[Tuple[int, int]] = None

        self.left_camera_matrix = np.eye(3)
        self.left_dist_coeffs = np.zeros(5)
        self.right_camera_matrix = np.eye(3)
        self.right_dist_coeffs = np.zeros(5)

    def add_stereo_pair(self, left_image: np.ndarray, right_image: np.ndarray) -> bool:
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY) if len(left_image.shape) == 3 else left_image
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY) if len(right_image.shape) == 3 else right_image

        if self.left_image_size is None:
            self.left_image_size = left_gray.shape[::-1]
        if self.right_image_size is None:
            self.right_image_size = right_gray.shape[::-1]

        left_found, left_corners = cv2.findChessboardCorners(left_gray, self.board_size, None)
        right_found, right_corners = cv2.findChessboardCorners(right_gray, self.board_size, None)

        if left_found and right_found:
            left_corners = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            right_corners = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            self.left_objpoints.append(self.objp)
            self.left_imgpoints.append(left_corners)
            self.right_objpoints.append(self.objp)
            self.right_imgpoints.append(right_corners)
            return True

        return False

    def calibrate(self, images: List[np.ndarray] = None, **kwargs) -> StereoCalibrationResult:
        if len(self.left_objpoints) < 10:
            return StereoCalibrationResult(
                CameraIntrinsics(500, 500, 320, 240),
                CameraIntrinsics(500, 500, 320, 240),
                np.eye(3), np.zeros(3),
                np.eye(3), np.eye(3),
                float('inf'), False
            )

        ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, R, T, E, F = cv2.stereoCalibrate(
            self.left_objpoints, self.left_imgpoints, self.right_imgpoints,
            self.left_camera_matrix, self.left_dist_coeffs,
            self.right_camera_matrix, self.right_dist_coeffs,
            self.left_image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=self.flags
        )

        left_intrinsics = CameraIntrinsics.from_K_D(left_camera_matrix, left_dist_coeffs, *self.left_image_size)
        right_intrinsics = CameraIntrinsics.from_K_D(right_camera_matrix, right_dist_coeffs, *self.right_image_size)

        return StereoCalibrationResult(
            left_intrinsics=left_intrinsics,
            right_intrinsics=right_intrinsics,
            rotation=R,
            translation=T.flatten(),
            essential_matrix=E,
            fundamental_matrix=F,
            reprojection_error=ret,
            success=True
        )


class HandEyeCalibrator(BaseCalibrator):
    def __init__(self, method: str = "tsai"):
        super().__init__(CalibrationMethod.HAND_EYE)
        self.method = method

        self.robot_poses: List[np.ndarray] = []
        self.camera_poses: List[np.ndarray] = []

    def add_correspondence(self, robot_pose: np.ndarray, camera_pose: np.ndarray):
        self.robot_poses.append(robot_pose)
        self.camera_poses.append(camera_pose)

    def calibrate(self, images: List[np.ndarray] = None, **kwargs) -> HandEyeCalibrationResult:
        if len(self.robot_poses) < 4:
            return HandEyeCalibrationResult(
                Extrinsics(np.eye(3), np.zeros(3)),
                Extrinsics(np.eye(3), np.zeros(3)),
                float('inf'), False
            )

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

        robot_to_camera = Extrinsics(R_cam2gripper, t_cam2gripper.flatten())
        camera_to_robot = robot_to_camera.inverse()

        errors = []
        for i in range(len(self.robot_poses)):
            R_est = self.robot_poses[i][:3, :3] @ R_cam2gripper
            t_est = self.robot_poses[i][:3, :3] @ t_cam2gripper + self.robot_poses[i][:3, 3]
            R_gt = self.camera_poses[i][:3, :3]
            t_gt = self.camera_poses[i][:3, 3]

            rot_err = cv2.Rodrigues(R_est @ R_gt.T)[0]
            trans_err = np.linalg.norm(t_est - t_gt)
            errors.append(np.linalg.norm(rot_err) + trans_err)

        return HandEyeCalibrationResult(
            robot_to_camera=robot_to_camera,
            camera_to_robot=camera_to_robot,
            error=np.mean(errors),
            success=True
        )


class LiDARCameraCalibrator(BaseCalibrator):
    def __init__(
        self,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 25.0
    ):
        super().__init__(CalibrationMethod.LIDAR_CAMERA)
        self.board_size = board_size
        self.square_size = square_size

    def calibrate(
        self,
        lidar_points: List[np.ndarray],
        camera_images: List[np.ndarray],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> Extrinsics:
        return Extrinsics(np.eye(3), np.zeros(3))


class RadarCameraCalibrator(BaseCalibrator):
    def __init__(self):
        super().__init__(CalibrationMethod.RADAR_CAMERA)

    def calibrate(
        self,
        radar_data: List[np.ndarray],
        camera_images: List[np.ndarray],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> Extrinsics:
        return Extrinsics(np.eye(3), np.zeros(3))


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
        camera_model: CameraIntrinsics,
        objpoints: List[np.ndarray],
        imgpoints: List[np.ndarray],
        rvecs: List[np.ndarray],
        tvecs: List[np.ndarray]
    ) -> Tuple[float, List[float]]:
        total_error = 0.0
        per_view_errors = []

        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                             camera_model.K, camera_model.D)
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


class CalibrationVisualizer:
    @staticmethod
    def draw_reprojection_errors(result: CalibrationResult, image_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not result.per_view_errors:
            return vis

        max_error = max(result.per_view_errors)
        min_error = min(result.per_view_errors)

        for i, error in enumerate(result.per_view_errors):
            x = int(50 + i * (image_size[0] - 100) / len(result.per_view_errors))
            height = int((error - min_error) / (max_error - min_error + 1e-6) * (image_size[1] - 100))
            y = image_size[1] - 50 - height

            color = (0, 255, 0) if error < result.reprojection_error else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + 20, image_size[1] - 50), color, -1)

        return vis

    @staticmethod
    def draw_undistortion(
        original: np.ndarray,
        undistorted: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> np.ndarray:
        return np.hstack([original, undistorted])

    @staticmethod
    def draw_stereo_rectification(
        left: np.ndarray,
        right: np.ndarray,
        left_rect: np.ndarray,
        right_rect: np.ndarray
    ) -> np.ndarray:
        top = np.hstack([left, right])
        bottom = np.hstack([left_rect, right_rect])
        return np.vstack([top, bottom])


class CalibrationExporter:
    @staticmethod
    def export_yaml(result: CalibrationResult, path: str):
        import yaml
        data = {
            'camera_matrix': result.camera_matrix.tolist(),
            'dist_coeffs': result.dist_coeffs.tolist(),
            'reprojection_error': float(result.reprojection_error),
            'per_view_errors': [float(e) for e in result.per_view_errors],
            'image_width': 640,
            'image_height': 480
        }
        with open(path, 'w') as f:
            yaml.dump(data, f)

    @staticmethod
    def export_opencv(result: CalibrationResult, path: str):
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("camera_matrix", result.camera_matrix)
        cv_file.write("dist_coeffs", result.dist_coeffs)
        cv_file.write("reprojection_error", result.reprojection_error)
        cv_file.release()

    @staticmethod
    def export_stereo_yaml(result: StereoCalibrationResult, path: str):
        import yaml
        data = {
            'left': {
                'camera_matrix': result.left_intrinsics.K.tolist(),
                'dist_coeffs': result.left_intrinsics.D.tolist()
            },
            'right': {
                'camera_matrix': result.right_intrinsics.K.tolist(),
                'dist_coeffs': result.right_intrinsics.D.tolist()
            },
            'rotation': result.rotation.tolist(),
            'translation': result.translation.tolist(),
            'essential_matrix': result.essential_matrix.tolist(),
            'fundamental_matrix': result.fundamental_matrix.tolist(),
            'reprojection_error': float(result.reprojection_error)
        }
        with open(path, 'w') as f:
            yaml.dump(data, f)

    @staticmethod
    def export_handeye(result: HandEyeCalibrationResult, path: str):
        import yaml
        data = {
            'robot_to_camera': result.robot_to_camera.transform.tolist(),
            'camera_to_robot': result.camera_to_robot.transform.tolist(),
            'error': float(result.error)
        }
        with open(path, 'w') as f:
            yaml.dump(data, f)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet