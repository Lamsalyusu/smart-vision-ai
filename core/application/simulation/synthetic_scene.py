import random
from dataclasses import dataclass

from core.domain.spatial.frustum import CameraIntrinsics
from core.domain.spatial.multicam import CameraExtrinsics
from core.domain.spatial.transforms import Vector3


@dataclass(frozen=True, slots=True)
class GroundTruthTarget3D:
    track_id: int
    class_name: str
    start_pos: Vector3
    velocity: Vector3
    width_m: float = 0.6
    height_m: float = 1.8
    depth_m: float = 0.4

    def position_at(self, t_seconds: float) -> Vector3:
        return Vector3(
            self.start_pos.x + self.velocity.x * t_seconds,
            self.start_pos.y + self.velocity.y * t_seconds,
            self.start_pos.z + self.velocity.z * t_seconds,
        )


@dataclass(frozen=True, slots=True)
class SimulatedDetection:
    camera_id: str
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    true_track_id: int
    noisy_depth_m: float
    confidence: float


@dataclass(frozen=True, slots=True)
class SyntheticSceneFrame:
    frame_sequence: int
    timestamp_seconds: float
    ground_truth_targets: list[tuple[int, Vector3]]
    camera_detections: dict[str, list[SimulatedDetection]]


class Synthetic3DSceneSimulator:
    """Multi-camera synthetic 3D perception scenario generator with physics-based kinematics,

    photometric noise, occlusions, and clutter.
    """

    def __init__(
        self,
        cameras: dict[str, tuple[CameraIntrinsics, CameraExtrinsics]],
        seed: int = 42,
    ) -> None:
        self.cameras = cameras
        self._rng = random.Random(seed)
        self._targets: list[GroundTruthTarget3D] = []

    def add_target(self, target: GroundTruthTarget3D) -> None:
        self._targets.append(target)

    def generate_scenario(
        self,
        duration_seconds: float,
        fps: float = 30.0,
        miss_probability: float = 0.05,
        false_alarm_rate: float = 0.02,
        pos_noise_std_px: float = 2.0,
        depth_noise_std_m: float = 0.08,
    ) -> list[SyntheticSceneFrame]:
        dt = 1.0 / fps
        total_frames = int(duration_seconds * fps)
        frames: list[SyntheticSceneFrame] = []

        for frame_idx in range(total_frames):
            t = frame_idx * dt
            gt_positions: list[tuple[int, Vector3]] = []
            cam_detections: dict[str, list[SimulatedDetection]] = {
                cam_id: [] for cam_id in self.cameras
            }

            # 1. Update ground truth target positions
            for target in self._targets:
                pos = target.position_at(t)
                gt_positions.append((target.track_id, pos))

                # Project into each camera view
                for cam_id, (intrinsics, extrinsics) in self.cameras.items():
                    # Transform point from world to camera frame: p_cam = R * p_world + t
                    r_mat = extrinsics.rotation
                    t_vec = extrinsics.translation
                    cx = (
                        r_mat[0][0] * pos.x
                        + r_mat[0][1] * pos.y
                        + r_mat[0][2] * pos.z
                        + t_vec[0]
                    )
                    cy = (
                        r_mat[1][0] * pos.x
                        + r_mat[1][1] * pos.y
                        + r_mat[1][2] * pos.z
                        + t_vec[1]
                    )
                    cz = (
                        r_mat[2][0] * pos.x
                        + r_mat[2][1] * pos.y
                        + r_mat[2][2] * pos.z
                        + t_vec[2]
                    )

                    # Behind camera or beyond max range
                    if cz <= 0.2 or cz > 50.0:
                        continue

                    # Projected center pixel
                    u_center, v_center = intrinsics.project(Vector3(cx, cy, cz))

                    # 2D Bounding box size approximation
                    box_w = (target.width_m * intrinsics.fx) / cz
                    box_h = (target.height_m * intrinsics.fy) / cz

                    x1 = u_center - box_w / 2.0
                    y1 = v_center - box_h / 2.0
                    x2 = u_center + box_w / 2.0
                    y2 = v_center + box_h / 2.0

                    # Check image bounds (1920x1080 default or within projection)
                    if (
                        x2 < 0
                        or y2 < 0
                        or x1 > intrinsics.cx * 2
                        or y1 > intrinsics.cy * 2
                    ):
                        continue

                    # Simulated missed detection
                    if self._rng.random() < miss_probability:
                        continue

                    # Add pixel position noise
                    noise_x = self._rng.gauss(0.0, pos_noise_std_px)
                    noise_y = self._rng.gauss(0.0, pos_noise_std_px)
                    noise_z = self._rng.gauss(0.0, depth_noise_std_m)

                    noisy_x1 = max(0.0, x1 + noise_x)
                    noisy_y1 = max(0.0, y1 + noise_y)
                    noisy_x2 = max(noisy_x1 + 5.0, x2 + noise_x)
                    noisy_y2 = max(noisy_y1 + 5.0, y2 + noise_y)

                    noisy_depth = max(0.2, cz + noise_z)
                    conf = max(
                        0.5,
                        min(
                            0.99,
                            1.0 - (cz / 50.0) * 0.3 + self._rng.gauss(0.0, 0.05),
                        ),
                    )

                    cam_detections[cam_id].append(
                        SimulatedDetection(
                            camera_id=cam_id,
                            class_name=target.class_name,
                            bbox_xyxy=(noisy_x1, noisy_y1, noisy_x2, noisy_y2),
                            true_track_id=target.track_id,
                            noisy_depth_m=noisy_depth,
                            confidence=conf,
                        )
                    )

            # 2. Inject false alarm clutter
            for cam_id, (intrinsics, _) in self.cameras.items():
                if self._rng.random() < false_alarm_rate:
                    rx = self._rng.uniform(50.0, intrinsics.cx * 1.8)
                    ry = self._rng.uniform(50.0, intrinsics.cy * 1.8)
                    rw = self._rng.uniform(30.0, 100.0)
                    rh = self._rng.uniform(30.0, 120.0)
                    cam_detections[cam_id].append(
                        SimulatedDetection(
                            camera_id=cam_id,
                            class_name="clutter",
                            bbox_xyxy=(rx, ry, rx + rw, ry + rh),
                            true_track_id=-1,
                            noisy_depth_m=self._rng.uniform(2.0, 15.0),
                            confidence=self._rng.uniform(0.4, 0.7),
                        )
                    )

            frames.append(
                SyntheticSceneFrame(
                    frame_sequence=frame_idx,
                    timestamp_seconds=t,
                    ground_truth_targets=gt_positions,
                    camera_detections=cam_detections,
                )
            )

        return frames
