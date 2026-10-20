from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import random
from collections import defaultdict


class SimulatorType(Enum):
    GAZEBO = "gazebo"
    PYBULLET = "pybullet"
    ISAAC = "isaac"
    MURPHY = "murphy"
    RAI_SIM = "rai_sim"
    MUJOCO = "mujoco"
    DRAKE = "drake"
    CUSTOM = "custom"


class RobotType(Enum):
    MANIPULATOR = "manipulator"
    MOBILE = "mobile"
    HUMANOID = "humanoid"
    QUADROTOR = "quadrotor"
    FIXED_WING = "fixed_wing"
    UNDERWATER = "underwater"
    SPACE = "space"


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    simulator_type: SimulatorType = SimulatorType.PYBULLET
    timestep: float = 1.0 / 240.0
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    real_time: bool = False
    gui: bool = True
    num_threads: int = 4
    physics_engine_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RobotState:
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    base_position: np.ndarray
    base_orientation: np.ndarray
    base_linear_velocity: np.ndarray
    base_angular_velocity: np.ndarray
    contact_forces: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SensorData:
    camera: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    lidar: Optional[np.ndarray] = None
    imu: Optional[Dict[str, np.ndarray]] = None
    joint_states: Optional[Dict[str, np.ndarray]] = None
    force_torque: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class ContactPoint:
    body_a: int
    body_b: int
    link_a: int
    link_b: int
    position: np.ndarray
    normal: np.ndarray
    force: float
    distance: float


class PhysicsEngine:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.bodies: Dict[int, Dict] = {}
        self.joints: Dict[int, Dict] = {}
        self.body_id_counter = 0
        self.joint_id_counter = 0

    def add_rigid_body(
        self,
        mass: float,
        position: np.ndarray,
        orientation: np.ndarray,
        collision_shape: Dict,
        visual_shape: Optional[Dict] = None,
        is_static: bool = False
    ) -> int:
        body_id = self.body_id_counter
        self.body_id_counter += 1

        self.bodies[body_id] = {
            'mass': mass,
            'position': position,
            'orientation': orientation,
            'collision_shape': collision_shape,
            'visual_shape': visual_shape,
            'is_static': is_static,
            'linear_velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'force': np.zeros(3),
            'torque': np.zeros(3)
        }

        return body_id

    def add_joint(
        self,
        body_a: int,
        body_b: int,
        joint_type: str,
        parent_frame: np.ndarray,
        child_frame: np.ndarray,
        axis: np.ndarray,
        limits: Optional[Tuple[float, float]] = None,
        max_force: float = 1000.0
    ) -> int:
        joint_id = self.joint_id_counter
        self.joint_id_counter += 1

        self.joints[joint_id] = {
            'body_a': body_a,
            'body_b': body_b,
            'type': joint_type,
            'parent_frame': parent_frame,
            'child_frame': child_frame,
            'axis': axis,
            'limits': limits,
            'max_force': max_force,
            'position': 0.0,
            'velocity': 0.0,
            'force': 0.0
        }

        return joint_id

    def step(self, dt: float):
        for body_id, body in self.bodies.items():
            if body['is_static']:
                continue

            mass = body['mass']
            if mass <= 0:
                continue

            force = body['force'] + np.array([0, 0, -9.81]) * mass
            torque = body['torque']

            linear_acc = force / mass
            angular_acc = torque / mass

            body['linear_velocity'] += linear_acc * dt
            body['angular_velocity'] += angular_acc * dt

            body['position'] += body['linear_velocity'] * dt

            angle = np.linalg.norm(body['angular_velocity']) * dt
            if angle > 1e-6:
                axis = body['angular_velocity'] / angle
                delta_q = np.array([
                    math.cos(angle / 2),
                    axis[0] * math.sin(angle / 2),
                    axis[1] * math.sin(angle / 2),
                    axis[2] * math.sin(angle / 2)
                ])
                body['orientation'] = self._quat_multiply(body['orientation'], delta_q)

            body['force'] = np.zeros(3)
            body['torque'] = np.zeros(3)

    def apply_force(self, body_id: int, force: np.ndarray, position: np.ndarray = None):
        if body_id in self.bodies:
            self.bodies[body_id]['force'] += force
            if position is not None:
                r = position - self.bodies[body_id]['position']
                torque = np.cross(r, force)
                self.bodies[body_id]['torque'] += torque

    def apply_torque(self, body_id: int, torque: np.ndarray):
        if body_id in self.bodies:
            self.bodies[body_id]['torque'] += torque

    def set_joint_position(self, joint_id: int, position: float):
        if joint_id in self.joints:
            self.joints[joint_id]['position'] = position

    def set_joint_velocity(self, joint_id: int, velocity: float):
        if joint_id in self.joints:
            self.joints[joint_id]['velocity'] = velocity

    def set_joint_force(self, joint_id: int, force: float):
        if joint_id in self.joints:
            self.joints[joint_id]['force'] = force

    def get_contacts(self) -> List[ContactPoint]:
        contacts = []

        body_ids = list(self.bodies.keys())
        for i in range(len(body_ids)):
            for j in range(i + 1, len(body_ids)):
                body_a = self.bodies[body_ids[i]]
                body_b = self.bodies[body_ids[j]]

                dist = np.linalg.norm(body_a['position'] - body_b['position'])
                radius_a = body_a['collision_shape'].get('radius', 0.5)
                radius_b = body_b['collision_shape'].get('radius', 0.5)

                if dist < radius_a + radius_b:
                    normal = (body_b['position'] - body_a['position']) / (dist + 1e-6)
                    contacts.append(ContactPoint(
                        body_a=body_ids[i],
                        body_b=body_ids[j],
                        link_a=-1,
                        link_b=-1,
                        position=body_a['position'] + normal * radius_a,
                        normal=normal,
                        force=100.0,
                        distance=dist - radius_a - radius_b
                    ))

        return contacts

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def get_body_state(self, body_id: int) -> Optional[RobotState]:
        if body_id not in self.bodies:
            return None

        body = self.bodies[body_id]
        return RobotState(
            joint_positions=np.zeros(0),
            joint_velocities=np.zeros(0),
            joint_torques=np.zeros(0),
            base_position=body['position'],
            base_orientation=body['orientation'],
            base_linear_velocity=body['linear_velocity'],
            base_angular_velocity=body['angular_velocity']
        )


class RobotModel:
    def __init__(self, urdf_path: str = None):
        self.urdf_path = urdf_path
        self.joints: List[Dict] = []
        self.links: List[Dict] = []
        self.actuated_joints: List[int] = []
        self.end_effectors: List[str] = []
        self.base_link = "base_link"

        if urdf_path:
            self._parse_urdf(urdf_path)

    def _parse_urdf(self, urdf_path: str):
        pass

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([j.get('lower', -math.pi) for j in self.joints if j.get('type') != 'fixed'])
        upper = np.array([j.get('upper', math.pi) for j in self.joints if j.get('type') != 'fixed'])
        return lower, upper

    def get_dof(self) -> int:
        return len([j for j in self.joints if j.get('type') != 'fixed'])

    def forward_kinematics(self, joint_positions: np.ndarray) -> Dict[str, np.ndarray]:
        poses = {}
        return poses

    def inverse_kinematics(
        self,
        target_pose: np.ndarray,
        seed: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        return []

    def jacobian(self, joint_positions: np.ndarray, link_name: str) -> np.ndarray:
        return np.zeros((6, self.get_dof()))

    def dynamics(self, joint_positions: np.ndarray, joint_velocities: np.ndarray, joint_accelerations: np.ndarray) -> np.ndarray:
        return np.zeros(self.get_dof())


class SensorSimulator:
    def __init__(self, physics_engine: PhysicsEngine):
        self.physics_engine = physics_engine
        self.cameras: Dict[str, Dict] = {}
        self.lidars: Dict[str, Dict] = {}
        self.imus: Dict[str, Dict] = {}

    def add_camera(
        self,
        name: str,
        body_id: int,
        offset: np.ndarray,
        orientation: np.ndarray,
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 100.0
    ):
        self.cameras[name] = {
            'body_id': body_id,
            'offset': offset,
            'orientation': orientation,
            'width': width,
            'height': height,
            'fov': fov,
            'near': near,
            'far': far,
            'intrinsics': self._compute_intrinsics(width, height, fov)
        }

    def add_lidar(
        self,
        name: str,
        body_id: int,
        offset: np.ndarray,
        orientation: np.ndarray,
        num_beams: int = 64,
        h_fov: float = 360.0,
        v_fov: float = 30.0,
        range_min: float = 0.1,
        range_max: float = 100.0
    ):
        self.lidars[name] = {
            'body_id': body_id,
            'offset': offset,
            'orientation': orientation,
            'num_beams': num_beams,
            'h_fov': h_fov,
            'v_fov': v_fov,
            'range_min': range_min,
            'range_max': range_max
        }

    def add_imu(
        self,
        name: str,
        body_id: int,
        offset: np.ndarray,
        noise_params: Dict[str, float] = None
    ):
        self.imus[name] = {
            'body_id': body_id,
            'offset': offset,
            'noise': noise_params or {
                'accel_noise': 0.01,
                'gyro_noise': 0.001,
                'accel_bias': np.zeros(3),
                'gyro_bias': np.zeros(3)
            }
        }

    def _compute_intrinsics(self, width: int, height: int, fov: float) -> np.ndarray:
        fx = width / (2 * math.tan(math.radians(fov) / 2))
        fy = fx
        cx = width / 2
        cy = height / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def get_sensor_data(self) -> SensorData:
        camera_data = None
        depth_data = None
        lidar_data = None
        imu_data = {}

        for name, cam in self.cameras.items():
            body = self.physics_engine.bodies.get(cam['body_id'])
            if body:
                pass

        for name, lidar in self.lidars.items():
            body = self.physics_engine.bodies.get(lidar['body_id'])
            if body:
                pass

        for name, imu in self.imus.items():
            body = self.physics_engine.bodies.get(imu['body_id'])
            if body:
                accel = np.array([0, 0, -9.81])
                gyro = body['angular_velocity']

                noise = imu['noise']
                accel += np.random.randn(3) * noise['accel_noise'] + noise['accel_bias']
                gyro += np.random.randn(3) * noise['gyro_noise'] + noise['gyro_bias']

                imu_data[name] = {
                    'accelerometer': accel,
                    'gyroscope': gyro
                }

        return SensorData(
            camera=camera_data,
            depth=depth_data,
            lidar=lidar_data,
            imu=imu_data
        )


class Simulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.physics_engine = PhysicsEngine(config)
        self.robot_models: Dict[str, RobotModel] = {}
        self.sensor_simulator = SensorSimulator(self.physics_engine)
        self.environment_objects: Dict[int, Dict] = {}
        self.time = 0.0
        self.running = False

    def load_robot(self, name: str, urdf_path: str, base_position: np.ndarray = None, base_orientation: np.ndarray = None):
        robot_model = RobotModel(urdf_path)
        self.robot_models[name] = robot_model

        if base_position is None:
            base_position = np.zeros(3)
        if base_orientation is None:
            base_orientation = np.array([1, 0, 0, 0])

        return robot_model

    def add_object(self, name: str, position: np.ndarray, collision_shape: Dict, visual_shape: Dict = None, mass: float = 1.0, is_static: bool = False) -> int:
        body_id = self.physics_engine.add_rigid_body(
            mass=mass if not is_static else 0,
            position=position,
            orientation=np.array([1, 0, 0, 0]),
            collision_shape=collision_shape,
            visual_shape=visual_shape,
            is_static=is_static
        )
        self.environment_objects[body_id] = {'name': name, 'mass': mass}
        return body_id

    def step(self, dt: float = None):
        dt = dt or self.config.timestep
        self.physics_engine.step(dt)
        self.time += dt

    def get_robot_state(self, robot_name: str) -> Optional[RobotState]:
        return None

    def set_joint_positions(self, robot_name: str, positions: np.ndarray):
        pass

    def set_joint_velocities(self, robot_name: str, velocities: np.ndarray):
        pass

    def apply_joint_torques(self, robot_name: str, torques: np.ndarray):
        pass

    def get_sensor_data(self) -> SensorData:
        return self.sensor_simulator.get_sensor_data()

    def run(self, duration: float, control_callback: Callable[[float], None] = None):
        self.running = True
        num_steps = int(duration / self.config.timestep)

        for i in range(num_steps):
            if not self.running:
                break

            if control_callback:
                control_callback(self.time)

            self.step()

        self.running = False

    def stop(self):
        self.running = False


class SimulationScenario:
    def __init__(self, name: str):
        self.name = name
        self.robots: List[Dict] = []
        self.objects: List[Dict] = []
        self.tasks: List[Dict] = []
        self.duration: float = 10.0

    def add_robot(self, name: str, urdf: str, position: np.ndarray, orientation: np.ndarray = None):
        self.robots.append({
            'name': name,
            'urdf': urdf,
            'position': position,
            'orientation': orientation or np.array([1, 0, 0, 0])
        })

    def add_object(self, name: str, position: np.ndarray, shape: Dict, mass: float = 1.0, is_static: bool = False):
        self.objects.append({
            'name': name,
            'position': position,
            'shape': shape,
            'mass': mass,
            'is_static': is_static
        })

    def add_task(self, task_type: str, params: Dict):
        self.tasks.append({
            'type': task_type,
            'params': params
        })


class ScenarioRunner:
    def __init__(self, simulator: Simulator):
        self.simulator = simulator
        self.current_scenario: Optional[SimulationScenario] = None
        self.results: List[Dict] = []

    def load_scenario(self, scenario: SimulationScenario):
        self.current_scenario = scenario

        for robot in scenario.robots:
            self.simulator.load_robot(
                robot['name'],
                robot['urdf'],
                robot['position'],
                robot['orientation']
            )

        for obj in scenario.objects:
            self.simulator.add_object(
                obj['name'],
                obj['position'],
                obj['shape'],
                obj.get('visual_shape'),
                obj.get('mass', 1.0),
                obj.get('is_static', False)
            )

    def run(self, control_callback: Callable[[float, Dict], None] = None) -> List[Dict]:
        if not self.current_scenario:
            return []

        results = []

        def step_callback(t):
            if control_callback:
                state = {
                    'time': t,
                    'robots': {}
                }
                for robot in self.current_scenario.robots:
                    state['robots'][robot['name']] = self.simulator.get_robot_state(robot['name'])
                control_callback(t, state)

            if self.current_scenario.tasks:
                for task in self.current_scenario.tasks:
                    pass

        self.simulator.run(self.current_scenario.duration, step_callback)
        return results


class SimulationVisualizer:
    @staticmethod
    def draw_scene(
        simulator: Simulator,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        for body_id, body in simulator.physics_engine.bodies.items():
            pos = body['position']
            x = int(pos[0] * 50 + image_size[0] // 2)
            y = int(image_size[1] - pos[1] * 50 - image_size[1] // 2)

            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                color = (128, 128, 128) if body['is_static'] else (0, 255, 0)
                cv2.circle(vis, (x, y), 10, color, -1)

        return vis

    @staticmethod
    def draw_robot_state(state: RobotState, image_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        for i, (pos, vel, tor) in enumerate(zip(state.joint_positions, state.joint_velocities, state.joint_torques)):
            y = 30 + i * 60
            cv2.putText(vis, f"Joint {i}: pos={pos:.3f}, vel={vel:.3f}, tor={tor:.3f}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis


class SimulationMetrics:
    @staticmethod
    def compute_tracking_error(
        desired: List[np.ndarray],
        actual: List[np.ndarray]
    ) -> Dict[str, float]:
        if len(desired) != len(actual):
            return {}

        pos_errors = []
        vel_errors = []

        for d, a in zip(desired, actual):
            pos_errors.append(np.linalg.norm(d - a))

        return {
            'mean_pos_error': float(np.mean(pos_errors)),
            'max_pos_error': float(np.max(pos_errors)),
            'rmse_pos': float(np.sqrt(np.mean(np.array(pos_errors)**2)))
        }

    @staticmethod
    def compute_energy_consumption(torques: List[np.ndarray], velocities: List[np.ndarray], dt: float) -> float:
        total_energy = 0.0
        for tau, vel in zip(torques, velocities):
            total_energy += np.sum(np.abs(tau * vel)) * dt
        return total_energy

    @staticmethod
    def compute_stability_margin(contacts: List, com: np.ndarray) -> float:
        return 0.0


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
