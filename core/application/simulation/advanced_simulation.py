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


class SimulationType(Enum):
    PHYSICS = "physics"
    SENSOR = "sensor"
    ENVIRONMENT = "environment"
    SCENARIO = "scenario"
    DIGITAL_TWIN = "digital_twin"
    MONTE_CARLO = "monte_carlo"
    HARDWARE_IN_LOOP = "hardware_in_loop"
    SOFTWARE_IN_LOOP = "software_in_loop"


class SensorType(Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    DEPTH = "depth"
    THERMAL = "thermal"
    EVENT = "event"
    SONAR = "sonar"
    ULTRASONIC = "ultrasonic"


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    simulation_type: SimulationType = SimulationType.PHYSICS
    timestep: float = 1.0 / 240.0
    duration: float = 10.0
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    enable_collision: bool = True
    enable_friction: bool = True
    random_seed: int = 42
    real_time_factor: float = 1.0


@dataclass(frozen=True, slots=True)
class EntityState:
    entity_id: int
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SensorReading:
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    data: Any
    noise_params: Dict[str, float] = field(default_factory=dict)
    ground_truth: Optional[Any] = None
    frame_id: int = 0


@dataclass(frozen=True, slots=True)
class SimulationResult:
    states: List[EntityState]
    sensor_readings: List[SensorReading]
    events: List[Dict[str, Any]]
    metrics: Dict[str, float]
    config: SimulationConfig


class PhysicsEngine:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.entities: Dict[int, EntityState] = {}
        self.forces: Dict[int, np.ndarray] = {}
        self.torques: Dict[int, np.ndarray] = {}
        self.entity_id_counter = 0
        self.collision_pairs: Set[Tuple[int, int]] = set()

        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

    def add_entity(
        self,
        mass: float = 1.0,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        orientation: np.ndarray = None,
        angular_velocity: np.ndarray = None,
        inertia_tensor: np.ndarray = None
    ) -> int:
        entity_id = self.entity_id_counter
        self.entity_id_counter += 1

        if position is None:
            position = np.zeros(3)
        if velocity is None:
            velocity = np.zeros(3)
        if orientation is None:
            orientation = np.array([1, 0, 0, 0])
        if angular_velocity is None:
            angular_velocity = np.zeros(3)
        if inertia_tensor is None:
            inertia_tensor = np.eye(3) * mass

        self.entities[entity_id] = EntityState(
            entity_id=entity_id,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            timestamp=0.0
        )
        self.forces[entity_id] = np.zeros(3)
        self.torques[entity_id] = np.zeros(3)

        return entity_id

    def remove_entity(self, entity_id: int):
        if entity_id in self.entities:
            del self.entities[entity_id]
            del self.forces[entity_id]
            del self.torques[entity_id]

    def apply_force(self, entity_id: int, force: np.ndarray, position: np.ndarray = None):
        if entity_id in self.forces:
            self.forces[entity_id] += force
            if position is not None and entity_id in self.entities:
                entity = self.entities[entity_id]
                r = position - entity.position
                torque = np.cross(r, force)
                self.torques[entity_id] += torque

    def apply_torque(self, entity_id: int, torque: np.ndarray):
        if entity_id in self.torques:
            self.torques[entity_id] += torque

    def step(self, dt: float = None):
        dt = dt or self.config.timestep

        for entity_id, entity in self.entities.items():
            force = self.forces.get(entity_id, np.zeros(3))
            torque = self.torques.get(entity_id, np.zeros(3))

            if entity.metadata.get('mass', 1.0) <= 0:
                continue

            mass = entity.metadata.get('mass', 1.0)
            accel = force / mass
            angular_accel = torque / mass

            new_velocity = entity.velocity + accel * dt
            angular_vel = entity.angular_velocity + angular_accel * dt

            new_position = entity.position + new_velocity * dt

            angle = np.linalg.norm(angular_vel) * dt
            if angle > 1e-6:
                axis = angular_vel / np.linalg.norm(angular_vel)
                delta_q = np.array([
                    math.cos(angle / 2),
                    axis[0] * math.sin(angle / 2),
                    axis[1] * math.sin(angle / 2),
                    axis[2] * math.sin(angle / 2)
                ])
                new_orientation = self._quat_multiply(entity.orientation, delta_q)
            else:
                new_orientation = entity.orientation

            self.entities[entity_id] = EntityState(
                entity_id=entity_id,
                position=new_position,
                velocity=new_velocity,
                orientation=new_orientation,
                angular_velocity=angular_vel,
                timestamp=entity.timestamp + dt,
                metadata=entity.metadata
            )

        self._detect_collisions()
        self._resolve_collisions()

        self.forces = {k: np.zeros(3) for k in self.forces}
        self.torques = {k: np.zeros(3) for k in self.torques}

    def _detect_collisions(self):
        entity_ids = list(self.entities.keys())
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                id_a, id_b = entity_ids[i], entity_ids[j]
                entity_a = self.entities[id_a]
                entity_b = self.entities[id_b]

                dist = np.linalg.norm(entity_a.position - entity_b.position)
                radius_a = entity_a.metadata.get('radius', 0.5)
                radius_b = entity_b.metadata.get('radius', 0.5)

                if dist < radius_a + radius_b:
                    self.collision_pairs.add((id_a, id_b))

    def _resolve_collisions(self):
        for id_a, id_b in self.collision_pairs:
            entity_a = self.entities[id_a]
            entity_b = self.entities[id_b]

            normal = entity_b.position - entity_a.position
            dist = np.linalg.norm(normal)
            if dist < 1e-6:
                continue
            normal = normal / dist

            mass_a = entity_a.metadata.get('mass', 1.0)
            mass_b = entity_b.metadata.get('mass', 1.0)
            inv_mass_sum = 1.0 / mass_a + 1.0 / mass_b

            rel_vel = entity_b.velocity - entity_a.velocity
            vel_along_normal = np.dot(rel_vel, normal)

            if vel_along_normal > 0:
                continue

            restitution = min(
                entity_a.metadata.get('restitution', 0.5),
                entity_b.metadata.get('restitution', 0.5)
            )

            j = -(1 + restitution) * vel_along_normal
            j /= inv_mass_sum

            impulse = normal * j

            new_vel_a = entity_a.velocity - impulse / mass_a
            new_vel_b = entity_b.velocity + impulse / mass_b

            self.entities[id_a] = EntityState(
                entity_id=id_a,
                position=entity_a.position,
                velocity=new_vel_a,
                orientation=entity_a.orientation,
                angular_velocity=entity_a.angular_velocity,
                timestamp=entity_a.timestamp,
                metadata=entity_a.metadata
            )
            self.entities[id_b] = EntityState(
                entity_id=id_b,
                position=entity_b.position,
                velocity=new_vel_b,
                orientation=entity_b.orientation,
                angular_velocity=entity_b.angular_velocity,
                timestamp=entity_b.timestamp,
                metadata=entity_b.metadata
            )

        self.collision_pairs.clear()

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def get_entity_state(self, entity_id: int) -> Optional[EntityState]:
        return self.entities.get(entity_id)


class CameraSensor:
    def __init__(
        self,
        sensor_id: str,
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 100.0,
        noise_params: Dict[str, float] = None
    ):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.noise_params = noise_params or {}
        self.intrinsics = self._compute_intrinsics(width, height, fov)

    def _compute_intrinsics(self, width: int, height: int, fov: float) -> np.ndarray:
        fx = width / (2 * math.tan(math.radians(fov) / 2))
        fy = fx
        cx = width / 2
        cy = height / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def render(self, entities: Dict[int, EntityState], config: SimulationConfig) -> SensorReading:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for entity_id, entity in entities.items():
            if 'radius' not in entity.metadata:
                continue

            pos_cam = entity.position
            if pos_cam[2] <= 0:
                continue

            u = int(self.intrinsics[0, 0] * pos_cam[0] / pos_cam[2] + self.intrinsics[0, 2])
            v = int(self.intrinsics[1, 1] * pos_cam[1] / pos_cam[2] + self.intrinsics[1, 2])
            radius_px = int(self.intrinsics[0, 0] * entity.metadata['radius'] / pos_cam[2])

            if 0 <= u < self.width and 0 <= v < self.height:
                cv2.circle(frame, (u, v), radius_px, (255, 255, 255), -1)

        if self.noise_params.get('noise_std', 0) > 0:
            noise = np.random.randn(self.height, self.width, 3) * self.noise_params['noise_std']
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=SensorType.CAMERA,
            timestamp=time.time(),
            data=frame,
            noise_params=self.noise_params
        )

    def _compute_intrinsics(self, width: int, height: int, fov: float) -> np.ndarray:
        fx = width / (2 * math.tan(math.radians(fov) / 2))
        fy = fx
        cx = width / 2
        cy = height / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


class LidarSensor:
    def __init__(
        self,
        sensor_id: str,
        num_beams: int = 64,
        h_fov: float = 360.0,
        v_fov: float = 30.0,
        range_min: float = 0.1,
        range_max: float = 100.0,
        noise_params: Dict[str, float] = None
    ):
        self.sensor_id = sensor_id
        self.num_beams = num_beams
        self.h_fov = h_fov
        self.v_fov = v_fov
        self.range_min = range_min
        self.range_max = range_max
        self.noise_params = noise_params or {}

    def scan(self, entities: Dict[int, EntityState], config: SimulationConfig) -> SensorReading:
        points = []

        for entity_id, entity in entities.items():
            if 'radius' not in entity.metadata:
                continue

            rel_pos = entity.position
            dist = np.linalg.norm(rel_pos)

            if dist < self.range_min or dist > self.range_max:
                continue

            angle = math.atan2(rel_pos[1], rel_pos[0])
            elev = math.asin(rel_pos[2] / dist)

            num_points = max(10, int(entity.metadata['radius'] / 0.1))
            for _ in range(num_points):
                angle_noise = angle + np.random.randn() * 0.01
                elev_noise = elev + np.random.randn() * 0.01
                dist_noise = dist + np.random.randn() * self.noise_params.get('noise_std', 0.01)

                x = dist_noise * math.cos(elev_noise) * math.cos(angle_noise)
                y = dist_noise * math.cos(elev_noise) * math.sin(angle_noise)
                z = dist_noise * math.sin(elev_noise)

                points.append([x, y, z])

        points_array = np.array(points) if points else np.zeros((0, 3))

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=SensorType.LIDAR,
            timestamp=time.time(),
            data=points_array,
            noise_params=self.noise_params
        )


class ImuSensor:
    def __init__(
        self,
        sensor_id: str,
        noise_params: Dict[str, float] = None
    ):
        self.sensor_id = sensor_id
        self.noise_params = noise_params or {}

    def read(self, entity: EntityState) -> SensorReading:
        accel = np.array([0, 0, -9.81])
        gyro = entity.angular_velocity

        noise = self.noise_params
        if noise.get('accel_noise', 0) > 0:
            accel += np.random.randn(3) * noise['accel_noise']
        if noise.get('gyro_noise', 0) > 0:
            gyro += np.random.randn(3) * noise['gyro_noise']
        if 'accel_bias' in noise:
            accel += noise['accel_bias']
        if 'gyro_bias' in noise:
            gyro += noise['gyro_bias']

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=SensorType.IMU,
            timestamp=time.time(),
            data={'accelerometer': accel, 'gyroscope': gyro},
            noise_params=noise
        )


class GpsSensor:
    def __init__(
        self,
        sensor_id: str,
        noise_params: Dict[str, float] = None
    ):
        self.sensor_id = sensor_id
        self.noise_params = noise_params or {}

    def read(self, entity: EntityState) -> SensorReading:
        position = entity.position.copy()
        noise = self.noise_params.get('pos_noise', 1.0)
        position += np.random.randn(3) * noise

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=SensorType.GPS,
            timestamp=time.time(),
            data=position,
            noise_params=self.noise_params
        )


class SensorSuite:
    def __init__(self):
        self.sensors: Dict[str, Any] = {}

    def add_sensor(self, sensor: Any):
        self.sensors[sensor.sensor_id] = sensor

    def remove_sensor(self, sensor_id: str):
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]

    def get_readings(self, entities: Dict[int, EntityState], config: SimulationConfig) -> List[SensorReading]:
        readings = []
        for sensor in self.sensors.values():
            if isinstance(sensor, CameraSensor):
                readings.append(sensor.render(entities, config))
            elif isinstance(sensor, LidarSensor):
                readings.append(sensor.scan(entities, config))
            elif isinstance(sensor, ImuSensor):
                for entity in entities.values():
                    if entity.metadata.get('has_imu', False):
                        readings.append(sensor.read(entity))
            elif isinstance(sensor, GpsSensor):
                for entity in entities.values():
                    if entity.metadata.get('has_gps', False):
                        readings.append(sensor.read(entity))
        return readings


class SimulationScenario:
    def __init__(self, name: str):
        self.name = name
        self.entities: List[Dict] = []
        self.sensors: List[Dict] = []
        self.events: List[Dict] = []
        self.duration: float = 10.0
        self.config: Optional[SimulationConfig] = None

    def add_entity(
        self,
        name: str,
        mass: float = 1.0,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        radius: float = 0.5,
        restitution: float = 0.5,
        has_imu: bool = False,
        has_gps: bool = False,
        metadata: Dict = None
    ):
        entity_data = {
            'name': name,
            'mass': mass,
            'position': position if position is not None else np.zeros(3),
            'velocity': velocity if velocity is not None else np.zeros(3),
            'radius': radius,
            'restitution': restitution,
            'has_imu': has_imu,
            'has_gps': has_gps,
            'metadata': metadata or {}
        }
        self.entities.append(entity_data)

    def add_camera(self, sensor_id: str, width: int = 640, height: int = 480, fov: float = 60.0, **kwargs):
        self.sensors.append({
            'type': 'camera',
            'sensor_id': sensor_id,
            'width': width,
            'height': height,
            'fov': fov,
            **kwargs
        })

    def add_lidar(self, sensor_id: str, num_beams: int = 64, **kwargs):
        self.sensors.append({
            'type': 'lidar',
            'sensor_id': sensor_id,
            'num_beams': num_beams,
            **kwargs
        })

    def add_imu(self, sensor_id: str, **kwargs):
        self.sensors.append({
            'type': 'imu',
            'sensor_id': sensor_id,
            **kwargs
        })

    def add_gps(self, sensor_id: str, **kwargs):
        self.sensors.append({
            'type': 'gps',
            'sensor_id': sensor_id,
            **kwargs
        })

    def add_event(self, time: float, event_type: str, params: Dict):
        self.events.append({
            'time': time,
            'type': event_type,
            'params': params
        })


class ScenarioRunner:
    def __init__(self, scenario: SimulationScenario):
        self.scenario = scenario
        self.simulator = PhysicsEngine(scenario.config or SimulationConfig())
        self.sensor_suite = SensorSuite()
        self.results: List[SimulationResult] = []

    def setup(self):
        for entity_data in self.scenario.entities:
            self.simulator.add_entity(
                mass=entity_data['mass'],
                position=entity_data['position'],
                velocity=entity_data['velocity'],
                orientation=np.array([1, 0, 0, 0]),
                metadata=entity_data
            )

        for sensor_data in self.scenario.sensors:
            sensor_type = sensor_data['type']
            if sensor_type == 'camera':
                self.sensor_suite.add_sensor(CameraSensor(
                    sensor_data['sensor_id'],
                    sensor_data.get('width', 640),
                    sensor_data.get('height', 480),
                    sensor_data.get('fov', 60.0)
                ))
            elif sensor_type == 'lidar':
                self.sensor_suite.add_sensor(LidarSensor(
                    sensor_data['sensor_id'],
                    sensor_data.get('num_beams', 64)
                ))
            elif sensor_type == 'imu':
                self.sensor_suite.add_sensor(ImuSensor(sensor_data['sensor_id']))
            elif sensor_type == 'gps':
                self.sensor_suite.add_sensor(GpsSensor(sensor_data['sensor_id']))

    def run(self) -> SimulationResult:
        self.setup()

        states = []
        sensor_readings = []
        events = []

        num_steps = int(self.scenario.duration / self.simulator.config.timestep)

        for step in range(num_steps):
            current_time = step * self.simulator.config.timestep

            for event in self.scenario.events:
                if abs(event['time'] - current_time) < self.simulator.config.timestep:
                    events.append(event)

            self.simulator.step()
            sensor_readings.extend(self.sensor_suite.get_readings(self.simulator.entities, self.simulator.config))

            for entity_id, entity in self.simulator.entities.items():
                states.append(entity)

        return SimulationResult(
            states=states,
            sensor_readings=sensor_readings,
            events=events,
            metrics={},
            config=self.scenario.config or SimulationConfig()
        )


class MonteCarloSimulator:
    def __init__(
        self,
        base_scenario: SimulationScenario,
        num_samples: int = 100,
        parameter_ranges: Dict[str, Tuple[float, float]] = None
    ):
        self.base_scenario = base_scenario
        self.num_samples = num_samples
        self.parameter_ranges = parameter_ranges or {}

    def run(self) -> List[SimulationResult]:
        results = []

        for i in range(self.num_samples):
            scenario = self._generate_varied_scenario()
            runner = ScenarioRunner(scenario)
            result = runner.run()
            results.append(result)

        return results

    def _generate_varied_scenario(self) -> SimulationScenario:
        scenario = SimulationScenario(f"{self.base_scenario.name}_mc_{random.randint(0, 10000)}")
        scenario.duration = self.base_scenario.duration
        scenario.config = self.base_scenario.config

        for entity in self.base_scenario.entities:
            varied_entity = entity.copy()
            if 'position_noise' in self.parameter_ranges:
                noise_range = self.parameter_ranges['position_noise']
                varied_entity['position'] = entity['position'] + np.random.uniform(
                    noise_range[0], noise_range[1], 3
                )
            if 'velocity_noise' in self.parameter_ranges:
                noise_range = self.parameter_ranges['velocity_noise']
                varied_entity['velocity'] = entity['velocity'] + np.random.uniform(
                    noise_range[0], noise_range[1], 3
                )
            if 'mass_noise' in self.parameter_ranges:
                noise_range = self.parameter_ranges['mass_noise']
                varied_entity['mass'] = entity['mass'] * np.random.uniform(
                    noise_range[0], noise_range[1]
                )
            scenario.entities.append(varied_entity)

        for sensor in self.base_scenario.sensors:
            scenario.sensors.append(sensor)

        return scenario

    def analyze_results(self, results: List[SimulationResult]) -> Dict[str, Any]:
        if not results:
            return {}

        all_final_positions = []
        for result in results:
            final_states = {}
            for state in result.states:
                if state.timestamp == max(s.timestamp for s in result.states):
                    final_states[state.entity_id] = state.position
            all_final_positions.append(final_states)

        stats = {}
        if all_final_positions:
            entity_ids = set()
            for positions in all_final_positions:
                entity_ids.update(positions.keys())

            for eid in entity_ids:
                positions = np.array([p[eid] for p in all_final_positions if eid in p])
                stats[f'entity_{eid}'] = {
                    'mean_position': np.mean(positions, axis=0).tolist(),
                    'std_position': np.std(positions, axis=0).tolist(),
                    'min_position': np.min(positions, axis=0).tolist(),
                    'max_position': np.max(positions, axis=0).tolist()
                }

        return stats


class SimulationVisualizer:
    @staticmethod
    def draw_trajectory(
        states: List[EntityState],
        image_size: Tuple[int, int] = (800, 600),
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if len(states) < 2:
            return vis

        positions = np.array([s.position[:2] for s in states])
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        prev_pt = None
        for state in states:
            x = int((state.position[0] - min_coords[0]) * scale + image_size[0] * 0.1)
            y = int(image_size[1] - (state.position[1] - min_coords[1]) * scale - image_size[1] * 0.1)

            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                cv2.circle(vis, (x, y), 2, color, -1)
                if prev_pt is not None:
                    cv2.line(vis, prev_pt, (x, y), color, 1)
                prev_pt = (x, y)

        return vis

    @staticmethod
    def draw_sensor_readings(
        readings: List[SensorReading],
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        for reading in readings:
            if reading.sensor_type == SensorType.LIDAR and isinstance(reading.data, np.ndarray):
                points = reading.data[:, :2]
                if len(points) == 0:
                    continue

                min_coords = np.min(points, axis=0)
                max_coords = np.max(points, axis=0)
                range_coords = max_coords - min_coords
                range_coords[range_coords == 0] = 1

                scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

                for point in points:
                    x = int((point[0] - min_coords[0]) * scale + image_size[0] * 0.1)
                    y = int(image_size[1] - (point[1] - min_coords[1]) * scale - image_size[1] * 0.1)

                    if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                        cv2.circle(vis, (x, y), 1, (255, 0, 0), -1)

        return vis


class SimulationMetrics:
    @staticmethod
    def compute_tracking_error(
        ground_truth: List[EntityState],
        estimated: List[EntityState]
    ) -> Dict[str, float]:
        if len(ground_truth) != len(estimated):
            return {}

        pos_errors = []
        vel_errors = []

        for gt, est in zip(ground_truth, estimated):
            pos_errors.append(np.linalg.norm(gt.position - est.position))
            vel_errors.append(np.linalg.norm(gt.velocity - est.velocity))

        return {
            'mean_pos_error': float(np.mean(pos_errors)),
            'max_pos_error': float(np.max(pos_errors)),
            'rmse_pos': float(np.sqrt(np.mean(np.array(pos_errors)**2))),
            'mean_vel_error': float(np.mean(vel_errors)),
            'rmse_vel': float(np.sqrt(np.mean(np.array(vel_errors)**2)))
        }

    @staticmethod
    def compute_energy_consumption(
        torques: List[np.ndarray],
        velocities: List[np.ndarray],
        dt: float
    ) -> float:
        total_energy = 0.0
        for tau, vel in zip(torques, velocities):
            total_energy += np.sum(np.abs(tau * vel)) * dt
        return total_energy

    @staticmethod
    def compute_collision_rate(states: List[EntityState]) -> float:
        return 0.0


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet