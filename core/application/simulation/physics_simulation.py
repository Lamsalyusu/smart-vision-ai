from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque
import copy


class SimulationType(Enum):
    PHYSICS = "physics"
    SENSOR = "sensor"
    ENVIRONMENT = "environment"
    SCENARIO = "scenario"
    DIGITAL_TWIN = "digital_twin"
    MONTE_CARLO = "monte_carlo"
    HARDWARE_IN_LOOP = "hardware_in_loop"


class SensorModelType(Enum):
    IDEAL = "ideal"
    NOISY = "noisy"
    REALISTIC = "realistic"
    DEGRADED = "degraded"
    FAULTY = "faulty"


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    timestep: float = 0.01
    duration: float = 10.0
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, -9.81, 0]))
    enable_collision: bool = True
    enable_friction: bool = True
    random_seed: int = 42


@dataclass(frozen=True, slots=True)
class EntityState:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SensorReading:
    sensor_id: str
    sensor_type: str
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
            id=entity_id,
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

    def apply_force(self, entity_id: int, force: np.ndarray):
        if entity_id in self.forces:
            self.forces[entity_id] += force

    def apply_torque(self, entity_id: int, torque: np.ndarray):
        if entity_id in self.torques:
            self.torques[entity_id] += torque

    def step(self, dt: float = None):
        dt = dt or self.config.timestep

        for entity_id, entity in self.entities.items():
            if entity_id not in self.forces:
                continue

            force = self.forces[entity_id]
            torque = self.torques[entity_id]

            accel = force / entity.metadata.get('mass', 1.0)
            new_velocity = entity.velocity + accel * dt

            if self.config.enable_friction:
                new_velocity *= 0.99

            new_position = entity.position + new_velocity * dt

            angular_accel = np.linalg.inv(np.eye(3)) @ torque
            new_angular_velocity = entity.angular_velocity + angular_accel * dt

            angle = np.linalg.norm(new_angular_velocity) * dt
            if angle > 1e-6:
                axis = new_angular_velocity / np.linalg.norm(new_angular_velocity)
                delta_q = np.array([math.cos(angle/2), axis[0]*math.sin(angle/2), axis[1]*math.sin(angle/2), axis[2]*math.sin(angle/2)])
                new_orientation = self._quat_multiply(entity.orientation, delta_q)
            else:
                new_orientation = entity.orientation

            self.entities[entity_id] = EntityState(
                id=entity_id,
                position=new_position,
                velocity=new_velocity,
                orientation=new_orientation,
                angular_velocity=new_angular_velocity,
                timestamp=entity.timestamp + dt
            )

        self._detect_collisions()
        self._resolve_collisions()
        self._clear_forces()

    def _detect_collisions(self):
        self.collision_pairs.clear()
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

            j = -(1.0 + restitution) * vel_along_normal
            j /= inv_mass_sum

            impulse = normal * j

            if not entity_a.metadata.get('is_static', False):
                entity_a.velocity = entity_a.velocity - impulse / mass_a
            if not entity_b.metadata.get('is_static', False):
                entity_b.velocity = entity_b.velocity + impulse / mass_b

        self.collision_pairs.clear()

    def _clear_forces(self):
        for entity_id in self.forces:
            self.forces[entity_id] = np.zeros(3)
            self.torques[entity_id] = np.zeros(3)

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
            noise_params=self.noise_params
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


class ScenarioGenerator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.scenarios: List[Dict[str, Any]] = []

    def add_scenario(self, scenario: Dict[str, Any]):
        self.scenarios.append(scenario)

    def generate_random_scenario(
        self,
        num_entities: int = 10,
        area_size: float = 100.0
    ) -> Dict[str, Any]:
        entities = []
        for i in range(num_entities):
            entities.append({
                'id': i,
                'type': random.choice(['vehicle', 'pedestrian', 'static']),
                'position': np.random.uniform(-area_size/2, area_size/2, 3),
                'velocity': np.random.uniform(-10, 10, 3),
                'mass': random.uniform(50, 2000),
                'length': random.uniform(4, 10),
                'width': random.uniform(1.8, 2.5)
            })

        return {
            'entities': entities,
            'duration': self.config.duration,
            'weather': {
                'fog_density': random.uniform(0, 0.1),
                'rain_intensity': random.uniform(0, 0.5)
            },
            'lighting': {
                'sun_direction': [0, -1, 0],
                'ambient': 0.3
            }
        }

    def generate_traffic_scenario(
        self,
        num_vehicles: int = 20,
        num_pedestrians: int = 10,
        road_length: float = 500.0,
        road_width: float = 12.0
    ) -> Dict[str, Any]:
        entities = []

        for i in range(num_vehicles):
            lane = random.randint(0, 2)
            pos_x = random.uniform(0, road_length)
            pos_y = (lane - 1) * (road_width / 3)
            speed = random.uniform(10, 30)

            entities.append({
                'id': i,
                'type': 'vehicle',
                'position': np.array([pos_x, pos_y, 0]),
                'velocity': np.array([speed, 0, 0]),
                'mass': random.uniform(1000, 2000),
                'length': random.uniform(4, 10),
                'width': random.uniform(1.8, 2.5)
            })

        for i in range(num_pedestrians):
            pos_x = random.uniform(0, road_length)
            pos_y = random.uniform(-road_width/2, road_width/2)
            speed = random.uniform(0.5, 2.0)

            entities.append({
                'id': num_vehicles + i,
                'type': 'pedestrian',
                'position': np.array([pos_x, pos_y, 0]),
                'velocity': np.array([0, speed, 0]),
                'mass': 70,
                'radius': 0.5
            })

        return {
            'entities': entities,
            'duration': self.config.duration,
            'weather': {'fog_density': 0.01, 'rain_intensity': 0.0},
            'lighting': {'sun_direction': [0, -1, 0], 'ambient': 0.4}
        }


class MonteCarloSimulator:
    def __init__(
        self,
        base_simulator: PhysicsSimulator,
        num_samples: int = 1000,
        parameter_ranges: Dict[str, Tuple[float, float]] = None
    ):
        self.base_simulator = base_simulator
        self.num_samples = num_samples
        self.parameter_ranges = parameter_ranges or {}

    def run(self) -> List[SimulationResult]:
        results = []

        for i in range(self.num_samples):
            params = {}
            for param, (min_val, max_val) in self.parameter_ranges.items():
                params[param] = random.uniform(min_val, max_val)

            sim = PhysicsSimulator(self.base_simulator.config)

            for entity_id, entity in self.base_simulator.entities.items():
                sim.add_entity(
                    position=entity.position + np.random.randn(3) * params.get('position_noise', 0.1),
                    velocity=entity.velocity + np.random.randn(3) * params.get('velocity_noise', 0.5)
                )

            result = sim.simulate()
            results.append(result)

        return results

    def analyze_results(self, results: List[SimulationResult]) -> Dict[str, Any]:
        if not results:
            return {}

        all_final_positions = []
        for result in results:
            if result.states:
                final_states = {}
                for state in result.states:
                    final_states[state.id] = state.position
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


class DigitalTwin:
    def __init__(
        self,
        physical_system: Dict[str, Any],
        simulation_config: SimulationConfig
    ):
        self.physical_system = physical_system
        self.config = simulation_config
        self.simulator = PhysicsSimulator(simulation_config)
        self.sensor_simulators: Dict[str, SensorSimulator] = {}
        self.state_estimate: Optional[EntityState] = None
        self.covariance = np.eye(6)

    def add_sensor_simulator(self, name: str, simulator: SensorSimulator):
        self.sensor_simulators[name] = simulator

    def sync_from_physical(self, sensor_data: Dict[str, SensorReading]):
        for name, reading in sensor_data.items():
            if name in self.sensor_simulators:
                self._update_state_estimate(reading)

    def _update_state_estimate(self, reading: SensorReading):
        pass

    def predict(self, horizon: float) -> List[EntityState]:
        return self.simulator.simulate(horizon).states

    def calibrate(self, physical_data: List[SensorReading], simulated_data: List[SensorReading]):
        pass


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
            if reading.sensor_type == "lidar" and isinstance(reading.data, np.ndarray):
                points = reading.data[:, :2]
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


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
