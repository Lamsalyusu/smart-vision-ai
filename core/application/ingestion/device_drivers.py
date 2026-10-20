from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import random
from collections import defaultdict, deque


class DeviceType(Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    DEPTH = "depth"
    THERMAL = "thermal"
    EVENT_CAMERA = "event_camera"
    ULTRASONIC = "ultrasonic"
    SONAR = "sonar"
    MICROPHONE = "microphone"
    FORCE_TORQUE = "force_torque"
    TACTILE = "tactile"
    JOINT_ENCODER = "joint_encoder"
    WHEEL_ENCODER = "wheel_encoder"
    ACTUATOR = "actuator"
    MOTOR = "motor"
    SERVO = "servo"
    GRIPPER = "gripper"


class MessageType(Enum):
    TELEMETRY = "telemetry"
    COMMAND = "command"
    EVENT = "event"
    STATUS = "status"
    CONFIG = "config"
    CALIBRATION = "calibration"
    FIRMWARE = "firmware"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    device_id: str
    device_type: DeviceType
    interface: str
    address: str
    baud_rate: int = 115200
    parameters: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class DeviceMessage:
    message_id: str = field(default_factory=lambda: f"msg_{int(time.time()*1000000)}")
    device_id: str
    message_type: MessageType
    timestamp: float = field(default_factory=time.time)
    payload: Any = None
    sequence_number: int = 0
    requires_ack: bool = False
    timeout: float = 5.0


@dataclass(frozen=True, slots=True)
class DeviceState:
    device_id: str
    status: str
    last_update: float
    data: Any
    health: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeviceDriver(ABC):
    def __init__(self, config: DeviceConfig):
        self.config = config
        self.connected = False
        self.last_error: Optional[str] = None
        self.message_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        pass

    @abstractmethod
    def send_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        pass

    @abstractmethod
    def read_data(self) -> Optional[Any]:
        pass

    @abstractmethod
    def get_status(self) -> DeviceState:
        pass

    def register_callback(self, event: str, callback: Callable):
        self.callbacks[event].append(callback)

    def _trigger_callback(self, event: str, data: Any):
        for callback in self.callbacks[event]:
            try:
                callback(data)
            except Exception:
                pass


class CameraDriver(DeviceDriver):
    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.width = config.parameters.get('width', 640)
        self.height = config.parameters.get('height', 480)
        self.fps = config.parameters.get('fps', 30)
        self.format = config.parameters.get('format', 'RGB')
        self.camera = None

    def connect(self) -> bool:
        try:
            import cv2
            device_id = int(self.config.address) if self.config.address.isdigit() else self.config.address
            self.camera = cv2.VideoCapture(device_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            self.connected = self.camera.isOpened()
            return self.connected
        except Exception as e:
            self.last_error = str(e)
            return False

    def disconnect(self) -> bool:
        if self.camera:
            self.camera.release()
            self.camera = None
        self.connected = False
        return True

    def send_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        if not self.camera:
            return False

        try:
            import cv2
            if command == "set_resolution":
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, params.get('width', self.width))
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, params.get('height', self.height))
            elif command == "set_fps":
                self.camera.set(cv2.CAP_PROP_FPS, params.get('fps', self.fps))
            elif command == "set_exposure":
                self.camera.set(cv2.CAP_PROP_EXPOSURE, params.get('exposure', -1))
            elif command == "set_gain":
                self.camera.set(cv2.CAP_PROP_GAIN, params.get('gain', 0))
            elif command == "set_auto_exposure":
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, params.get('enabled', 1))
            elif command == "set_auto_white_balance":
                self.camera.set(cv2.CAP_PROP_AUTO_WB, params.get('enabled', 1))
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def read_data(self) -> Optional[Any]:
        if not self.camera:
            return None

        try:
            import cv2
            ret, frame = self.camera.read()
            if ret:
                return frame
        except Exception as e:
            self.last_error = str(e)
        return None

    def get_status(self) -> DeviceState:
        return DeviceState(
            device_id=self.config.device_id,
            status="connected" if self.connected else "disconnected",
            last_update=time.time(),
            data={
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "format": self.format
            },
            health=1.0 if self.connected else 0.0
        )


class LidarDriver(DeviceDriver):
    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.serial_port = None
        self.scan_data = None
        self.scan_rate = config.parameters.get('scan_rate', 10)
        self.max_range = config.parameters.get('max_range', 100.0)
        self.angle_resolution = config.parameters.get('angle_resolution', 1.0)

    def connect(self) -> bool:
        try:
            import serial
            self.serial_port = serial.Serial(
                port=self.config.address,
                baudrate=self.config.baud_rate,
                timeout=1.0
            )
            self.connected = self.serial_port.is_open
            return self.connected
        except Exception as e:
            self.last_error = str(e)
            return False

    def disconnect(self) -> bool:
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.connected = False
        return True

    def send_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        if not self.serial_port or not self.serial_port.is_open:
            return False

        try:
            if command == "start_scan":
                self.serial_port.write(b"START\n")
            elif command == "stop_scan":
                self.serial_port.write(b"STOP\n")
            elif command == "set_scan_rate":
                rate = params.get('scan_rate', self.scan_rate)
                self.serial_port.write(f"RATE {rate}\n".encode())
            elif command == "set_max_range":
                rng = params.get('max_range', self.max_range)
                self.serial_port.write(f"RANGE {rng}\n".encode())
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def read_data(self) -> Optional[Any]:
        if not self.serial_port or not self.serial_port.is_open:
            return None

        try:
            line = self.serial_port.readline().decode().strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    angle = float(parts[0])
                    distance = float(parts[1])
                    self.scan_data = {
                        'angle': angle,
                        'distance': distance,
                        'timestamp': time.time()
                    }
                    return self.scan_data
        except Exception as e:
            self.last_error = str(e)
        return None

    def get_status(self) -> DeviceState:
        return DeviceState(
            device_id=self.config.device_id,
            status="connected" if self.connected else "disconnected",
            last_update=time.time(),
            data={
                "scan_rate": self.scan_rate,
                "max_range": self.max_range,
                "angle_resolution": self.angle_resolution
            },
            health=1.0 if self.connected else 0.0
        )


class IMUDriver(DeviceDriver):
    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.serial_port = None
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.accel_scale = np.ones(3)
        self.gyro_scale = np.ones(3)
        self.sample_rate = config.parameters.get('sample_rate', 100)

    def connect(self) -> bool:
        try:
            import serial
            self.serial_port = serial.Serial(
                port=self.config.address,
                baudrate=self.config.baud_rate,
                timeout=1.0
            )
            self.connected = self.serial_port.is_open
            return self.connected
        except Exception as e:
            self.last_error = str(e)
            return False

    def disconnect(self) -> bool:
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.connected = False
        return True

    def send_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        if not self.serial_port or not self.serial_port.is_open:
            return False

        try:
            if command == "calibrate_accel":
                self._calibrate_accel()
            elif command == "calibrate_gyro":
                self._calibrate_gyro()
            elif command == "set_sample_rate":
                self.sample_rate = params.get('sample_rate', self.sample_rate)
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def _calibrate_accel(self):
        pass

    def _calibrate_gyro(self):
        pass

    def read_data(self) -> Optional[Any]:
        if not self.serial_port or not self.serial_port.is_open:
            return None

        try:
            line = self.serial_port.readline().decode().strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 6:
                    accel = np.array([float(parts[0]), float(parts[1]), float(parts[2])]) * self.accel_scale + self.accel_bias
                    gyro = np.array([float(parts[3]), float(parts[4]), float(parts[5])]) * self.gyro_scale + self.gyro_bias

                    return {
                        'accelerometer': accel,
                        'gyroscope': gyro,
                        'timestamp': time.time()
                    }
        except Exception as e:
            self.last_error = str(e)
        return None

    def get_status(self) -> DeviceState:
        return DeviceState(
            device_id=self.config.device_id,
            status="connected" if self.connected else "disconnected",
            last_update=time.time(),
            data={
                "sample_rate": self.sample_rate,
                "accel_bias": self.accel_bias.tolist(),
                "gyro_bias": self.gyro_bias.tolist()
            },
            health=1.0 if self.connected else 0.0
        )


class GPSDriver(DeviceDriver):
    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.serial_port = None
        self.last_fix = None
        self.fix_type = 0
        self.satellites = 0

    def connect(self) -> bool:
        try:
            import serial
            self.serial_port = serial.Serial(
                port=self.config.address,
                baudrate=self.config.baud_rate,
                timeout=1.0
            )
            self.connected = self.serial_port.is_open
            return self.connected
        except Exception as e:
            self.last_error = str(e)
            return False

    def disconnect(self) -> bool:
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.connected = False
        return True

    def send_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        return False

    def read_data(self) -> Optional[Any]:
        if not self.serial_port or not self.serial_port.is_open:
            return None

        try:
            line = self.serial_port.readline().decode().strip()
            if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                parts = line.split(',')
                if len(parts) >= 10:
                    lat = self._parse_coordinate(parts[2], parts[3])
                    lon = self._parse_coordinate(parts[4], parts[5])
                    self.fix_type = int(parts[6]) if parts[6] else 0
                    self.satellites = int(parts[7]) if parts[7] else 0
                    altitude = float(parts[9]) if parts[9] else 0.0

                    self.last_fix = {
                        'latitude': lat,
                        'longitude': lon,
                        'altitude': altitude,
                        'fix_type': self.fix_type,
                        'satellites': self.satellites,
                        'timestamp': time.time()
                    }
                    return self.last_fix
        except Exception as e:
            self.last_error = str(e)
        return None

    def _parse_coordinate(self, coord: str, direction: str) -> float:
        if not coord:
            return 0.0
        degrees = float(coord[:2]) if len(coord) > 4 else float(coord[:3])
        minutes = float(coord[2:]) if len(coord) > 4 else float(coord[3:])
        value = degrees + minutes / 60.0
        if direction in ['S', 'W']:
            value = -value
        return value

    def get_status(self) -> DeviceState:
        return DeviceState(
            device_id=self.config.device_id,
            status="connected" if self.connected else "disconnected",
            last_update=time.time(),
            data={
                "last_fix": self.last_fix,
                "fix_type": self.fix_type,
                "satellites": self.satellites
            },
            health=1.0 if self.connected and self.fix_type > 0 else 0.5
        )


class ActuatorDriver(DeviceDriver):
    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.position = 0.0
        self.velocity = 0.0
        self.current = 0.0
        self.target_position = 0.0
        self.target_velocity = 0.0
        self.max_position = config.parameters.get('max_position', 360.0)
        self.min_position = config.parameters.get('min_position', 0.0)
        self.max_velocity = config.parameters.get('max_velocity', 100.0)
        self.max_current = config.parameters.get('max_current', 10.0)

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def send_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        if command == "set_position":
            self.target_position = np.clip(params.get('position', 0.0), self.min_position, self.max_position)
        elif command == "set_velocity":
            self.target_velocity = np.clip(params.get('velocity', 0.0), -self.max_velocity, self.max_velocity)
        elif command == "enable":
            pass
        elif command == "disable":
            pass
        elif command == "home":
            self.target_position = self.min_position
        return True

    def read_data(self) -> Optional[Any]:
        self.position += (self.target_position - self.position) * 0.1
        self.velocity = (self.target_position - self.position) * 10.0
        return {
            'position': self.position,
            'velocity': self.velocity,
            'current': self.current,
            'target_position': self.target_position,
            'timestamp': time.time()
        }

    def get_status(self) -> DeviceState:
        return DeviceState(
            device_id=self.config.device_id,
            status="connected" if self.connected else "disconnected",
            last_update=time.time(),
            data={
                "position": self.position,
                "velocity": self.velocity,
                "target_position": self.target_position
            },
            health=1.0 if self.connected else 0.0
        )


class GripperDriver(DeviceDriver):
    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.position = 0.0
        self.force = 0.0
        self.is_grasping = False
        self.max_width = config.parameters.get('max_width', 0.1)
        self.min_width = config.parameters.get('min_width', 0.0)
        self.max_force = config.parameters.get('max_force', 50.0)

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def send_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        if command == "open":
            self.position = self.max_width
            self.is_grasping = False
        elif command == "close":
            self.position = self.min_width
            self.is_grasping = True
        elif command == "set_position":
            self.position = np.clip(params.get('position', 0.0), self.min_width, self.max_width)
            self.is_grasping = self.position < self.max_width * 0.1
        elif command == "set_force":
            pass
        return True

    def read_data(self) -> Optional[Any]:
        return {
            'position': self.position,
            'force': self.force,
            'is_grasping': self.is_grasping,
            'timestamp': time.time()
        }

    def get_status(self) -> DeviceState:
        return DeviceState(
            device_id=self.config.device_id,
            status="connected" if self.connected else "disconnected",
            last_update=time.time(),
            data={
                "position": self.position,
                "force": self.force,
                "is_grasping": self.is_grasping
            },
            health=1.0 if self.connected else 0.0
        )


class DeviceManager:
    def __init__(self):
        self.devices: Dict[str, DeviceDriver] = {}
        self.device_configs: Dict[str, DeviceConfig] = {}
        self.message_queues: Dict[str, queue.Queue] = defaultdict(queue.Queue)
        self.lock = threading.RLock()

    def add_device(self, config: DeviceConfig) -> bool:
        with self.lock:
            if config.device_id in self.devices:
                return False

            driver = self._create_driver(config)
            if not driver:
                return False

            self.devices[config.device_id] = driver
            self.device_configs[config.device_id] = config
            return True

    def _create_driver(self, config: DeviceConfig) -> Optional[DeviceDriver]:
        if config.device_type == DeviceType.CAMERA:
            return CameraDriver(config)
        elif config.device_type == DeviceType.LIDAR:
            return LidarDriver(config)
        elif config.device_type == DeviceType.IMU:
            return IMUDriver(config)
        elif config.device_type == DeviceType.GPS:
            return GPSDriver(config)
        elif config.device_type == DeviceType.ACTUATOR:
            return ActuatorDriver(config)
        elif config.device_type == DeviceType.GRIPPER:
            return GripperDriver(config)
        return None

    def remove_device(self, device_id: str) -> bool:
        with self.lock:
            if device_id in self.devices:
                self.devices[device_id].disconnect()
                del self.devices[device_id]
                del self.device_configs[device_id]
                return True
            return False

    def connect_device(self, device_id: str) -> bool:
        with self.lock:
            if device_id in self.devices:
                return self.devices[device_id].connect()
            return False

    def disconnect_device(self, device_id: str) -> bool:
        with self.lock:
            if device_id in self.devices:
                return self.devices[device_id].disconnect()
            return False

    def connect_all(self) -> Dict[str, bool]:
        with self.lock:
            results = {}
            for device_id, driver in self.devices.items():
                results[device_id] = driver.connect()
            return results

    def disconnect_all(self):
        with self.lock:
            for driver in self.devices.values():
                driver.disconnect()

    def send_command(self, device_id: str, command: str, params: Dict = None) -> bool:
        with self.lock:
            if device_id in self.devices:
                return self.devices[device_id].send_command(command, params)
            return False

    def read_data(self, device_id: str) -> Optional[Any]:
        with self.lock:
            if device_id in self.devices:
                return self.devices[device_id].read_data()
            return None

    def read_all_data(self) -> Dict[str, Any]:
        with self.lock:
            data = {}
            for device_id, driver in self.devices.items():
                data[device_id] = driver.read_data()
            return data

    def get_device_state(self, device_id: str) -> Optional[DeviceState]:
        with self.lock:
            if device_id in self.devices:
                return self.devices[device_id].get_status()
            return None

    def get_all_states(self) -> Dict[str, DeviceState]:
        with self.lock:
            return {device_id: driver.get_status() for device_id, driver in self.devices.items()}

    def register_callback(self, device_id: str, event: str, callback: Callable):
        with self.lock:
            if device_id in self.devices:
                self.devices[device_id].register_callback(event, callback)

    def get_device_config(self, device_id: str) -> Optional[DeviceConfig]:
        with self.lock:
            return self.device_configs.get(device_id)

    def list_devices(self) -> List[str]:
        with self.lock:
            return list(self.devices.keys())


class DeviceRegistry:
    def __init__(self):
        self.devices: Dict[str, DeviceConfig] = {}
        self.lock = threading.RLock()

    def register(self, config: DeviceConfig) -> bool:
        with self.lock:
            if config.device_id in self.devices:
                return False
            self.devices[config.device_id] = config
            return True

    def unregister(self, device_id: str) -> bool:
        with self.lock:
            if device_id in self.devices:
                del self.devices[device_id]
                return True
            return False

    def get_config(self, device_id: str) -> Optional[DeviceConfig]:
        with self.lock:
            return self.devices.get(device_id)

    def list_devices(self) -> List[DeviceConfig]:
        with self.lock:
            return list(self.devices.values())

    def find_by_type(self, device_type: DeviceType) -> List[DeviceConfig]:
        with self.lock:
            return [config for config in self.devices.values() if config.device_type == device_type]

    def find_by_interface(self, interface: str) -> List[DeviceConfig]:
        with self.lock:
            return [config for config in self.devices.values() if config.interface == interface]


class DeviceDiscovery:
    def __init__(self):
        self.known_devices: Dict[str, DeviceConfig] = {}
        self.scanners: Dict[str, Callable[[], List[DeviceConfig]]] = {}

    def register_scanner(self, name: str, scanner: Callable[[], List[DeviceConfig]]):
        self.scanners[name] = scanner

    def scan(self) -> List[DeviceConfig]:
        found = []
        for name, scanner in self.scanners.items():
            try:
                devices = scanner()
                for device in devices:
                    if device.device_id not in self.known_devices:
                        self.known_devices[device.device_id] = device
                        found.append(device)
            except Exception:
                pass
        return found

    def get_known_devices(self) -> Dict[str, DeviceConfig]:
        return self.known_devices.copy()


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
