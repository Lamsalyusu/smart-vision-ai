from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class IntegratorType(Enum):
    EULER = "euler"
    SYMPLECTIC_EULER = "symplectic_euler"
    VERLET = "verlet"
    VELOCITY_VERLET = "velocity_verlet"
    LEAPFROG = "leapfrog"
    RUNGE_KUTTA_4 = "runge_kutta_4"
    RUNGE_KUTTA_45 = "runge_kutta_45"
    DORMAND_PRINCE = "dormand_prince"


class ConstraintType(Enum):
    DISTANCE = "distance"
    POINT_TO_POINT = "point_to_point"
    HINGE = "hinge"
    SLIDER = "slider"
    CONICAL = "conical"
    CONTACT = "contact"
    FRICTION = "friction"
    MOTOR = "motor"


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_array(arr: np.ndarray) -> 'Vector3':
        return Vector3(arr[0], arr[1], arr[2])

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3') -> 'Vector3':
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> 'Vector3':
        n = self.norm()
        return Vector3(self.x/n, self.y/n, self.z/n) if n > 0 else Vector3()


@dataclass(frozen=True, slots=True)
class Quaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z])

    @staticmethod
    def from_array(arr: np.ndarray) -> 'Quaternion':
        return Quaternion(arr[0], arr[1], arr[2], arr[3])

    def to_rotation_matrix(self) -> np.ndarray:
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])

    @staticmethod
    def from_rotation_matrix(R: np.ndarray) -> 'Quaternion':
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2,1] - R[1,2]) * s
            y = (R[0,2] - R[2,0]) * s
            z = (R[1,0] - R[0,1]) * s
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
        return Quaternion(w, x, y, z)

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )

    def rotate_vector(self, v: Vector3) -> Vector3:
        qv = Quaternion(0, v.x, v.y, v.z)
        q_conj = Quaternion(self.w, -self.x, -self.y, -self.z)
        result = self * qv * q_conj
        return Vector3(result.x, result.y, result.z)

    def conjugate(self) -> 'Quaternion':
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> 'Quaternion':
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        return self.conjugate() * (1.0 / norm_sq)


@dataclass(frozen=True, slots=True)
class Transform:
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)

    def to_matrix(self) -> np.ndarray:
        R = self.rotation.to_rotation_matrix()
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = self.position.to_array()
        return M

    @staticmethod
    def from_matrix(M: np.ndarray) -> 'Transform':
        pos = Vector3.from_array(M[:3, 3])
        rot = Quaternion.from_rotation_matrix(M[:3, :3])
        return Transform(pos, rot)

    def transform_point(self, point: Vector3) -> Vector3:
        return self.position + self.rotation.rotate_vector(point)

    def transform_vector(self, vector: Vector3) -> Vector3:
        return self.rotation.rotate_vector(vector)

    def inverse(self) -> 'Transform':
        R_inv = self.rotation.to_rotation_matrix().T
        pos_inv = -R_inv @ self.position.to_array()
        return Transform(Vector3.from_array(pos_inv), Quaternion.from_rotation_matrix(R_inv))


@dataclass(frozen=True, slots=True)
class InertiaTensor:
    Ixx: float = 1.0
    Iyy: float = 1.0
    Izz: float = 1.0
    Ixy: float = 0.0
    Ixz: float = 0.0
    Iyz: float = 0.0

    def to_matrix(self) -> np.ndarray:
        return np.array([
            [self.Ixx, self.Ixy, self.Ixz],
            [self.Ixy, self.Iyy, self.Iyz],
            [self.Ixz, self.Iyz, self.Izz]
        ])

    @staticmethod
    def from_matrix(M: np.ndarray) -> 'InertiaTensor':
        return InertiaTensor(
            Ixx=M[0,0], Iyy=M[1,1], Izz=M[2,2],
            Ixy=M[0,1], Ixz=M[0,2], Iyz=M[1,2]
        )

    def inverse(self) -> 'InertiaTensor':
        M = self.to_matrix()
        M_inv = np.linalg.inv(M)
        return InertiaTensor.from_matrix(M_inv)

    def transform(self, R: np.ndarray) -> 'InertiaTensor':
        M = self.to_matrix()
        M_transformed = R @ M @ R.T
        return InertiaTensor.from_matrix(M_transformed)


@dataclass(frozen=True, slots=True)
class RigidBody:
    id: int
    mass: float = 1.0
    inertia: InertiaTensor = field(default_factory=InertiaTensor)
    transform: Transform = field(default_factory=Transform)
    linear_velocity: Vector3 = field(default_factory=Vector3)
    angular_velocity: Vector3 = field(default_factory=Vector3)
    linear_momentum: Vector3 = field(default_factory=Vector3)
    angular_momentum: Vector3 = field(default_factory=Vector3)
    force: Vector3 = field(default_factory=Vector3)
    torque: Vector3 = field(default_factory=Vector3)
    is_static: bool = False
    is_kinematic: bool = False
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    gravity_scale: float = 1.0
    material: Optional['PhysicsMaterial'] = None

    def apply_force(self, force: Vector3, point: Vector3 = None):
        self.force = self.force + force
        if point is not None:
            r = point - self.transform.position
            self.torque = self.torque + r.cross(force)

    def apply_torque(self, torque: Vector3):
        self.torque = self.torque + torque

    def apply_impulse(self, impulse: Vector3, point: Vector3 = None):
        self.linear_momentum = self.linear_momentum + impulse
        self.linear_velocity = self.linear_momentum * (1.0 / self.mass)

        if point is not None:
            r = point - self.transform.position
            angular_impulse = r.cross(impulse)
            self.angular_momentum = self.angular_momentum + angular_impulse
            self.angular_velocity = self.inertia.inverse().to_matrix() @ self.angular_momentum.to_array()

    def get_velocity_at_point(self, point: Vector3) -> Vector3:
        r = point - self.transform.position
        return self.linear_velocity + self.angular_velocity.cross(r)

    def kinetic_energy(self) -> float:
        linear_ke = 0.5 * self.mass * self.linear_velocity.dot(self.linear_velocity)
        angular_ke = 0.5 * self.angular_velocity.dot(self.angular_momentum.to_array())
        return linear_ke + angular_ke


@dataclass(frozen=True, slots=True)
class PhysicsMaterial:
    friction: float = 0.5
    restitution: float = 0.0
    density: float = 1.0
    rolling_friction: float = 0.0
    spinning_friction: float = 0.0


class Integrator(ABC):
    @abstractmethod
    def integrate(
        self,
        body: RigidBody,
        dt: float,
        gravity: Vector3
    ) -> Tuple[Transform, Vector3, Vector3]:
        pass


class EulerIntegrator(Integrator):
    def integrate(
        self,
        body: RigidBody,
        dt: float,
        gravity: Vector3
    ) -> Tuple[Transform, Vector3, Vector3]:
        if body.is_static or body.is_kinematic:
            return body.transform, body.linear_velocity, body.angular_velocity

        inv_mass = 1.0 / body.mass
        inv_inertia = body.inertia.inverse().to_matrix()

        acceleration = body.force.to_array() * inv_mass + gravity.to_array() * body.gravity_scale
        angular_acc = inv_inertia @ body.torque.to_array()

        new_linear_vel = body.linear_velocity.to_array() + acceleration * dt
        new_angular_vel = body.angular_velocity.to_array() + angular_acc * dt

        new_linear_vel = new_linear_vel * (1.0 - body.linear_damping * dt)
        new_angular_vel = new_angular_vel * (1.0 - body.angular_damping * dt)

        new_pos = body.transform.position.to_array() + new_linear_vel * dt

        omega = Vector3.from_array(new_angular_vel)
        angle = omega.norm() * dt
        if angle > 1e-6:
            axis = omega.normalized()
            delta_q = Quaternion(
                math.cos(angle * 0.5),
                axis.x * math.sin(angle * 0.5),
                axis.y * math.sin(angle * 0.5),
                axis.z * math.sin(angle * 0.5)
            )
            new_rot = delta_q * body.transform.rotation
        else:
            new_rot = body.transform.rotation

        return (
            Transform(Vector3.from_array(new_pos), new_rot),
            Vector3.from_array(new_linear_vel),
            Vector3.from_array(new_angular_vel)
        )


class SymplecticEulerIntegrator(Integrator):
    def integrate(
        self,
        body: RigidBody,
        dt: float,
        gravity: Vector3
    ) -> Tuple[Transform, Vector3, Vector3]:
        if body.is_static or body.is_kinematic:
            return body.transform, body.linear_velocity, body.angular_velocity

        inv_mass = 1.0 / body.mass
        inv_inertia = body.inertia.inverse().to_matrix()

        acceleration = body.force.to_array() * inv_mass + gravity.to_array() * body.gravity_scale
        angular_acc = inv_inertia @ body.torque.to_array()

        new_linear_vel = body.linear_velocity.to_array() + acceleration * dt
        new_angular_vel = body.angular_velocity.to_array() + angular_acc * dt

        new_linear_vel = new_linear_vel * (1.0 - body.linear_damping * dt)
        new_angular_vel = new_angular_vel * (1.0 - body.angular_damping * dt)

        new_pos = body.transform.position.to_array() + new_linear_vel * dt

        omega = Vector3.from_array(new_angular_vel)
        angle = omega.norm() * dt
        if angle > 1e-6:
            axis = omega.normalized()
            delta_q = Quaternion(
                math.cos(angle * 0.5),
                axis.x * math.sin(angle * 0.5),
                axis.y * math.sin(angle * 0.5),
                axis.z * math.sin(angle * 0.5)
            )
            new_rot = delta_q * body.transform.rotation
        else:
            new_rot = body.transform.rotation

        return (
            Transform(Vector3.from_array(new_pos), new_rot),
            Vector3.from_array(new_linear_vel),
            Vector3.from_array(new_angular_vel)
        )


class VelocityVerletIntegrator(Integrator):
    def integrate(
        self,
        body: RigidBody,
        dt: float,
        gravity: Vector3
    ) -> Tuple[Transform, Vector3, Vector3]:
        if body.is_static or body.is_kinematic:
            return body.transform, body.linear_velocity, body.angular_velocity

        inv_mass = 1.0 / body.mass
        inv_inertia = body.inertia.inverse().to_matrix()

        a1 = body.force.to_array() * inv_mass + gravity.to_array() * body.gravity_scale
        alpha1 = inv_inertia @ body.torque.to_array()

        v_half = body.linear_velocity.to_array() + a1 * (dt * 0.5)
        omega_half = body.angular_velocity.to_array() + alpha1 * (dt * 0.5)

        pos = body.transform.position.to_array() + v_half * dt

        omega_half_vec = Vector3.from_array(omega_half)
        angle_half = omega_half_vec.norm() * dt
        if angle_half > 1e-6:
            axis = omega_half_vec.normalized()
            delta_q = Quaternion(
                math.cos(angle_half * 0.5),
                axis.x * math.sin(angle_half * 0.5),
                axis.y * math.sin(angle_half * 0.5),
                axis.z * math.sin(angle_half * 0.5)
            )
            rot_half = delta_q * body.transform.rotation
        else:
            rot_half = body.transform.rotation

        new_force = body.force
        new_torque = body.torque

        a2 = new_force.to_array() * inv_mass + gravity.to_array() * body.gravity_scale
        alpha2 = inv_inertia @ new_torque.to_array()

        v_final = v_half + a2 * (dt * 0.5)
        omega_final = omega_half + alpha2 * (dt * 0.5)

        v_final = v_final * (1.0 - body.linear_damping * dt)
        omega_final = omega_final * (1.0 - body.angular_damping * dt)

        omega_final_vec = Vector3.from_array(omega_final)
        angle_final = omega_final_vec.norm() * dt
        if angle_final > 1e-6:
            axis = omega_final_vec.normalized()
            delta_q = Quaternion(
                math.cos(angle_final * 0.5),
                axis.x * math.sin(angle_final * 0.5),
                axis.y * math.sin(angle_final * 0.5),
                axis.z * math.sin(angle_final * 0.5)
            )
            rot_final = delta_q * rot_half
        else:
            rot_final = rot_half

        return (
            Transform(Vector3.from_array(pos), rot_final),
            Vector3.from_array(v_final),
            Vector3.from_array(omega_final)
        )


class RungeKutta4Integrator(Integrator):
    def integrate(
        self,
        body: RigidBody,
        dt: float,
        gravity: Vector3
    ) -> Tuple[Transform, Vector3, Vector3]:
        if body.is_static or body.is_kinematic:
            return body.transform, body.linear_velocity, body.angular_velocity

        def acceleration(state: np.ndarray) -> np.ndarray:
            pos = state[:3]
            vel = state[3:6]
            rot = Quaternion.from_array(state[6:10])
            omega = state[10:13]

            acc = body.force.to_array() / body.mass + gravity.to_array() * body.gravity_scale
            angular_acc = body.inertia.inverse().to_matrix() @ body.torque.to_array()

            omega_vec = Vector3.from_array(omega)
            rot_quat = rot
            if omega_vec.norm() > 1e-6:
                axis = omega_vec.normalized()
                delta_q = Quaternion(
                    math.cos(omega_vec.norm() * 0.5),
                    axis.x * math.sin(omega_vec.norm() * 0.5),
                    axis.y * math.sin(omega_vec.norm() * 0.5),
                    axis.z * math.sin(omega_vec.norm() * 0.5)
                )
                rot_deriv = delta_q * rot
            else:
                rot_deriv = Quaternion()

            return np.concatenate([vel, acc, np.array([rot_deriv.w, rot_deriv.x, rot_deriv.y, rot_deriv.z]), angular_acc])

        state = np.concatenate([
            body.transform.position.to_array(),
            body.linear_velocity.to_array(),
            body.transform.rotation.to_array(),
            body.angular_velocity.to_array()
        ])

        k1 = acceleration(state) * dt
        k2 = acceleration(state + k1 * 0.5) * dt
        k3 = acceleration(state + k2 * 0.5) * dt
        k4 = acceleration(state + k3) * dt

        new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        new_pos = Vector3.from_array(new_state[:3])
        new_vel = Vector3.from_array(new_state[3:6])
        new_rot = Quaternion.from_array(new_state[6:10])
        new_omega = Vector3.from_array(new_state[10:13])

        return (Transform(new_pos, new_rot), new_vel, new_omega)


class IntegratorFactory:
    @staticmethod
    def create(integrator_type: IntegratorType) -> Integrator:
        if integrator_type == IntegratorType.EULER:
            return EulerIntegrator()
        elif integrator_type == IntegratorType.SYMPLECTIC_EULER:
            return SymplecticEulerIntegrator()
        elif integrator_type == IntegratorType.VELOCITY_VERLET:
            return VelocityVerletIntegrator()
        elif integrator_type == IntegratorType.RUNGE_KUTTA_4:
            return RungeKutta4Integrator()
        else:
            return SymplecticEulerIntegrator()


class ConstraintSolver:
    def __init__(
        self,
        max_iterations: int = 10,
        tolerance: float = 1e-4,
        baumgarte: float = 0.2
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.baumgarte = baumgarte

    def solve(
        self,
        bodies: List[RigidBody],
        constraints: List['Constraint'],
        dt: float
    ):
        for _ in range(self.max_iterations):
            max_error = 0.0

            for constraint in constraints:
                error = constraint.solve(bodies, dt)
                max_error = max(max_error, abs(error))

            if max_error < self.tolerance:
                break


class Constraint(ABC):
    @abstractmethod
    def solve(self, bodies: List[RigidBody], dt: float) -> float:
        pass


class DistanceConstraint(Constraint):
    def __init__(
        self,
        body_a: RigidBody,
        body_b: RigidBody,
        anchor_a: Vector3,
        anchor_b: Vector3,
        distance: float
    ):
        self.body_a = body_a
        self.body_b = body_b
        self.anchor_a = anchor_a
        self.anchor_b = anchor_b
        self.distance = distance

    def solve(self, bodies: List[RigidBody], dt: float) -> float:
        pos_a = self.body_a.transform.position + self.body_a.transform.rotation.rotate_vector(self.anchor_a)
        pos_b = self.body_b.transform.position + self.body_b.transform.rotation.rotate_vector(self.anchor_b)

        diff = pos_b - pos_a
        dist = diff.norm()

        if dist < 1e-6:
            return 0.0

        correction = (dist - self.distance) / dist
        correction_vec = diff * correction

        inv_mass_a = 1.0 / self.body_a.mass if not self.body_a.is_static else 0.0
        inv_mass_b = 1.0 / self.body_b.mass if not self.body_b.is_static else 0.0
        inv_mass_sum = inv_mass_a + inv_mass_b

        if inv_mass_sum > 0:
            move_a = correction_vec * (inv_mass_a / inv_mass_sum) * 0.5
            move_b = correction_vec * (inv_mass_b / inv_mass_sum) * 0.5

            self.body_a.transform = Transform(
                self.body_a.transform.position - move_a,
                self.body_a.transform.rotation
            )
            self.body_b.transform = Transform(
                self.body_b.transform.position + move_b,
                self.body_b.transform.rotation
            )

        return abs(dist - self.distance)


class ContactConstraint(Constraint):
    def __init__(
        self,
        body_a: RigidBody,
        body_b: RigidBody,
        contact_point: Vector3,
        normal: Vector3,
        penetration: float,
        friction: float = 0.5,
        restitution: float = 0.0
    ):
        self.body_a = body_a
        self.body_b = body_b
        self.contact_point = contact_point
        self.normal = normal
        self.penetration = penetration
        self.friction = friction
        self.restitution = restitution

    def solve(self, bodies: List[RigidBody], dt: float) -> float:
        rel_vel = self.body_b.linear_velocity - self.body_a.linear_velocity
        vel_along_normal = rel_vel.dot(self.normal)

        if vel_along_normal > 0:
            return 0.0

        inv_mass_a = 1.0 / self.body_a.mass if not self.body_a.is_static else 0.0
        inv_mass_b = 1.0 / self.body_b.mass if not self.body_b.is_static else 0.0
        inv_mass_sum = inv_mass_a + inv_mass_b

        if inv_mass_sum == 0:
            return 0.0

        e = min(
            self.body_a.material.restitution if self.body_a.material else 0.0,
            self.body_b.material.restitution if self.body_b.material else 0.0
        )

        j = -(1.0 + e) * vel_along_normal
        j /= inv_mass_sum

        impulse = self.normal * j

        if not self.body_a.is_static:
            self.body_a.linear_velocity = self.body_a.linear_velocity - impulse * (1.0 / self.body_a.mass)
        if not self.body_b.is_static:
            self.body_b.linear_velocity = self.body_b.linear_velocity + impulse * (1.0 / self.body_b.mass)

        if self.penetration > 0:
            correction = self.normal * (self.penetration / inv_mass_sum * 0.5)
            if not self.body_a.is_static:
                self.body_a.transform = Transform(
                    self.body_a.transform.position - correction,
                    self.body_a.transform.rotation
                )
            if not self.body_b.is_static:
                self.body_b.transform = Transform(
                    self.body_b.transform.position + correction,
                    self.body_b.transform.rotation
                )

        return abs(self.penetration)


class FrictionConstraint(Constraint):
    def __init__(
        self,
        body_a: RigidBody,
        body_b: RigidBody,
        contact_point: Vector3,
        normal: Vector3,
        friction: float,
        tangent_1: Vector3,
        tangent_2: Vector3
    ):
        self.body_a = body_a
        self.body_b = body_b
        self.contact_point = contact_point
        self.normal = normal
        self.friction = friction
        self.tangent_1 = tangent_1
        self.tangent_2 = tangent_2

    def solve(self, bodies: List[RigidBody], dt: float) -> float:
        rel_vel = self.body_b.linear_velocity - self.body_a.linear_velocity

        vt1 = rel_vel.dot(self.tangent_1)
        vt2 = rel_vel.dot(self.tangent_2)

        inv_mass_a = 1.0 / self.body_a.mass if not self.body_a.is_static else 0.0
        inv_mass_b = 1.0 / self.body_b.mass if not self.body_b.is_static else 0.0
        inv_mass_sum = inv_mass_a + inv_mass_b

        if inv_mass_sum == 0:
            return 0.0

        jt1 = -vt1 / inv_mass_sum
        jt2 = -vt2 / inv_mass_sum

        jt_mag = math.sqrt(jt1**2 + jt2**2)
        max_friction = self.friction

        if jt_mag > max_friction:
            scale = max_friction / jt_mag
            jt1 *= scale
            jt2 *= scale

        friction_impulse = self.tangent_1 * jt1 + self.tangent_2 * jt2

        if not self.body_a.is_static:
            self.body_a.linear_velocity = self.body_a.linear_velocity - friction_impulse * (1.0 / self.body_a.mass)
        if not self.body_b.is_static:
            self.body_b.linear_velocity = self.body_b.linear_velocity + friction_impulse * (1.0 / self.body_b.mass)

        return abs(jt1) + abs(jt2)


class PhysicsSystem:
    def __init__(
        self,
        integrator: IntegratorType = IntegratorType.SYMPLECTIC_EULER,
        gravity: Vector3 = Vector3(0, -9.81, 0),
        dt: float = 1.0 / 60.0
    ):
        self.integrator = IntegratorFactory.create(integrator)
        self.gravity = gravity
        self.dt = dt

        self.bodies: Dict[int, RigidBody] = {}
        self.constraints: List[Constraint] = []
        self.body_id_counter = 0
        self.constraint_solver = ConstraintSolver()

    def add_body(self, body: RigidBody) -> int:
        if body.id in self.bodies:
            return body.id

        if body.id == 0:
            body.id = self.body_id_counter
            self.body_id_counter += 1

        self.bodies[body.id] = body
        return body.id

    def remove_body(self, body_id: int):
        if body_id in self.bodies:
            del self.bodies[body_id]

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)

    def step(self, dt: float = None):
        dt = dt or self.dt

        for body in self.bodies.values():
            new_transform, new_linear_vel, new_angular_vel = self.integrator.integrate(
                body, dt, self.gravity
            )
            body.transform = new_transform
            body.linear_velocity = new_linear_vel
            body.angular_velocity = new_angular_vel

        self.constraint_solver.solve(list(self.bodies.values()), self.constraints, dt)

    def add_contact(
        self,
        body_a: int,
        body_b: int,
        contact_point: Vector3,
        normal: Vector3,
        penetration: float,
        friction: float = 0.5,
        restitution: float = 0.0
    ):
        body_a_obj = self.bodies.get(body_a)
        body_b_obj = self.bodies.get(body_b)

        if body_a_obj and body_b_obj:
            tangent_1 = Vector3(-normal.y, normal.x, 0).normalized()
            tangent_2 = normal.cross(tangent_1).normalized()

            self.constraints.append(ContactConstraint(
                body_a_obj, body_b_obj, contact_point, normal, penetration,
                friction, restitution
            ))
            self.constraints.append(FrictionConstraint(
                body_a_obj, body_b_obj, contact_point, normal, friction,
                tangent_1, tangent_2
            ))


class RigidBodyDynamics:
    def __init__(
        self,
        integrator: IntegratorType = IntegratorType.SYMPLECTIC_EULER,
        gravity: Vector3 = Vector3(0, -9.81, 0),
        dt: float = 1.0 / 60.0
    ):
        self.system = PhysicsSystem(integrator, gravity, dt)

    def add_rigid_body(
        self,
        mass: float,
        position: Vector3 = Vector3(),
        orientation: Quaternion = Quaternion(),
        linear_velocity: Vector3 = Vector3(),
        angular_velocity: Vector3 = Vector3(),
        inertia: InertiaTensor = None,
        material: PhysicsMaterial = None,
        is_static: bool = False,
        is_kinematic: bool = False
    ) -> RigidBody:
        if inertia is None:
            inertia = InertiaTensor()

        body = RigidBody(
            id=0,
            mass=mass,
            inertia=inertia,
            transform=Transform(position, orientation),
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            material=material,
            is_static=is_static,
            is_kinematic=is_kinematic
        )

        self.system.add_body(body)
        return body

    def step(self, dt: float = None):
        self.system.step(dt)

    def get_body_state(self, body_id: int) -> Optional[RigidBody]:
        return self.system.bodies.get(body_id)


class DynamicsMetrics:
    @staticmethod
    def total_energy(bodies: List[RigidBody]) -> float:
        return sum(body.kinetic_energy() for body in bodies)

    @staticmethod
    def total_momentum(bodies: List[RigidBody]) -> Vector3:
        total = Vector3()
        for body in bodies:
            total = total + body.linear_momentum
        return total

    @staticmethod
    def total_angular_momentum(bodies: List[RigidBody]) -> Vector3:
        total = Vector3()
        for body in bodies:
            total = total + body.angular_momentum
        return total

    @staticmethod
    def energy_error(initial: float, current: float) -> float:
        return abs(current - initial) / (initial + 1e-10)

    @staticmethod
    def momentum_error(initial: Vector3, current: Vector3) -> float:
        return (initial - current).norm() / (initial.norm() + 1e-10)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
