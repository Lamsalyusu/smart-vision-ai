from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class PhysicsBodyType(Enum):
    RIGID = "rigid"
    SOFT = "soft"
    FLUID = "fluid"
    CLOTH = "cloth"
    PARTICLE = "particle"
    DEFORMABLE = "deformable"


class JointType(Enum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"
    PLANAR = "planar"
    DISTANCE = "distance"
    GEAR = "gear"
    RACK_PINION = "rack_pinion"


class CollisionShapeType(Enum):
    SPHERE = "sphere"
    BOX = "box"
    CAPSULE = "capsule"
    CYLINDER = "cylinder"
    CONE = "cone"
    CONVEX_HULL = "convex_hull"
    TRIANGLE_MESH = "triangle_mesh"
    HEIGHT_FIELD = "height_field"


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
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
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
        R = self.rotation.to_rotation_matrix().T
        pos = -R @ self.position.to_array()
        return Transform(Vector3.from_array(pos), Quaternion.from_rotation_matrix(R))


@dataclass(frozen=True, slots=True)
class RigidBody:
    id: int
    body_type: PhysicsBodyType = PhysicsBodyType.RIGID
    mass: float = 1.0
    inertia_tensor: np.ndarray = field(default_factory=lambda: np.eye(3))
    transform: Transform = field(default_factory=Transform)
    linear_velocity: Vector3 = field(default_factory=Vector3)
    angular_velocity: Vector3 = field(default_factory=Vector3)
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    gravity_scale: float = 1.0
    is_kinematic: bool = False
    is_static: bool = False
    collision_shape: Optional['CollisionShape'] = None
    material: Optional['PhysicsMaterial'] = None
    force: Vector3 = field(default_factory=Vector3)
    torque: Vector3 = field(default_factory=Vector3)


@dataclass(frozen=True, slots=True)
class PhysicsMaterial:
    friction: float = 0.5
    restitution: float = 0.0
    density: float = 1.0
    rolling_friction: float = 0.0
    spinning_friction: float = 0.0


@dataclass(frozen=True, slots=True)
class CollisionShape:
    shape_type: CollisionShapeType
    dimensions: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    radius: float = 1.0
    height: float = 1.0
    vertices: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None
    local_transform: Transform = field(default_factory=Transform)


@dataclass(frozen=True, slots=True)
class Joint:
    id: int
    joint_type: JointType
    body_a: int
    body_b: int
    local_anchor_a: Vector3 = field(default_factory=Vector3)
    local_anchor_b: Vector3 = field(default_factory=Vector3)
    local_axis_a: Vector3 = field(default_factory=lambda: Vector3(1, 0, 0))
    local_axis_b: Vector3 = field(default_factory=lambda: Vector3(1, 0, 0))
    reference_angle: float = 0.0
    lower_limit: float = -math.inf
    upper_limit: float = math.inf
    max_force: float = math.inf
    max_torque: float = math.inf
    motor_speed: float = 0.0
    motor_enabled: bool = False
    stiffness: float = 0.0
    damping: float = 0.0


@dataclass(frozen=True, slots=True)
class Contact:
    body_a: int
    body_b: int
    point: Vector3
    normal: Vector3
    penetration: float
    friction: float
    restitution: float
    tangent_1: Vector3
    tangent_2: Vector3


class PhysicsEngine:
    def __init__(
        self,
        gravity: Vector3 = Vector3(0, -9.81, 0),
        substeps: int = 4,
        fixed_timestep: float = 1.0 / 60.0
    ):
        self.gravity = gravity
        self.substeps = substeps
        self.fixed_timestep = fixed_timestep
        self.bodies: Dict[int, RigidBody] = {}
        self.joints: Dict[int, Joint] = {}
        self.contacts: List[Contact] = []
        self.body_id_counter = 0
        self.joint_id_counter = 0
        self.broadphase: Optional['Broadphase'] = None
        self.narrowphase: Optional['Narrowphase'] = None
        self.solver: Optional['ConstraintSolver'] = None
        self.collision_dispatcher: Optional['CollisionDispatcher'] = None

    def add_body(self, body: RigidBody) -> int:
        if body.id in self.bodies:
            return body.id
        self.bodies[body.id] = body
        return body.id

    def remove_body(self, body_id: int):
        if body_id in self.bodies:
            del self.bodies[body_id]

    def add_joint(self, joint: Joint) -> int:
        if joint.id in self.joints:
            return joint.id
        self.joints[joint.id] = joint
        return joint.id

    def remove_joint(self, joint_id: int):
        if joint_id in self.joints:
            del self.joints[joint_id]

    def step(self, dt: float):
        sub_dt = dt / self.substeps
        for _ in range(self.substeps):
            self._step_simulation(sub_dt)

    def _step_simulation(self, dt: float):
        self._apply_gravity(dt)
        self._apply_forces(dt)
        self._integrate_velocities(dt)
        self._detect_collisions()
        self._solve_constraints(dt)
        self._integrate_positions(dt)
        self._clear_forces()

    def _apply_gravity(self, dt: float):
        for body in self.bodies.values():
            if not body.is_static and not body.is_kinematic and body.mass > 0:
                gravity_force = self.gravity * body.mass * body.gravity_scale
                body.force = body.force + gravity_force

    def _apply_forces(self, dt: float):
        for body in self.bodies.values():
            if not body.is_static:
                pass

    def _integrate_velocities(self, dt: float):
        for body in self.bodies.values():
            if body.is_static or body.is_kinematic or body.mass <= 0:
                continue

            inv_mass = 1.0 / body.mass
            inv_inertia = np.linalg.inv(body.inertia_tensor) if not body.is_static else np.zeros((3, 3))

            body.linear_velocity = body.linear_velocity + body.force.to_array() * inv_mass * dt
            body.angular_velocity = body.angular_velocity + (inv_inertia @ body.torque.to_array()) * dt

            body.linear_velocity = body.linear_velocity * (1.0 - body.linear_damping * dt)
            body.angular_velocity = body.angular_velocity * (1.0 - body.angular_damping * dt)

    def _detect_collisions(self):
        self.contacts.clear()
        body_list = list(self.bodies.values())

        for i in range(len(body_list)):
            for j in range(i + 1, len(body_list)):
                contacts = self._collide(body_list[i], body_list[j])
                self.contacts.extend(contacts)

    def _collide(self, body_a: RigidBody, body_b: RigidBody) -> List[Contact]:
        if body_a.collision_shape is None or body_b.collision_shape is None:
            return []

        shape_a = body_a.collision_shape
        shape_b = body_b.collision_shape

        if shape_a.shape_type == CollisionShapeType.SPHERE and shape_b.shape_type == CollisionShapeType.SPHERE:
            return self._collide_sphere_sphere(body_a, body_b)
        elif shape_a.shape_type == CollisionShapeType.BOX and shape_b.shape_type == CollisionShapeType.BOX:
            return self._collide_box_box(body_a, body_b)
        elif shape_a.shape_type == CollisionShapeType.SPHERE and shape_b.shape_type == CollisionShapeType.BOX:
            return self._collide_sphere_box(body_a, body_b)
        elif shape_a.shape_type == CollisionShapeType.BOX and shape_b.shape_type == CollisionShapeType.SPHERE:
            return self._collide_sphere_box(body_b, body_a)

        return []

    def _collide_sphere_sphere(self, body_a: RigidBody, body_b: RigidBody) -> List[Contact]:
        pos_a = body_a.transform.position
        pos_b = body_b.transform.position
        radius_a = body_a.collision_shape.radius
        radius_b = body_b.collision_shape.radius

        diff = pos_b - pos_a
        dist = diff.norm()
        min_dist = radius_a + radius_b

        if dist < min_dist and dist > 1e-6:
            normal = diff * (1.0 / dist)
            penetration = min_dist - dist
            point = pos_a + normal * radius_a

            return [Contact(
                body_a=body_a.id,
                body_b=body_b.id,
                point=point,
                normal=normal,
                penetration=penetration,
                friction=0.5,
                restitution=0.0,
                tangent_1=Vector3(),
                tangent_2=Vector3()
            )]
        return []

    def _collide_box_box(self, body_a: RigidBody, body_b: RigidBody) -> List[Contact]:
        return []

    def _collide_sphere_box(self, sphere_body: RigidBody, box_body: RigidBody) -> List[Contact]:
        return []

    def _solve_constraints(self, dt: float):
        for contact in self.contacts:
            body_a = self.bodies.get(contact.body_a)
            body_b = self.bodies.get(contact.body_b)

            if not body_a or not body_b:
                continue

            if body_a.is_static and body_b.is_static:
                continue

            inv_mass_a = 0.0 if body_a.is_static or body_a.mass <= 0 else 1.0 / body_a.mass
            inv_mass_b = 0.0 if body_b.is_static or body_b.mass <= 0 else 1.0 / body_b.mass
            inv_mass_sum = inv_mass_a + inv_mass_b

            if inv_mass_sum == 0:
                continue

            normal = contact.normal
            rel_vel = body_b.linear_velocity - body_a.linear_velocity
            vel_along_normal = rel_vel.dot(normal)

            if vel_along_normal > 0:
                continue

            e = min(body_a.material.restitution if body_a.material else 0.0,
                    body_b.material.restitution if body_b.material else 0.0)

            j = -(1.0 + e) * vel_along_normal
            j /= inv_mass_sum

            impulse = normal * j
            if not body_a.is_static:
                body_a.linear_velocity = body_a.linear_velocity - impulse * inv_mass_a
            if not body_b.is_static:
                body_b.linear_velocity = body_b.linear_velocity + impulse * inv_mass_b

            if contact.penetration > 0:
                correction = normal * (contact.penetration / inv_mass_sum * 0.5)
                if not body_a.is_static:
                    body_a.transform = Transform(body_a.transform.position - correction, body_a.transform.rotation)
                if not body_b.is_static:
                    body_b.transform = Transform(body_b.transform.position + correction, body_b.transform.rotation)

    def _integrate_positions(self, dt: float):
        for body in self.bodies.values():
            if body.is_static or body.is_kinematic:
                continue

            body.transform = Transform(
                body.transform.position + body.linear_velocity * dt,
                body.transform.rotation
            )

            if body.angular_velocity.norm() > 1e-6:
                omega = body.angular_velocity
                angle = omega.norm() * dt
                axis = omega * (1.0 / omega.norm())
                delta_q = Quaternion(math.cos(angle * 0.5),
                                   axis.x * math.sin(angle * 0.5),
                                   axis.y * math.sin(angle * 0.5),
                                   axis.z * math.sin(angle * 0.5))
                new_rotation = delta_q * body.transform.rotation
                body.transform = Transform(body.transform.position, new_rotation)

    def _clear_forces(self):
        for body in self.bodies.values():
            body.force = Vector3()
            body.torque = Vector3()


class Broadphase:
    def __init__(self, world_size: Vector3 = Vector3(100, 100, 100), cell_size: float = 10.0):
        self.world_size = world_size
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], List[int]] = {}

    def get_cell(self, position: Vector3) -> Tuple[int, int, int]:
        return (
            int(position.x / self.cell_size),
            int(position.y / self.cell_size),
            int(position.z / self.cell_size)
        )

    def add_body(self, body_id: int, position: Vector3):
        cell = self.get_cell(position)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(body_id)

    def remove_body(self, body_id: int, position: Vector3):
        cell = self.get_cell(position)
        if cell in self.grid and body_id in self.grid[cell]:
            self.grid[cell].remove(body_id)

    def get_potential_pairs(self) -> List[Tuple[int, int]]:
        pairs = []
        for cell_bodies in self.grid.values():
            for i in range(len(cell_bodies)):
                for j in range(i + 1, len(cell_bodies)):
                    pairs.append((cell_bodies[i], cell_bodies[j]))
        return pairs


class Narrowphase:
    def __init__(self):
        pass

    def collide(self, shape_a: CollisionShape, transform_a: Transform,
                shape_b: CollisionShape, transform_b: Transform) -> List[Contact]:
        return []


class ConstraintSolver:
    def __init__(self, iterations: int = 10, tolerance: float = 1e-4):
        self.iterations = iterations
        self.tolerance = tolerance

    def solve(self, contacts: List[Contact], bodies: Dict[int, RigidBody], dt: float):
        pass


class CollisionDispatcher:
    def __init__(self):
        self.callbacks: Dict[Tuple[CollisionShapeType, CollisionShapeType], Callable] = {}

    def register(self, type_a: CollisionShapeType, type_b: CollisionShapeType, callback: Callable):
        self.callbacks[(type_a, type_b)] = callback
        self.callbacks[(type_b, type_a)] = callback

    def dispatch(self, shape_a: CollisionShape, transform_a: Transform,
                 shape_b: CollisionShape, transform_b: Transform) -> List[Contact]:
        key = (shape_a.shape_type, shape_b.shape_type)
        if key in self.callbacks:
            return self.callbacks[key](shape_a, transform_a, shape_b, transform_b)
        return []


class PhysicsWorld:
    def __init__(self, engine: PhysicsEngine):
        self.engine = engine
        self.debug_draw = False
        self.contact_callbacks: List[Callable[[Contact], None]] = []

    def step(self, dt: float):
        self.engine.step(dt)

    def add_rigid_body(
        self,
        mass: float,
        transform: Transform,
        collision_shape: CollisionShape,
        material: Optional[PhysicsMaterial] = None,
        is_kinematic: bool = False,
        is_static: bool = False
    ) -> RigidBody:
        body_id = self.engine.body_id_counter
        self.engine.body_id_counter += 1

        inertia = np.eye(3)
        if collision_shape.shape_type == CollisionShapeType.SPHERE:
            r = collision_shape.radius
            inertia = np.eye(3) * (2.0/5.0 * mass * r * r)
        elif collision_shape.shape_type == CollisionShapeType.BOX:
            x, y, z = collision_shape.dimensions.x, collision_shape.dimensions.y, collision_shape.dimensions.z
            inertia = np.array([
                [mass*(y*y+z*z)/12, 0, 0],
                [0, mass*(x*x+z*z)/12, 0],
                [0, 0, mass*(x*x+y*y)/12]
            ])

        body = RigidBody(
            id=body_id,
            mass=mass,
            inertia_tensor=inertia,
            transform=transform,
            collision_shape=collision_shape,
            material=material,
            is_kinematic=is_kinematic,
            is_static=is_static
        )

        self.engine.add_body(body)
        return body

    def add_joint(
        self,
        joint_type: JointType,
        body_a: int,
        body_b: int,
        local_anchor_a: Vector3 = Vector3(),
        local_anchor_b: Vector3 = Vector3(),
        **kwargs
    ) -> Joint:
        joint_id = self.engine.joint_id_counter
        self.engine.joint_id_counter += 1

        joint = Joint(
            id=joint_id,
            joint_type=joint_type,
            body_a=body_a,
            body_b=body_b,
            local_anchor_a=local_anchor_a,
            local_anchor_b=local_anchor_b,
            **kwargs
        )

        self.engine.add_joint(joint)
        return joint

    def ray_cast(
        self,
        origin: Vector3,
        direction: Vector3,
        max_distance: float = math.inf
    ) -> Optional[Contact]:
        return None

    def add_contact_callback(self, callback: Callable[[Contact], None]):
        self.contact_callbacks.append(callback)


class SoftBodySimulation:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        mass: float = 1.0,
        stiffness: float = 1000.0,
        damping: float = 10.0
    ):
        self.vertices = vertices.copy()
        self.faces = faces
        self.num_vertices = len(vertices)
        self.mass = mass / self.num_vertices
        self.stiffness = stiffness
        self.damping = damping

        self.velocities = np.zeros_like(vertices)
        self.forces = np.zeros_like(vertices)

        self.edges = self._build_edges()
        self.rest_lengths = self._compute_rest_lengths()

    def _build_edges(self) -> List[Tuple[int, int]]:
        edges = set()
        for face in self.faces:
            for i in range(3):
                a, b = face[i], face[(i+1)%3]
                edges.add(tuple(sorted((a, b))))
        return list(edges)

    def _compute_rest_lengths(self) -> np.ndarray:
        lengths = []
        for a, b in self.edges:
            lengths.append(np.linalg.norm(self.vertices[a] - self.vertices[b]))
        return np.array(lengths)

    def step(self, dt: float, gravity: Vector3 = Vector3(0, -9.81, 0)):
        self.forces.fill(0)

        for i, (a, b) in enumerate(self.edges):
            dir_vec = self.vertices[b] - self.vertices[a]
            length = np.linalg.norm(dir_vec)
            if length > 1e-6:
                force_dir = dir_vec / length
                force_mag = self.stiffness * (length - self.rest_lengths[i])
                damping_mag = self.damping * np.dot(self.velocities[b] - self.velocities[a], force_dir)
                force = force_dir * (force_mag + damping_mag)

                self.forces[a] += force
                self.forces[b] -= force

        for i in range(self.num_vertices):
            self.forces[i] += np.array([gravity.x, gravity.y, gravity.z]) * self.mass

        for i in range(self.num_vertices):
            self.velocities[i] += self.forces[i] / self.mass * dt
            self.vertices[i] += self.velocities[i] * dt


class ClothSimulation:
    def __init__(
        self,
        width: int,
        height: int,
        spacing: float = 0.1,
        mass: float = 0.1,
        structural_stiffness: float = 1000.0,
        shear_stiffness: float = 500.0,
        bend_stiffness: float = 100.0,
        damping: float = 10.0
    ):
        self.width = width
        self.height = height
        self.spacing = spacing
        self.num_particles = width * height
        self.mass = mass
        self.structural_stiffness = structural_stiffness
        self.shear_stiffness = shear_stiffness
        self.bend_stiffness = bend_stiffness
        self.damping = damping

        self.positions = np.zeros((self.num_particles, 3))
        self.velocities = np.zeros((self.num_particles, 3))
        self.forces = np.zeros((self.num_particles, 3))
        self.pinned = np.zeros(self.num_particles, dtype=bool)

        self._initialize_positions()
        self.constraints = self._build_constraints()

    def _initialize_positions(self):
        idx = 0
        for y in range(self.height):
            for x in range(self.width):
                self.positions[idx] = [x * self.spacing, 0, -y * self.spacing]
                idx += 1

    def _build_constraints(self) -> List[Tuple[int, int, float, str]]:
        constraints = []

        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x

                if x + 1 < self.width:
                    idx2 = y * self.width + (x + 1)
                    constraints.append((idx, idx2, self.spacing, 'structural'))

                if y + 1 < self.height:
                    idx2 = (y + 1) * self.width + x
                    constraints.append((idx, idx2, self.spacing, 'structural'))

                if x + 1 < self.width and y + 1 < self.height:
                    idx2 = (y + 1) * self.width + (x + 1)
                    constraints.append((idx, idx2, self.spacing * math.sqrt(2), 'shear'))

                if x + 1 < self.width and y > 0:
                    idx2 = (y - 1) * self.width + (x + 1)
                    constraints.append((idx, idx2, self.spacing * math.sqrt(2), 'shear'))

        return constraints

    def step(self, dt: float, gravity: Vector3 = Vector3(0, -9.81, 0)):
        self.forces.fill(0)

        for i in range(self.num_particles):
            if not self.pinned[i]:
                self.forces[i] += np.array([gravity.x, gravity.y, gravity.z]) * self.mass

        for a, b, rest_len, constraint_type in self.constraints:
            dir_vec = self.positions[b] - self.positions[a]
            length = np.linalg.norm(dir_vec)

            if length > 1e-6:
                force_dir = dir_vec / length

                if constraint_type == 'structural':
                    stiffness = self.structural_stiffness
                elif constraint_type == 'shear':
                    stiffness = self.shear_stiffness
                else:
                    stiffness = self.bend_stiffness

                force_mag = stiffness * (length - rest_len)
                damping_mag = self.damping * np.dot(self.velocities[b] - self.velocities[a], force_dir)
                force = force_dir * (force_mag + damping_mag)

                if not self.pinned[a]:
                    self.forces[a] += force
                if not self.pinned[b]:
                    self.forces[b] -= force

        for i in range(self.num_particles):
            if not self.pinned[i]:
                self.velocities[i] += self.forces[i] / self.mass * dt
                self.positions[i] += self.velocities[i] * dt


class FluidSimulation:
    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (32, 32, 32),
        cell_size: float = 0.1,
        viscosity: float = 0.01,
        density: float = 1000.0
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.viscosity = viscosity
        self.density = density

        self.velocity = np.zeros((*grid_size, 3))
        self.pressure = np.zeros(grid_size)
        self.density_field = np.ones(grid_size) * density

    def step(self, dt: float):
        self._advect_velocity(dt)
        self._apply_viscosity(dt)
        self._project_velocity(dt)
        self._advect_density(dt)

    def _advect_velocity(self, dt: float):
        pass

    def _apply_viscosity(self, dt: float):
        pass

    def _project_velocity(self, dt: float):
        pass

    def _advect_density(self, dt: float):
        pass


class PhysicsDebugDraw:
    @staticmethod
    def draw_body(body: RigidBody, camera_matrix: np.ndarray, image: np.ndarray, color: Tuple[int, int, int] = (255, 255, 255)):
        pass

    @staticmethod
    def draw_contact(contact: Contact, camera_matrix: np.ndarray, image: np.ndarray):
        pass

    @staticmethod
    def draw_joint(joint: Joint, body_a: RigidBody, body_b: RigidBody, camera_matrix: np.ndarray, image: np.ndarray):
        pass

    @staticmethod
    def draw_soft_body(soft_body: SoftBodySimulation, camera_matrix: np.ndarray, image: np.ndarray):
        pass

    @staticmethod
    def draw_cloth(cloth: ClothSimulation, camera_matrix: np.ndarray, image: np.ndarray):
        pass


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
