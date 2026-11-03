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


class GraspType(Enum):
    PARALLEL_JAW = "parallel_jaw"
    MULTI_FINGER = "multi_finger"
    SUCTION = "suction"
    MAGNETIC = "magnetic"
    ADHESIVE = "adhesive"
    ENVELOPING = "enveloping"
    PINCH = "pinch"
    POWER = "power"
    PRECISION = "precision"


class GraspQualityMetric(Enum):
    FORCE_CLOSURE = "force_closure"
    FORM_CLOSURE = "form_closure"
    EPSILON_QUALITY = "epsilon_quality"
    VOLUME_QUALITY = "volume_quality"
    GRASP_WRENCH_SPACE = "grasp_wrench_space"
    CONTACT_NORMAL_ALIGNMENT = "contact_normal_alignment"


@dataclass(frozen=True, slots=True)
class ContactPoint:
    position: np.ndarray
    normal: np.ndarray
    friction_coefficient: float = 0.5
    contact_area: float = 1e-4
    force_limit: float = 10.0

    def wrench_basis(self) -> np.ndarray:
        r = self.position
        n = self.normal
        f = np.array([n[0], n[1], n[2], r[1]*n[2]-r[2]*n[1], r[2]*n[0]-r[0]*n[2], r[0]*n[1]-r[1]*n[0]])
        return f


@dataclass(frozen=True, slots=True)
class Grasp:
    id: int
    type: GraspType
    contacts: List[ContactPoint]
    pregrasp_pose: np.ndarray
    grasp_pose: np.ndarray
    quality: float = 0.0
    score: float = 0.0
    object_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_contacts(self) -> int:
        return len(self.contacts)

    @property
    def centroid(self) -> np.ndarray:
        if not self.contacts:
            return np.zeros(3)
        return np.mean([c.position for c in self.contacts], axis=0)

    @property
    def approach_vector(self) -> np.ndarray:
        if not self.contacts:
            return np.array([0, 0, -1])
        return -np.mean([c.normal for c in self.contacts], axis=0)


@dataclass(frozen=True, slots=True)
class GraspCandidate:
    grasp: Grasp
    reachability: float
    collision_free: bool
    ik_solutions: List[np.ndarray]
    approach_path: Optional[List[np.ndarray]] = None
    retreat_path: Optional[List[np.ndarray]] = None


@dataclass(frozen=True, slots=True)
class ObjectModel:
    id: int
    name: str
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    convex_hull_vertices: Optional[np.ndarray] = None
    convex_hull_faces: Optional[np.ndarray] = None
    center_of_mass: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia_tensor: np.ndarray = field(default_factory=lambda: np.eye(3))
    mass: float = 1.0
    friction_coefficient: float = 0.5
    bounding_box: Tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (np.zeros(3), np.ones(3)))


class GraspSampler(ABC):
    @abstractmethod
    def sample_grasps(
        self,
        object_model: ObjectModel,
        num_samples: int,
        gripper_params: Dict[str, Any]
    ) -> List[Grasp]:
        pass


class ParallelJawGraspSampler(GraspSampler):
    def __init__(
        self,
        finger_width: float = 0.08,
        finger_length: float = 0.06,
        max_opening: float = 0.1,
        approach_distance: float = 0.1
    ):
        self.finger_width = finger_width
        self.finger_length = finger_length
        self.max_opening = max_opening
        self.approach_distance = approach_distance

    def sample_grasps(
        self,
        object_model: ObjectModel,
        num_samples: int,
        gripper_params: Dict[str, Any]
    ) -> List[Grasp]:
        grasps = []
        vertices = object_model.mesh_vertices
        faces = object_model.mesh_faces

        if object_model.convex_hull_vertices is not None:
            vertices = object_model.convex_hull_vertices
            faces = object_model.convex_hull_faces

        for _ in range(num_samples):
            face_idx = np.random.randint(len(faces))
            face = faces[face_idx]

            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            u, v = np.random.rand(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            point = v0 + u * (v1 - v0) + v * (v2 - v0)

            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / (np.linalg.norm(normal) + 1e-6)

            width = np.random.uniform(0.01, self.max_opening)
            if width > self.max_opening:
                continue

            contact1 = ContactPoint(
                position=point + normal * width / 2,
                normal=-normal,
                friction_coefficient=object_model.friction_coefficient
            )
            contact2 = ContactPoint(
                position=point - normal * width / 2,
                normal=normal,
                friction_coefficient=object_model.friction_coefficient
            )

            approach_dir = -normal
            pregrasp_pose = np.eye(4)
            pregrasp_pose[:3, 3] = point + approach_dir * self.approach_distance
            pregrasp_pose[:3, :3] = self._rotation_matrix_from_approach(approach_dir)

            grasp_pose = np.eye(4)
            grasp_pose[:3, 3] = point
            grasp_pose[:3, :3] = pregrasp_pose[:3, :3]

            grasp = Grasp(
                id=len(grasps),
                type=GraspType.PARALLEL_JAW,
                contacts=[contact1, contact2],
                pregrasp_pose=pregrasp_pose,
                grasp_pose=grasp_pose,
                object_id=object_model.id
            )
            grasps.append(grasp)

        return grasps

    def _rotation_matrix_from_approach(self, approach: np.ndarray) -> np.ndarray:
        approach = approach / (np.linalg.norm(approach) + 1e-6)
        up = np.array([0, 0, 1]) if abs(approach[2]) < 0.9 else np.array([0, 1, 0])
        right = np.cross(up, approach)
        right = right / (np.linalg.norm(right) + 1e-6)
        up = np.cross(approach, right)
        return np.column_stack([right, up, -approach])


class MultiFingerGraspSampler(GraspSampler):
    def __init__(
        self,
        num_fingers: int = 3,
        finger_radius: float = 0.01,
        finger_length: float = 0.05,
        palm_radius: float = 0.03
    ):
        self.num_fingers = num_fingers
        self.finger_radius = finger_radius
        self.finger_length = finger_length
        self.palm_radius = palm_radius

    def sample_grasps(
        self,
        object_model: ObjectModel,
        num_samples: int,
        gripper_params: Dict[str, Any]
    ) -> List[Grasp]:
        grasps = []
        vertices = object_model.mesh_vertices

        for _ in range(num_samples):
            idx = np.random.randint(len(vertices))
            center = vertices[idx]

            contacts = []
            for i in range(self.num_fingers):
                angle = 2 * np.pi * i / self.num_fingers
                dir_vec = np.array([math.cos(angle), math.sin(angle), 0])
                contact_pos = center + dir_vec * self.palm_radius
                normal = -dir_vec

                contacts.append(ContactPoint(
                    position=contact_pos,
                    normal=normal,
                    friction_coefficient=object_model.friction_coefficient
                ))

            pregrasp_pose = np.eye(4)
            pregrasp_pose[:3, 3] = center + np.array([0, 0, self.approach_distance])
            grasp_pose = np.eye(4)
            grasp_pose[:3, 3] = center

            grasp = Grasp(
                id=len(grasps),
                type=GraspType.MULTI_FINGER,
                contacts=contacts,
                pregrasp_pose=pregrasp_pose,
                grasp_pose=grasp_pose,
                object_id=object_model.id
            )
            grasps.append(grasp)

        return grasps


class SuctionGraspSampler(GraspSampler):
    def __init__(
        self,
        suction_radius: float = 0.02,
        min_curvature: float = 0.1,
        approach_distance: float = 0.1
    ):
        self.suction_radius = suction_radius
        self.min_curvature = min_curvature
        self.approach_distance = approach_distance

    def sample_grasps(
        self,
        object_model: ObjectModel,
        num_samples: int,
        gripper_params: Dict[str, Any]
    ) -> List[Grasp]:
        grasps = []
        vertices = object_model.mesh_vertices
        faces = object_model.mesh_faces

        for _ in range(num_samples):
            face_idx = np.random.randint(len(faces))
            face = faces[face_idx]

            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            u, v = np.random.rand(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            point = v0 + u * (v1 - v0) + v * (v2 - v0)

            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / (np.linalg.norm(normal) + 1e-6)

            if normal[2] < self.min_curvature:
                continue

            contact = ContactPoint(
                position=point,
                normal=normal,
                friction_coefficient=1.0,
                contact_area=math.pi * self.suction_radius ** 2,
                force_limit=50.0
            )

            approach_dir = -normal
            pregrasp_pose = np.eye(4)
            pregrasp_pose[:3, 3] = point + approach_dir * self.approach_distance
            pregrasp_pose[:3, :3] = self._rotation_matrix_from_approach(approach_dir)

            grasp_pose = np.eye(4)
            grasp_pose[:3, 3] = point
            grasp_pose[:3, :3] = pregrasp_pose[:3, :3]

            grasp = Grasp(
                id=len(grasps),
                type=GraspType.SUCTION,
                contacts=[contact],
                pregrasp_pose=pregrasp_pose,
                grasp_pose=grasp_pose,
                object_id=object_model.id
            )
            grasps.append(grasp)

        return grasps

    def _rotation_matrix_from_approach(self, approach: np.ndarray) -> np.ndarray:
        approach = approach / (np.linalg.norm(approach) + 1e-6)
        up = np.array([0, 0, 1]) if abs(approach[2]) < 0.9 else np.array([0, 1, 0])
        right = np.cross(up, approach)
        right = right / (np.linalg.norm(right) + 1e-6)
        up = np.cross(approach, right)
        return np.column_stack([right, up, -approach])


class GraspQualityEvaluator:
    def __init__(self, metrics: List[GraspQualityMetric] = None):
        self.metrics = metrics or [GraspQualityMetric.FORCE_CLOSURE, GraspQualityMetric.EPSILON_QUALITY]

    def evaluate(self, grasp: Grasp) -> Dict[GraspQualityMetric, float]:
        results = {}
        for metric in self.metrics:
            if metric == GraspQualityMetric.FORCE_CLOSURE:
                results[metric] = self._force_closure(grasp)
            elif metric == GraspQualityMetric.FORM_CLOSURE:
                results[metric] = self._form_closure(grasp)
            elif metric == GraspQualityMetric.EPSILON_QUALITY:
                results[metric] = self._epsilon_quality(grasp)
            elif metric == GraspQualityMetric.VOLUME_QUALITY:
                results[metric] = self._volume_quality(grasp)
            elif metric == GraspQualityMetric.GRASP_WRENCH_SPACE:
                results[metric] = self._grasp_wrench_space(grasp)
            elif metric == GraspQualityMetric.CONTACT_NORMAL_ALIGNMENT:
                results[metric] = self._contact_normal_alignment(grasp)
        return results

    def _force_closure(self, grasp: Grasp) -> float:
        if len(grasp.contacts) < 2:
            return 0.0

        G = np.zeros((6, len(grasp.contacts)))
        for i, contact in enumerate(grasp.contacts):
            G[:, i] = contact.wrench_basis()

        try:
            rank = np.linalg.matrix_rank(G)
            if rank < 6:
                return 0.0
            S = G @ G.T
            eigvals = np.linalg.eigvals(S)
            min_eig = np.min(eigvals[eigvals > 1e-10])
            return float(min_eig) if min_eig > 0 else 0.0
        except:
            return 0.0

    def _form_closure(self, grasp: Grasp) -> float:
        if len(grasp.contacts) < 7:
            return 0.0

        G = np.zeros((6, len(grasp.contacts)))
        for i, contact in enumerate(grasp.contacts):
            G[:, i] = contact.wrench_basis()

        try:
            rank = np.linalg.matrix_rank(G)
            if rank < 6:
                return 0.0
            return 1.0
        except:
            return 0.0

    def _epsilon_quality(self, grasp: Grasp) -> float:
        if len(grasp.contacts) < 2:
            return 0.0

        G = np.zeros((6, len(grasp.contacts)))
        for i, contact in enumerate(grasp.contacts):
            G[:, i] = contact.wrench_basis()

        try:
            U, S, Vt = np.linalg.svd(G, full_matrices=False)
            min_singular = S[-1]
            return float(min_singular)
        except:
            return 0.0

    def _volume_quality(self, grasp: Grasp) -> float:
        if len(grasp.contacts) < 2:
            return 0.0

        G = np.zeros((6, len(grasp.contacts)))
        for i, contact in enumerate(grasp.contacts):
            G[:, i] = contact.wrench_basis()

        try:
            S = G @ G.T
            det = np.linalg.det(S)
            return float(det ** (1.0 / 6.0)) if det > 0 else 0.0
        except:
            return 0.0

    def _grasp_wrench_space(self, grasp: Grasp) -> float:
        if len(grasp.contacts) < 2:
            return 0.0

        G = np.zeros((6, len(grasp.contacts)))
        for i, contact in enumerate(grasp.contacts):
            G[:, i] = contact.wrench_basis()

        try:
            eigvals = np.linalg.eigvals(G @ G.T)
            positive_eig = eigvals[eigvals > 1e-10]
            if len(positive_eig) < 6:
                return 0.0
            return float(np.prod(positive_eig) ** (1.0 / 6.0))
        except:
            return 0.0

    def _contact_normal_alignment(self, grasp: Grasp) -> float:
        if len(grasp.contacts) < 2:
            return 0.0

        normals = np.array([c.normal for c in grasp.contacts])
        pairwise_dots = []
        for i in range(len(normals)):
            for j in range(i + 1, len(normals)):
                pairwise_dots.append(abs(np.dot(normals[i], normals[j])))

        return float(1.0 - np.mean(pairwise_dots)) if pairwise_dots else 0.0


class GraspPlanner:
    def __init__(
        self,
        samplers: Dict[GraspType, GraspSampler],
        quality_evaluator: GraspQualityEvaluator,
        collision_checker: Optional[Any] = None,
        ik_solver: Optional[Any] = None
    ):
        self.samplers = samplers
        self.quality_evaluator = quality_evaluator
        self.collision_checker = collision_checker
        self.ik_solver = ik_solver

    def plan_grasps(
        self,
        object_model: ObjectModel,
        gripper_type: GraspType,
        num_samples: int = 1000,
        num_candidates: int = 50,
        gripper_params: Dict[str, Any] = None
    ) -> List[GraspCandidate]:
        gripper_params = gripper_params or {}

        sampler = self.samplers.get(gripper_type)
        if sampler is None:
            return []

        grasps = sampler.sample_grasps(object_model, num_samples, gripper_params)

        scored_grasps = []
        for grasp in grasps:
            quality_scores = self.quality_evaluator.evaluate(grasp)
            grasp.quality = np.mean(list(quality_scores.values()))
            scored_grasps.append(grasp)

        scored_grasps.sort(key=lambda g: g.quality, reverse=True)
        top_grasps = scored_grasps[:num_candidates]

        candidates = []
        for grasp in top_grasps:
            collision_free = True
            if self.collision_checker:
                collision_free = not self.collision_checker.check_collision(grasp.grasp_pose, object_model)

            ik_solutions = []
            if self.ik_solver:
                ik_solutions = self.ik_solver.solve(grasp.grasp_pose)

            candidate = GraspCandidate(
                grasp=grasp,
                reachability=1.0 if ik_solutions else 0.0,
                collision_free=collision_free,
                ik_solutions=ik_solutions
            )
            candidates.append(candidate)

        candidates.sort(key=lambda c: (c.collision_free, c.reachability, c.grasp.quality), reverse=True)
        return candidates


class ForceClosureChecker:
    @staticmethod
    def check_force_closure(contacts: List[ContactPoint], num_facets: int = 16) -> Tuple[bool, float]:
        if len(contacts) < 2:
            return False, 0.0

        G = np.zeros((6, len(contacts)))
        for i, contact in enumerate(contacts):
            G[:, i] = contact.wrench_basis()

        try:
            rank = np.linalg.matrix_rank(G)
            if rank < 6:
                return False, 0.0

            from scipy.spatial import ConvexHull
            wrenches = G.T
            hull = ConvexHull(wrenches)
            origin_in_hull = np.all(hull.equations @ np.array([0,0,0,0,0,0,1]) <= 1e-6)

            if origin_in_hull:
                min_dist = np.min(np.abs(hull.equations[:, :-1] @ np.zeros(6) + hull.equations[:, -1]))
                return True, min_dist
            return False, 0.0
        except:
            return False, 0.0

    @staticmethod
    def compute_grasp_wrench_space(contacts: List[ContactPoint], num_directions: int = 100) -> np.ndarray:
        if len(contacts) < 2:
            return np.zeros((6, num_directions))

        G = np.zeros((6, len(contacts)))
        for i, contact in enumerate(contacts):
            G[:, i] = contact.wrench_basis()

        directions = np.random.randn(num_directions, 6)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        wrench_space = G @ directions
        return wrench_space


class GraspSynthesis:
    def __init__(
        self,
        object_model: ObjectModel,
        gripper_type: GraspType,
        num_samples: int = 5000
    ):
        self.object_model = object_model
        self.gripper_type = gripper_type
        self.num_samples = num_samples

        self.samplers = {
            GraspType.PARALLEL_JAW: ParallelJawGraspSampler(),
            GraspType.MULTI_FINGER: MultiFingerGraspSampler(),
            GraspType.SUCTION: SuctionGraspSampler(),
        }
        self.quality_evaluator = GraspQualityEvaluator()

    def synthesize(self) -> List[Grasp]:
        sampler = self.samplers.get(self.gripper_type)
        if sampler is None:
            return []

        grasps = sampler.sample_grasps(self.object_model, self.num_samples, {})

        for grasp in grasps:
            scores = self.quality_evaluator.evaluate(grasp)
            grasp.quality = np.mean(list(scores.values()))
            grasp.score = grasp.quality

        grasps.sort(key=lambda g: g.quality, reverse=True)
        return grasps


class GraspRefinement:
    @staticmethod
    def refine_grasp(
        grasp: Grasp,
        object_model: ObjectModel,
        max_iterations: int = 100,
        step_size: float = 0.001
    ) -> Grasp:
        best_grasp = grasp
        best_quality = grasp.quality

        for _ in range(max_iterations):
            for contact in best_grasp.contacts:
                perturbation = np.random.randn(3) * step_size
                contact.position += perturbation
                contact.normal = contact.normal / (np.linalg.norm(contact.normal) + 1e-6)

            scores = GraspQualityEvaluator().evaluate(best_grasp)
            quality = np.mean(list(scores.values()))

            if quality > best_quality:
                best_quality = quality
            else:
                break

        best_grasp.quality = best_quality
        return best_grasp

    @staticmethod
    def optimize_contact_positions(
        grasp: Grasp,
        object_model: ObjectModel,
        optimizer: str = "gradient"
    ) -> Grasp:
        return grasp


class GraspExecution:
    def __init__(
        self,
        robot_model: Any,
        gripper_model: Any,
        approach_speed: float = 0.1,
        grasp_speed: float = 0.05,
        lift_speed: float = 0.1
    ):
        self.robot_model = robot_model
        self.gripper_model = gripper_model
        self.approach_speed = approach_speed
        self.grasp_speed = grasp_speed
        self.lift_speed = lift_speed

    def execute_grasp(self, candidate: GraspCandidate) -> Tuple[bool, Dict[str, Any]]:
        if not candidate.collision_free or not candidate.ik_solutions:
            return False, {"error": "No valid IK solution or collision"}

        result = {
            "approach_success": False,
            "grasp_success": False,
            "lift_success": False,
            "final_pose": None,
            "forces": [],
            "contact_points": []
        }

        for ik_sol in candidate.ik_solutions:
            approach_success = self._move_to_pose(candidate.grasp.pregrasp_pose, self.approach_speed)
            result["approach_success"] = approach_success

            if approach_success:
                grasp_success = self._move_to_pose(candidate.grasp.grasp_pose, self.grasp_speed)
                result["grasp_success"] = grasp_success

                if grasp_success:
                    lift_pose = candidate.grasp.grasp_pose.copy()
                    lift_pose[:3, 3] += np.array([0, 0, 0.1])
                    lift_success = self._move_to_pose(lift_pose, self.lift_speed)
                    result["lift_success"] = lift_success
                    result["final_pose"] = lift_pose

                    if lift_success:
                        return True, result

        return False, result

    def _move_to_pose(self, pose: np.ndarray, speed: float) -> bool:
        return True


class GraspVisualizer:
    @staticmethod
    def draw_grasp(
        image: np.ndarray,
        grasp: Grasp,
        camera_matrix: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        vis = image.copy()

        for contact in grasp.contacts:
            pt_3d = contact.position
            pt_2d = camera_matrix @ pt_3d
            if pt_2d[2] > 0:
                u = int(pt_2d[0] / pt_2d[2])
                v = int(pt_2d[1] / pt_2d[2])
                cv2.circle(vis, (u, v), 5, color, -1)

                normal_3d = contact.position + contact.normal * 0.02
                normal_2d = camera_matrix @ normal_3d
                if normal_2d[2] > 0:
                    u2 = int(normal_2d[0] / normal_2d[2])
                    v2 = int(normal_2d[1] / normal_2d[2])
                    cv2.line(vis, (u, v), (u2, v2), color, 2)

        return vis

    @staticmethod
    def draw_grasp_3d(
        grasp: Grasp,
        size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        vis = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        for contact in grasp.contacts:
            x = int(contact.position[0] * 100 + size[0] // 2)
            y = int(-contact.position[1] * 100 + size[1] // 2)
            if 0 <= x < size[0] and 0 <= y < size[1]:
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)

                nx = int((contact.position[0] + contact.normal[0] * 0.02) * 100 + size[0] // 2)
                ny = int(-(contact.position[1] + contact.normal[1] * 0.02) * 100 + size[1] // 2)
                cv2.line(vis, (x, y), (nx, ny), (0, 255, 0), 1)

        centroid = grasp.centroid
        cx = int(centroid[0] * 100 + size[0] // 2)
        cy = int(-centroid[1] * 100 + size[1] // 2)
        cv2.drawMarker(vis, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)

        return vis


class GraspDataset:
    def __init__(self, path: str):
        self.path = path
        self.grasps: List[Grasp] = []
        self.objects: Dict[int, ObjectModel] = {}

    def add_grasp(self, grasp: Grasp):
        self.grasps.append(grasp)

    def add_object(self, obj: ObjectModel):
        self.objects[obj.id] = obj

    def save(self):
        np.savez_compressed(
            self.path,
            grasps=[{
                'id': g.id,
                'type': g.type.value,
                'contacts': [(c.position, c.normal, c.friction_coefficient) for c in g.contacts],
                'pregrasp_pose': g.pregrasp_pose,
                'grasp_pose': g.grasp_pose,
                'quality': g.quality,
                'score': g.score,
                'object_id': g.object_id
            } for g in self.grasps],
            objects={oid: {
                'name': obj.name,
                'vertices': obj.mesh_vertices,
                'faces': obj.mesh_faces,
                'com': obj.center_of_mass,
                'mass': obj.mass
            } for oid, obj in self.objects.items()}
        )

    def load(self):
        data = np.load(self.path, allow_pickle=True)
        self.grasps = []
        for g_data in data['grasps']:
            contacts = [ContactPoint(pos, norm, fric) for pos, norm, fric in g_data['contacts']]
            grasp = Grasp(
                id=g_data['id'],
                type=GraspType(g_data['type']),
                contacts=contacts,
                pregrasp_pose=g_data['pregrasp_pose'],
                grasp_pose=g_data['grasp_pose'],
                quality=g_data['quality'],
                score=g_data['score'],
                object_id=g_data['object_id']
            )
            self.grasps.append(grasp)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
