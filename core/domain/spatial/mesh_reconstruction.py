from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict
import copy


class MeshType(Enum):
    TRIANGLE = "triangle"
    QUAD = "quad"
    POLYGON = "polygon"
    TETRAHEDRON = "tetrahedron"
    HEXAHEDRON = "hexahedron"
    POINT_CLOUD = "point_cloud"


class ReconstructionMethod(Enum):
    POISSON = "poisson"
    MARCHING_CUBES = "marching_cubes"
    BALL_PIVOTING = "ball_pivoting"
    DELAUNAY = "delaunay"
    ALPHA_SHAPES = "alpha_shapes"
    RIEMANNIAN = "riemannian"
    RBF = "rbf"
    MOVING_LEAST_SQUARES = "moving_least_squares"
    NEURAL_IMPLICIT = "neural_implicit"


@dataclass(frozen=True, slots=True)
class Vertex:
    position: np.ndarray
    normal: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None
    curvature: float = 0.0
    id: int = -1


@dataclass(frozen=True, slots=True)
class Face:
    vertex_indices: Tuple[int, int, int]
    normal: Optional[np.ndarray] = None
    material_id: int = 0
    area: float = 0.0


@dataclass(frozen=True, slots=True)
class Mesh:
    vertices: List[Vertex]
    faces: List[Face]
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.bbox_min, self.bbox_max)


@dataclass(frozen=True, slots=True)
class PointCloud:
    points: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    intensities: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None


class MeshProcessor:
    @staticmethod
    def compute_normals(mesh: Mesh) -> Mesh:
        vertices = [v.position for v in mesh.vertices]
        faces = [(f.vertex_indices[0], f.vertex_indices[1], f.vertex_indices[2]) for f in mesh.faces]

        face_normals = np.zeros((len(faces), 3))
        vertex_normals = np.zeros((len(vertices), 3))
        face_areas = np.zeros(len(faces))

        for i, (a, b, c) in enumerate(faces):
            v0 = vertices[a]
            v1 = vertices[b]
            v2 = vertices[c]

            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)

            if area > 1e-10:
                normal = normal / area
                face_normals[i] = normal
                face_areas[i] = area

                vertex_normals[a] += normal * area
                vertex_normals[b] += normal * area
                vertex_normals[c] += normal * area

        for i in range(len(vertex_normals)):
            norm = np.linalg.norm(vertex_normals[i])
            if norm > 1e-10:
                vertex_normals[i] = vertex_normals[i] / norm

        new_vertices = []
        for i, v in enumerate(mesh.vertices):
            new_vertices.append(Vertex(
                position=v.position,
                normal=vertex_normals[i],
                color=v.color,
                uv=v.uv,
                curvature=v.curvature,
                id=v.id
            ))

        new_faces = []
        for i, f in enumerate(mesh.faces):
            new_faces.append(Face(
                vertex_indices=f.vertex_indices,
                normal=face_normals[i],
                material_id=f.material_id,
                area=face_areas[i]
            ))

        return Mesh(
            vertices=new_vertices,
            faces=new_faces,
            bbox_min=mesh.bbox_min,
            bbox_max=mesh.bbox_max,
            metadata=mesh.metadata
        )

    @staticmethod
    def smooth_laplacian(mesh: Mesh, iterations: int = 10, lambda_factor: float = 0.5) -> Mesh:
        vertices = np.array([v.position for v in mesh.vertices])
        adjacency = MeshProcessor._build_adjacency(mesh)

        for _ in range(iterations):
            new_vertices = vertices.copy()
            for i, v in enumerate(vertices):
                neighbors = adjacency[i]
                if len(neighbors) > 0:
                    avg = np.mean(vertices[neighbors], axis=0)
                    new_vertices[i] = v + lambda_factor * (avg - v)
            vertices = new_vertices

        new_vertices_list = []
        for i, v in enumerate(mesh.vertices):
            new_vertices_list.append(Vertex(
                position=vertices[i],
                normal=v.normal,
                color=v.color,
                uv=v.uv,
                curvature=v.curvature,
                id=v.id
            ))

        return Mesh(
            vertices=new_vertices_list,
            faces=mesh.faces,
            bbox_min=mesh.bbox_min,
            bbox_max=mesh.bbox_max,
            metadata=mesh.metadata
        )

    @staticmethod
    def _build_adjacency(mesh: Mesh) -> List[List[int]]:
        adjacency = [[] for _ in range(len(mesh.vertices))]
        for face in mesh.faces:
            a, b, c = face.vertex_indices
            adjacency[a].extend([b, c])
            adjacency[b].extend([a, c])
            adjacency[c].extend([a, b])

        for i in range(len(adjacency)):
            adjacency[i] = list(set(adjacency[i]))

        return adjacency

    @staticmethod
    def decimate(mesh: Mesh, target_ratio: float = 0.5) -> Mesh:
        return mesh

    @staticmethod
    def subdivide(mesh: Mesh, method: str = "loop") -> Mesh:
        if method == "loop":
            return MeshProcessor._loop_subdivision(mesh)
        elif method == "catmull_clark":
            return MeshProcessor._catmull_clark_subdivision(mesh)
        return mesh

    @staticmethod
    def _loop_subdivision(mesh: Mesh) -> Mesh:
        vertices = [v.position for v in mesh.vertices]
        faces = [(f.vertex_indices[0], f.vertex_indices[1], f.vertex_indices[2]) for f in mesh.faces]

        edge_vertices = {}
        edge_map = {}

        for a, b, c in faces:
            edges = [(a, b), (b, c), (c, a)]
            for e in edges:
                e_sorted = tuple(sorted(e))
                if e_sorted not in edge_vertices:
                    edge_vertices[e_sorted] = len(vertices)
                    vertices.append((vertices[e[0]] + vertices[e[1]]) / 2)

        new_faces = []
        for a, b, c in faces:
            ab = edge_vertices[tuple(sorted((a, b)))]
            bc = edge_vertices[tuple(sorted((b, c)))]
            ca = edge_vertices[tuple(sorted((c, a)))]

            new_faces.extend([
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca)
            ])

        return Mesh(
            vertices=[Vertex(position=v) for v in vertices],
            faces=[Face(vertex_indices=f) for f in new_faces]
        )

    @staticmethod
    def _catmull_clark_subdivision(mesh: Mesh) -> Mesh:
        return mesh


class MeshReconstructor:
    def __init__(self, method: ReconstructionMethod = ReconstructionMethod.POISSON):
        self.method = method

    def reconstruct(self, point_cloud: PointCloud) -> Mesh:
        if self.method == ReconstructionMethod.POISSON:
            return self._poisson_reconstruction(point_cloud)
        elif self.method == ReconstructionMethod.MARCHING_CUBES:
            return self._marching_cubes(point_cloud)
        elif self.method == ReconstructionMethod.BALL_PIVOTING:
            return self._ball_pivoting(point_cloud)
        elif self.method == ReconstructionMethod.DELAUNAY:
            return self._delaunay_reconstruction(point_cloud)
        elif self.method == ReconstructionMethod.ALPHA_SHAPES:
            return self._alpha_shapes(point_cloud)
        elif self.method == ReconstructionMethod.RBF:
            return self._rbf_reconstruction(point_cloud)
        elif self.method == ReconstructionMethod.MOVING_LEAST_SQUARES:
            return self._mls_reconstruction(point_cloud)
        elif self.method == ReconstructionMethod.NEURAL_IMPLICIT:
            return self._neural_implicit_reconstruction(point_cloud)
        else:
            return self._poisson_reconstruction(point_cloud)

    def _poisson_reconstruction(self, point_cloud: PointCloud) -> Mesh:
        if point_cloud.normals is None:
            point_cloud = self._estimate_normals(point_cloud)

        return Mesh(vertices=[], faces=[])

    def _marching_cubes(self, point_cloud: PointCloud) -> Mesh:
        return Mesh(vertices=[], faces=[])

    def _ball_pivoting(self, point_cloud: PointCloud) -> Mesh:
        return Mesh(vertices=[], faces=[])

    def _delaunay_reconstruction(self, point_cloud: PointCloud) -> Mesh:
        return Mesh(vertices=[], faces=[])

    def _alpha_shapes(self, point_cloud: PointCloud) -> Mesh:
        return Mesh(vertices=[], faces=[])

    def _rbf_reconstruction(self, point_cloud: PointCloud) -> Mesh:
        return Mesh(vertices=[], faces=[])

    def _mls_reconstruction(self, point_cloud: PointCloud) -> Mesh:
        return Mesh(vertices=[], faces=[])

    def _neural_implicit_reconstruction(self, point_cloud: PointCloud) -> Mesh:
        return Mesh(vertices=[], faces=[])

    def _estimate_normals(self, point_cloud: PointCloud) -> PointCloud:
        from scipy.spatial import KDTree

        points = point_cloud.points
        if point_cloud.normals is not None:
            return point_cloud

        tree = KDTree(points)
        normals = np.zeros_like(points)

        for i, point in enumerate(points):
            indices = tree.query_ball_point(point, 0.1)
            if len(indices) < 3:
                normals[i] = [0, 0, 1]
                continue

            neighbors = points[indices]
            centroid = np.mean(neighbors, axis=0)
            cov = (neighbors - centroid).T @ (neighbors - centroid)
            _, _, Vt = np.linalg.svd(cov)
            normals[i] = Vt[-1]

        return PointCloud(
            points=points,
            normals=normals,
            colors=point_cloud.colors,
            intensities=point_cloud.intensities,
            labels=point_cloud.labels,
            timestamps=point_cloud.timestamps
        )


class MeshExporter:
    @staticmethod
    def export_obj(mesh: Mesh, path: str):
        with open(path, 'w') as f:
            f.write("# OBJ file\n")

            for v in mesh.vertices:
                f.write(f"v {v.position[0]} {v.position[1]} {v.position[2]}\n")

            if mesh.vertices[0].normal is not None:
                for v in mesh.vertices:
                    f.write(f"vn {v.normal[0]} {v.normal[1]} {v.normal[2]}\n")

            if mesh.vertices[0].uv is not None:
                for v in mesh.vertices:
                    f.write(f"vt {v.uv[0]} {v.uv[1]}\n")

            for face in mesh.faces:
                indices = [i + 1 for i in face.vertex_indices]
                f.write(f"f {' '.join(map(str, indices))}\n")

    @staticmethod
    def export_ply(mesh: Mesh, path: str, binary: bool = False):
        with open(path, 'wb' if binary else 'w') as f:
            if binary:
                f.write(b"ply\n")
                f.write(b"format binary_little_endian 1.0\n")
            else:
                f.write("ply\n")
                f.write("format ascii 1.0\n")

            f.write(f"element vertex {len(mesh.vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")

            if mesh.vertices[0].normal is not None:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")

            if mesh.vertices[0].color is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            f.write(f"element face {len(mesh.faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            for v in mesh.vertices:
                if binary:
                    f.write(struct.pack('fff', *v.position))
                    if v.normal is not None:
                        f.write(struct.pack('fff', *v.normal))
                    if v.color is not None:
                        f.write(struct.pack('BBB', *v.color))
                else:
                    line = f"{v.position[0]} {v.position[1]} {v.position[2]}"
                    if v.normal is not None:
                        line += f" {v.normal[0]} {v.normal[1]} {v.normal[2]}"
                    if v.color is not None:
                        line += f" {int(v.color[0]*255)} {int(v.color[1]*255)} {int(v.color[2]*255)}"
                    f.write(line + "\n")

            for face in mesh.faces:
                if binary:
                    f.write(struct.pack('BIII', 3, *face.vertex_indices))
                else:
                    f.write(f"3 {face.vertex_indices[0]} {face.vertex_indices[1]} {face.vertex_indices[2]}\n")

    @staticmethod
    def export_stl(mesh: Mesh, path: str, binary: bool = True):
        with open(path, 'wb' if binary else 'w') as f:
            if binary:
                f.write(b"solid mesh\n")
                for face in mesh.faces:
                    v0 = mesh.vertices[face.vertex_indices[0]].position
                    v1 = mesh.vertices[face.vertex_indices[1]].position
                    v2 = mesh.vertices[face.vertex_indices[2]].position

                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    normal = np.cross(edge1, edge2)
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm

                    f.write(struct.pack('fff', *normal))
                    f.write(struct.pack('fff', *v0))
                    f.write(struct.pack('fff', *v1))
                    f.write(struct.pack('fff', *v2))
                    f.write(struct.pack('H', 0))
                f.write(b"endsolid mesh\n")
            else:
                f.write("solid mesh\n")
                for face in mesh.faces:
                    v0 = mesh.vertices[face.vertex_indices[0]].position
                    v1 = mesh.vertices[face.vertex_indices[1]].position
                    v2 = mesh.vertices[face.vertex_indices[2]].position

                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    normal = np.cross(edge1, edge2)
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm

                    f.write(f"facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                    f.write("  outer loop\n")
                    f.write(f"  vertex {v0[0]} {v0[1]} {v0[2]}\n")
                    f.write(f"  vertex {v1[0]} {v1[1]} {v1[2]}\n")
                    f.write(f"  vertex {v2[0]} {v2[1]} {v2[2]}\n")
                    f.write("  endloop\n")
                    f.write("endfacet\n")
                f.write("endsolid mesh\n")

    @staticmethod
    def export_gltf(mesh: Mesh, path: str):
        import json

        def pack_vec3(arr):
            return base64.b64encode(struct.pack('<fff', *arr)).decode()

        def pack_vec2(arr):
            return base64.b64encode(struct.pack('<ff', *arr)).decode()

        accessors = []
        buffer_views = []
        buffers = []

        vertex_data = b""
        for v in mesh.vertices:
            vertex_data += struct.pack('<fff', *v.position)
            if v.normal is not None:
                vertex_data += struct.pack('<fff', *v.normal)
            if v.color is not None:
                vertex_data += struct.pack('<BBB', *v.color)

        index_data = b""
        for face in mesh.faces:
            index_data += struct.pack('<III', *face.vertex_indices)

        accessors.append({
            "bufferView": 0,
            "componentType": 5126,
            "count": len(mesh.vertices),
            "type": "VEC3",
            "min": mesh.bbox_min.tolist(),
            "max": mesh.bbox_max.tolist()
        })

        accessors.append({
            "bufferView": 1,
            "componentType": 5125,
            "count": len(mesh.faces) * 3,
            "type": "SCALAR"
        })

        gltf = {
            "asset": {"version": "2.0"},
            "meshes": [{
                "primitives": [{
                    "attributes": {"POSITION": 0},
                    "indices": 1,
                    "material": 0
                }]
            }],
            "accessors": accessors,
            "bufferViews": [
                {"buffer": 0, "byteLength": len(vertex_data), "target": 34962},
                {"buffer": 0, "byteLength": len(index_data), "target": 34963}
            ],
            "buffers": [{"byteLength": len(vertex_data) + len(index_data)}]
        }

        with open(path, 'w') as f:
            json.dump(gltf, f)


class MeshImporter:
    @staticmethod
    def import_obj(path: str) -> Mesh:
        vertices = []
        faces = []

        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == 'v':
                    vertices.append(Vertex(position=np.array([float(parts[1]), float(parts[2]), float(parts[3])])))
                elif parts[0] == 'f':
                    indices = []
                    for p in parts[1:]:
                        idx = p.split('/')[0]
                        indices.append(int(idx) - 1)
                    if len(indices) == 3:
                        faces.append(Face(vertex_indices=tuple(indices)))

        return Mesh(vertices=vertices, faces=faces)

    @staticmethod
    def import_ply(path: str) -> Mesh:
        return Mesh(vertices=[], faces=[])

    @staticmethod
    def import_stl(path: str) -> Mesh:
        return Mesh(vertices=[], faces=[])


class MeshSimplifier:
    @staticmethod
    def simplify(mesh: Mesh, target_faces: int, method: str = "quadric") -> Mesh:
        if method == "quadric":
            return MeshSimplifier._quadric_decimation(mesh, target_faces)
        return mesh

    @staticmethod
    def _quadric_decimation(mesh: Mesh, target_faces: int) -> Mesh:
        return mesh


class MeshMetrics:
    @staticmethod
    def surface_area(mesh: Mesh) -> float:
        area = 0.0
        vertices = [v.position for v in mesh.vertices]

        for face in mesh.faces:
            a, b, c = face.vertex_indices
            v0 = vertices[a]
            v1 = vertices[b]
            v2 = vertices[c]

            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)
            area += area

        return area

    @staticmethod
    def volume(mesh: Mesh) -> float:
        volume = 0.0
        vertices = [v.position for v in mesh.vertices]

        for face in mesh.faces:
            a, b, c = face.vertex_indices
            v0 = vertices[a]
            v1 = vertices[b]
            v2 = vertices[c]

            volume = np.dot(v0, np.cross(v1, v2)) / 6.0
            volume += volume

        return abs(volume)

    @staticmethod
    def euler_characteristic(mesh: Mesh) -> int:
        V = len(mesh.vertices)
        E = len(mesh.faces) * 3 // 2
        F = len(mesh.faces)
        return V - E + F

    @staticmethod
    def is_manifold(mesh: Mesh) -> bool:
        edge_count = defaultdict(int)

        for face in mesh.faces:
            a, b, c = face.vertex_indices
            edges = [(a, b), (b, c), (c, a)]
            for e in edges:
                edge_count[tuple(sorted(e))] += 1

        for count in edge_count.values():
            if count > 2:
                return False
        return True

    @staticmethod
    def is_watertight(mesh: Mesh) -> bool:
        edge_count = defaultdict(int)

        for face in mesh.faces:
            a, b, c = face.vertex_indices
            edges = [(a, b), (b, c), (c, a)]
            for e in edges:
                edge_count[tuple(sorted(e))] += 1

        for count in edge_count.values():
            if count != 2:
                return False
        return True


class MeshVisualizer:
    @staticmethod
    def render_mesh(
        mesh: Mesh,
        image_size: Tuple[int, int] = (800, 600),
        camera_matrix: np.ndarray = None,
        light_dir: np.ndarray = None
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if camera_matrix is None:
            camera_matrix = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=np.float32)
        if light_dir is None:
            light_dir = np.array([0, 0, 1])

        vertices = np.array([v.position for v in mesh.vertices])

        for face in mesh.faces:
            a, b, c = face.vertex_indices
            v0 = vertices[face.vertex_indices[0]]
            v1 = vertices[face.vertex_indices[1]]
            v2 = vertices[face.vertex_indices[2]]

            points_3d = np.array([v0, v1, v2])
            points_homo = np.hstack([points_3d, np.ones((3, 1))])
            points_2d = (camera_matrix @ points_homo.T).T
            points_2d = points_2d[:, :2] / points_2d[:, 2:3]

            pts = points_2d.astype(np.int32)
            cv2.fillPoly(vis, [pts], (200, 200, 200))
            cv2.polylines(vis, [pts], True, (255, 255, 255), 1)

        return vis


import struct
import base64
from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
