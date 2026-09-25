from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signed_distance_field import VoxelGridMetadata

# Standard Marching Cubes Edge Table (12-bit mask indicating which edges are intersected)
EDGE_TABLE = np.array([
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xba6, 0xaaf, 0x9a5, 0x8ac, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8ac, 0x9a5, 0xaaf, 0xba6,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
], dtype=np.int32)

# Cube edge vertex pairs: (v1, v2) for edges 0..11
EDGE_VERTICES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

# Standard unit-cube corner offsets
CORNER_OFFSETS = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
], dtype=np.float64)


@dataclass
class TriangleMesh:
    vertices: np.ndarray  # (V, 3) float64
    faces: np.ndarray     # (F, 3) int32
    normals: np.ndarray   # (V, 3) float64


class MarchingCubesReconstructor:
    """Extracts 2D triangle manifold isosurfaces from 3D scalar TSDF grids."""

    def __init__(self, isovalue: float = 0.0) -> None:
        self.isovalue = isovalue

    @staticmethod
    def _interpolate_vertex(
        p1: np.ndarray,
        p2: np.ndarray,
        val1: float,
        val2: float,
        isovalue: float,
    ) -> np.ndarray:
        if abs(val2 - val1) < 1e-6:
            return (p1 + p2) * 0.5
        t = (isovalue - val1) / (val2 - val1)
        t = max(0.0, min(1.0, t))
        return p1 + t * (p2 - p1)

    def extract_mesh(self, tsdf_grid: np.ndarray, metadata: VoxelGridMetadata) -> TriangleMesh:
        """March through voxels and extract triangle mesh."""
        return self.reconstruct(tsdf_grid, metadata)

    def reconstruct(self, tsdf_grid: np.ndarray, metadata: VoxelGridMetadata) -> TriangleMesh:
        """March through voxels and extract triangle mesh.

        Args:
            tsdf_grid: (nx, ny, nz) 3D scalar values
            metadata: VoxelGridMetadata
        Returns:
            TriangleMesh: vertices, faces, and vertex normals
        """
        nx, ny, nz = metadata.dimensions
        h = metadata.voxel_size
        origin = metadata.origin

        vertices: list[np.ndarray] = []
        faces: list[tuple[int, int, int]] = []

        for x in range(nx - 1):
            for y in range(ny - 1):
                for z in range(nz - 1):
                    # 8 corner values
                    vals = np.array([
                        tsdf_grid[x, y, z],
                        tsdf_grid[x + 1, y, z],
                        tsdf_grid[x + 1, y + 1, z],
                        tsdf_grid[x, y + 1, z],
                        tsdf_grid[x, y, z + 1],
                        tsdf_grid[x + 1, y, z + 1],
                        tsdf_grid[x + 1, y + 1, z + 1],
                        tsdf_grid[x, y + 1, z + 1],
                    ], dtype=np.float64)

                    # Determine cube index
                    cube_index = 0
                    for i in range(8):
                        if vals[i] < self.isovalue:
                            cube_index |= (1 << i)

                    edge_mask = EDGE_TABLE[cube_index]
                    if edge_mask == 0:
                        continue

                    # Corner world positions
                    cell_origin = origin + np.array([x, y, z]) * h
                    corners = cell_origin + CORNER_OFFSETS * h

                    # Interpolate intersected edge vertices
                    edge_pts: dict[int, int] = {}
                    for e in range(12):
                        if edge_mask & (1 << e):
                            v1_idx, v2_idx = EDGE_VERTICES[e]
                            interp_pt = self._interpolate_vertex(
                                corners[v1_idx],
                                corners[v2_idx],
                                vals[v1_idx],
                                vals[v2_idx],
                                self.isovalue,
                            )
                            idx = len(vertices)
                            vertices.append(interp_pt)
                            edge_pts[e] = idx

                    # Connect edges to create planar triangles
                    e_keys = list(edge_pts.values())
                    if len(e_keys) >= 3:
                        # Triangulate polygon fan
                        for tri_idx in range(1, len(e_keys) - 1):
                            faces.append((e_keys[0], e_keys[tri_idx], e_keys[tri_idx + 1]))

        if not vertices:
            return TriangleMesh(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int32),
                normals=np.zeros((0, 3), dtype=np.float64),
            )

        verts_arr = np.array(vertices, dtype=np.float64)
        faces_arr = np.array(faces, dtype=np.int32)

        # Compute vertex normals
        normals_arr = self._compute_vertex_normals(verts_arr, faces_arr)

        return TriangleMesh(vertices=verts_arr, faces=faces_arr, normals=normals_arr)

    @staticmethod
    def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        normals = np.zeros_like(vertices)
        if len(faces) == 0:
            return normals

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        lens = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = np.divide(face_normals, np.maximum(1e-8, lens))

        for f_idx, (i0, i1, i2) in enumerate(faces):
            fn = face_normals[f_idx]
            normals[i0] += fn
            normals[i1] += fn
            normals[i2] += fn

        n_lens = np.linalg.norm(normals, axis=1, keepdims=True)
        return np.divide(normals, np.maximum(1e-8, n_lens))
