from __future__ import annotations

from collections import defaultdict

import numpy as np

from .marching_cubes import TriangleMesh


class MeshPostProcessor:
    """Performs Laplacian smoothing, vertex clustering decimation, and metric calculation."""

    @staticmethod
    def laplacian_smooth(mesh: TriangleMesh, iterations: int = 3, lambda_factor: float = 0.5) -> TriangleMesh:
        """Apply umbrella Laplacian mesh smoothing to eliminate voxel staircasing artifacts."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return mesh

        # Build vertex adjacency map
        adjacency: dict[int, set[int]] = defaultdict(set)
        for i0, i1, i2 in mesh.faces:
            adjacency[i0].add(i1)
            adjacency[i0].add(i2)
            adjacency[i1].add(i0)
            adjacency[i1].add(i2)
            adjacency[i2].add(i0)
            adjacency[i2].add(i1)

        smoothed_verts = np.copy(mesh.vertices)

        for _ in range(iterations):
            new_verts = np.copy(smoothed_verts)
            for i, neighbors in adjacency.items():
                if not neighbors:
                    continue
                neighbor_mean = np.mean(smoothed_verts[list(neighbors)], axis=0)
                new_verts[i] = smoothed_verts[i] + lambda_factor * (neighbor_mean - smoothed_verts[i])
            smoothed_verts = new_verts

        return TriangleMesh(
            vertices=smoothed_verts,
            faces=mesh.faces,
            normals=mesh.normals,
        )

    @staticmethod
    def decimate_grid_clustering(mesh: TriangleMesh, cluster_voxel_size: float = 0.05) -> TriangleMesh:
        """Decimate triangle mesh using uniform spatial vertex clustering."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return mesh

        # Map vertices to grid cell keys
        cell_coords = np.floor(mesh.vertices / cluster_voxel_size).astype(np.int64)
        cell_to_id: dict[tuple[int, int, int], int] = {}
        cluster_verts: list[list[np.ndarray]] = []
        vert_remap: dict[int, int] = {}

        for idx, coord in enumerate(cell_coords):
            key = (int(coord[0]), int(coord[1]), int(coord[2]))
            if key not in cell_to_id:
                new_id = len(cluster_verts)
                cell_to_id[key] = new_id
                cluster_verts.append([])
            cluster_id = cell_to_id[key]
            cluster_verts[cluster_id].append(mesh.vertices[idx])
            vert_remap[idx] = cluster_id

        # New representative vertices: centroid of each cluster
        new_vertices = np.array([np.mean(pts, axis=0) for pts in cluster_verts], dtype=np.float64)

        # Re-index faces and eliminate degenerates
        new_faces: list[tuple[int, int, int]] = []
        for i0, i1, i2 in mesh.faces:
            n0 = vert_remap[i0]
            n1 = vert_remap[i1]
            n2 = vert_remap[i2]
            # Eliminate degenerate triangles where two or more vertices collapse
            if n0 != n1 and n1 != n2 and n0 != n2:
                new_faces.append((n0, n1, n2))

        faces_arr = np.array(new_faces, dtype=np.int32) if new_faces else np.zeros((0, 3), dtype=np.int32)

        return TriangleMesh(
            vertices=new_vertices,
            faces=faces_arr,
            normals=mesh.normals,
        )

    @staticmethod
    def compute_surface_area(mesh: TriangleMesh) -> float:
        """Compute the total surface area in square meters."""
        if len(mesh.faces) == 0:
            return 0.0
        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]
        cross_prod = np.cross(v1 - v0, v2 - v0)
        return float(0.5 * np.sum(np.linalg.norm(cross_prod, axis=1)))
