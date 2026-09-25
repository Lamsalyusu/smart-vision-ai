from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lie_algebra import SE3


@dataclass
class PoseGraphEdge:
    """Relative pose constraint between vertex i and vertex j with 6x6 information matrix."""
    from_id: int
    to_id: int
    t_measured: SE3
    information: np.ndarray  # 6x6 inverse covariance matrix


class PoseGraphOptimizer:
    """Nonlinear Least Squares 6-DoF Pose Graph SLAM Optimizer using Levenberg-Marquardt."""

    def __init__(self, damping: float = 1e-3, max_iters: int = 25) -> None:
        self.damping = damping
        self.max_iters = max_iters
        self.vertices: dict[int, SE3] = {}
        self.edges: list[PoseGraphEdge] = []

    def add_vertex(self, vertex_id: int, pose: SE3) -> None:
        self.vertices[vertex_id] = pose

    def add_edge(self, from_id: int, to_id: int, relative_pose: SE3, information: np.ndarray) -> None:
        self.edges.append(PoseGraphEdge(from_id, to_id, relative_pose, information))

    def compute_error(self, edge: PoseGraphEdge) -> np.ndarray:
        """Error residual: e_ij = log(T_ij_meas^-1 * (T_i^-1 * T_j)) in R^6."""
        t_i = self.vertices[edge.from_id]
        t_j = self.vertices[edge.to_id]
        relative = t_i.inverse().compose(t_j)
        err_se3 = edge.t_measured.inverse().compose(relative)
        return err_se3.log()

    def total_cost(self) -> float:
        """Compute sum of squared Mahalanobis residuals: E = sum(e^T * Omega * e)."""
        cost = 0.0
        for edge in self.edges:
            err = self.compute_error(edge)
            cost += float(err.T @ edge.information @ err)
        return cost

    def optimize(self) -> float:
        """Run Levenberg-Marquardt pose graph optimization."""
        vertex_ids = sorted(self.vertices.keys())
        n_vertices = len(vertex_ids)
        if n_vertices < 2:
            return 0.0

        id_to_idx = {vid: idx for idx, vid in enumerate(vertex_ids)}
        # Fix vertex 0 as gauge freedom anchor
        dim = 6 * (n_vertices - 1)

        current_cost = self.total_cost()
        current_damping = self.damping

        for iteration in range(self.max_iters):
            h_mat = np.zeros((dim, dim), dtype=np.float64)
            b_vec = np.zeros(dim, dtype=np.float64)

            for edge in self.edges:
                idx_i = id_to_idx[edge.from_id]
                idx_j = id_to_idx[edge.to_id]

                err = self.compute_error(edge)
                omega = edge.information

                # Analytical / numerical Jacobians Ji and Jj
                t_j = self.vertices[edge.to_id]
                adj_inv = t_j.inverse().adjoint()
                j_i = -adj_inv
                j_j = np.eye(6)

                # Assemble block entries into H and b
                for idx_src, jac_src in [(idx_i, j_i), (idx_j, j_j)]:
                    if idx_src == 0:
                        continue  # Anchor fixed
                    blk_src = (idx_src - 1) * 6
                    b_vec[blk_src:blk_src + 6] -= jac_src.T @ omega @ err

                    for idx_dst, jac_dst in [(idx_i, j_i), (idx_j, j_j)]:
                        if idx_dst == 0:
                            continue
                        blk_dst = (idx_dst - 1) * 6
                        h_mat[blk_src:blk_src + 6, blk_dst:blk_dst + 6] += jac_src.T @ omega @ jac_dst

            # Add Levenberg-Marquardt diagonal damping: H + lambda * diag(H)
            diag_h = np.diag(np.diagonal(h_mat))
            h_damped = h_mat + current_damping * (diag_h + 1e-4 * np.eye(dim))

            try:
                delta_xi = np.linalg.solve(h_damped, b_vec)
            except np.linalg.LinAlgError:
                current_damping *= 4.0
                continue

            # Trial update
            backup_vertices = {vid: SE3(p.r.copy(), p.t.copy()) for vid, p in self.vertices.items()}
            for idx, vid in enumerate(vertex_ids):
                if idx == 0:
                    continue
                d_xi = delta_xi[(idx - 1) * 6:idx * 6]
                update_se3 = SE3.exp(d_xi)
                self.vertices[vid] = self.vertices[vid].compose(update_se3)

            new_cost = self.total_cost()
            if new_cost < current_cost:
                current_damping = max(1e-6, current_damping / 2.0)
                current_cost = new_cost
                if np.linalg.norm(delta_xi) < 1e-5:
                    break
            else:
                self.vertices = backup_vertices
                current_damping *= 4.0

        return current_cost
