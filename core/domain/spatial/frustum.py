import math
from collections.abc import Sequence
from dataclasses import dataclass

from core.domain.spatial.transforms import Quaternion, Vector3


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters matrix K."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be strictly positive")
        if not (math.isfinite(self.cx) and math.isfinite(self.cy)):
            raise ValueError("principal point coordinates must be finite")

    def unproject(self, u: float, v: float, depth_m: float) -> Vector3:
        """Back-project a 2D pixel coordinate (u, v) with metric depth to a 3D camera ray."""
        if not math.isfinite(depth_m) or depth_m <= 0:
            raise ValueError("depth must be finite and strictly positive")
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        return Vector3(x, y, depth_m)

    def project(self, point: Vector3) -> tuple[float, float]:
        """Project a 3D point in camera coordinates onto the 2D image plane."""
        if point.z <= 0:
            raise ValueError("cannot project point behind or at camera plane")
        u = (point.x * self.fx / point.z) + self.cx
        v = (point.y * self.fy / point.z) + self.cy
        return u, v


@dataclass(frozen=True, slots=True)
class OrientedBox3D:
    """3D Oriented Bounding Box (O-BB) with 6-DoF pose and spatial extents."""

    center: Vector3
    size: Vector3  # width, height, depth (extents along local axes)
    orientation: Quaternion

    def __post_init__(self) -> None:
        if self.size.x <= 0 or self.size.y <= 0 or self.size.z <= 0:
            raise ValueError("box sizes must be strictly positive")

    @property
    def volume(self) -> float:
        return self.size.x * self.size.y * self.size.z

    def contains_point(self, point: Vector3) -> bool:
        """Test if a 3D point lies inside the oriented bounding box."""
        # Vector from center to point
        rel_vec = Vector3(
            point.x - self.center.x,
            point.y - self.center.y,
            point.z - self.center.z,
        )
        # Transform point into box's local coordinate frame using conjugate orientation
        inv_orient = self.orientation.conjugate()
        local_p = inv_orient.rotate(rel_vec)

        half_w = self.size.x / 2.0
        half_h = self.size.y / 2.0
        half_d = self.size.z / 2.0

        return (
            -half_w <= local_p.x <= half_w
            and -half_h <= local_p.y <= half_h
            and -half_d <= local_p.z <= half_d
        )


def estimate_surface_normal(
    points: Sequence[Vector3],
) -> Vector3:
    """Estimate unit surface normal from a cluster of 3D surface points using covariance analysis."""
    if len(points) < 3:
        raise ValueError("at least 3 non-collinear points required for normal estimation")

    # Approximate normal by cross product of two orthogonal point vectors
    p0 = points[0]
    p1 = points[1]
    p2 = points[2]

    v1 = Vector3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)
    v2 = Vector3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z)

    # Cross product
    nx = v1.y * v2.z - v1.z * v2.y
    ny = v1.z * v2.x - v1.x * v2.z
    nz = v1.x * v2.y - v1.y * v2.x

    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 1e-9:
        # Fallback to standard vertical normal if points are collinear
        return Vector3(0.0, 1.0, 0.0)

    # Normalize and orient towards camera (z < 0)
    normal = Vector3(nx / length, ny / length, nz / length)
    if normal.z > 0:
        return Vector3(-normal.x, -normal.y, -normal.z)
    return normal


def backproject_frustum_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    depth_m: float,
    depth_span_m: float,
    intrinsics: CameraIntrinsics,
) -> OrientedBox3D:
    """Construct 3D Oriented Bounding Box from 2D pixel box and depth estimate."""
    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid 2D bounding box dimensions")

    cx_2d = (x1 + x2) / 2.0
    cy_2d = (y1 + y2) / 2.0

    # Backproject center and corners
    center_3d = intrinsics.unproject(cx_2d, cy_2d, depth_m)
    tl_3d = intrinsics.unproject(x1, y1, depth_m)
    br_3d = intrinsics.unproject(x2, y2, depth_m)

    width = abs(br_3d.x - tl_3d.x)
    height = abs(br_3d.y - tl_3d.y)
    depth = max(0.1, depth_span_m)

    return OrientedBox3D(
        center=center_3d,
        size=Vector3(width, height, depth),
        orientation=Quaternion.identity(),
    )
