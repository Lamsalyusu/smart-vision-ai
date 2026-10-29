from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x, self.y, self.z)):
            raise ValueError("vector coordinates must be finite")

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scaled(self, factor: float) -> "Vector3":
        if not math.isfinite(factor):
            raise ValueError("scale factor must be finite")
        return Vector3(self.x * factor, self.y * factor, self.z * factor)

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


@dataclass(frozen=True, slots=True)
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        components = (self.w, self.x, self.y, self.z)
        if not all(math.isfinite(value) for value in components):
            raise ValueError("quaternion components must be finite")
        norm_squared = sum(value * value for value in components)
        if not math.isclose(norm_squared, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("rotation quaternion must have unit length")

    @classmethod
    def from_components(cls, w: float, x: float, y: float, z: float) -> "Quaternion":
        components = (w, x, y, z)
        if not all(math.isfinite(value) for value in components):
            raise ValueError("quaternion components must be finite")
        norm = math.sqrt(sum(value * value for value in components))
        if norm == 0:
            raise ValueError("zero quaternion cannot represent a rotation")
        return cls(*(value / norm for value in components))

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(1.0, 0.0, 0.0, 0.0)

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def multiplied(self, other: "Quaternion") -> "Quaternion":
        return Quaternion.from_components(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def rotate(self, vector: Vector3) -> Vector3:
        vector_part = Vector3(self.x, self.y, self.z)
        doubled_cross = vector_part.cross(vector).scaled(2.0)
        return vector + doubled_cross.scaled(self.w) + vector_part.cross(doubled_cross)


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    frame_id: str

    def __post_init__(self) -> None:
        if not self.frame_id or self.frame_id.strip() != self.frame_id:
            raise ValueError("frame id must be non-empty and have no surrounding whitespace")


@dataclass(frozen=True, slots=True)
class Transform3D:
    from_frame: CoordinateFrame
    to_frame: CoordinateFrame
    translation: Vector3
    rotation: Quaternion

    def transform_point(self, point: Vector3) -> Vector3:
        return self.rotation.rotate(point) + self.translation

    def inverse(self) -> "Transform3D":
        inverse_rotation = self.rotation.conjugate()
        inverse_translation = inverse_rotation.rotate(self.translation.scaled(-1.0))
        return Transform3D(
            from_frame=self.to_frame,
            to_frame=self.from_frame,
            translation=inverse_translation,
            rotation=inverse_rotation,
        )

    def then(self, following: "Transform3D") -> "Transform3D":
        if self.to_frame != following.from_frame:
            raise ValueError("transform frames do not compose")
        return Transform3D(
            from_frame=self.from_frame,
            to_frame=following.to_frame,
            translation=following.rotation.rotate(self.translation) + following.translation,
            rotation=following.rotation.multiplied(self.rotation),
        )