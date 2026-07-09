from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class TrackState3D:
    x: float
    y: float
    z: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    heading_rad: float
    heading_rate_radps: float

    def __post_init__(self) -> None:
        values = (
            self.x,
            self.y,
            self.z,
            self.velocity_x,
            self.velocity_y,
            self.velocity_z,
            self.heading_rad,
            self.heading_rate_radps,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("track state values must be finite")

    def as_vector(self) -> tuple[float, float, float, float, float, float, float, float]:
        return (
            self.x,
            self.y,
            self.z,
            self.velocity_x,
            self.velocity_y,
            self.velocity_z,
            self.heading_rad,
            self.heading_rate_radps,
        )

    def predict(self, elapsed_seconds: float) -> "TrackState3D":
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed time must be finite and non-negative")
        return TrackState3D(
            x=self.x + self.velocity_x * elapsed_seconds,
            y=self.y + self.velocity_y * elapsed_seconds,
            z=self.z + self.velocity_z * elapsed_seconds,
            velocity_x=self.velocity_x,
            velocity_y=self.velocity_y,
            velocity_z=self.velocity_z,
            heading_rad=self.heading_rad + self.heading_rate_radps * elapsed_seconds,
            heading_rate_radps=self.heading_rate_radps,
        )