import math
from dataclasses import dataclass, field
from enum import Enum


class TrackStatus(str, Enum):
    """Formal lifecycle states for multi-object spatial tracking."""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    RE_IDENTIFIED = "re_identified"
    LOST = "lost"
    RETIRED = "retired"


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

    @property
    def speed_mps(self) -> float:
        return math.sqrt(self.velocity_x**2 + self.velocity_y**2 + self.velocity_z**2)

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


@dataclass(frozen=True, slots=True)
class TrackKinematicLimits:
    max_speed_mps: float = 35.0  # ~126 km/h for urban/vehicular tracking
    max_acceleration_mps2: float = 20.0


@dataclass
class TrackLifecycle:
    """Enterprise state machine governing track transitions and confidence."""
    track_id: int
    class_name: str
    status: TrackStatus = TrackStatus.TENTATIVE
    hit_streak: int = 1
    total_hits: int = 1
    age_frames: int = 1
    time_since_update: int = 0
    min_hits_to_confirm: int = 3
    max_coasting_frames: int = 30
    last_state: TrackState3D | None = None
    kinematic_limits: TrackKinematicLimits = field(default_factory=TrackKinematicLimits)

    def register_hit(self, state: TrackState3D, is_reid: bool = False) -> None:
        """Process an associated observation and transition lifecycle."""
        # Kinematic validation
        if self.last_state is not None:
            dt = max(0.033, self.time_since_update * 0.033)
            dx = math.sqrt(
                (state.x - self.last_state.x) ** 2
                + (state.y - self.last_state.y) ** 2
                + (state.z - self.last_state.z) ** 2
            )
            implied_speed = dx / dt
            if implied_speed > self.kinematic_limits.max_speed_mps * 2.0:
                # Discard implausible teleportation jump
                return

        self.last_state = state
        self.time_since_update = 0
        self.hit_streak += 1
        self.total_hits += 1
        self.age_frames += 1

        if self.status == TrackStatus.TENTATIVE:
            if self.total_hits >= self.min_hits_to_confirm:
                self.status = TrackStatus.CONFIRMED
        elif self.status == TrackStatus.COASTING:
            self.status = TrackStatus.RE_IDENTIFIED if is_reid else TrackStatus.CONFIRMED
        elif self.status == TrackStatus.RE_IDENTIFIED:
            self.status = TrackStatus.CONFIRMED

    def register_miss(self) -> TrackStatus:
        """Process a missed observation during frame update."""
        self.hit_streak = 0
        self.time_since_update += 1
        self.age_frames += 1

        if self.status in (TrackStatus.CONFIRMED, TrackStatus.RE_IDENTIFIED):
            if self.time_since_update <= self.max_coasting_frames:
                self.status = TrackStatus.COASTING
            else:
                self.status = TrackStatus.LOST
        elif self.status == TrackStatus.COASTING and self.time_since_update > self.max_coasting_frames or self.status == TrackStatus.TENTATIVE and self.time_since_update > 2:
            self.status = TrackStatus.LOST

        return self.status

    def retire(self) -> None:
        self.status = TrackStatus.RETIRED

    @property
    def is_active(self) -> bool:
        return self.status in (
            TrackStatus.CONFIRMED,
            TrackStatus.COASTING,
            TrackStatus.RE_IDENTIFIED,
        )

    @property
    def is_visible(self) -> bool:
        return self.status in (TrackStatus.CONFIRMED, TrackStatus.RE_IDENTIFIED)