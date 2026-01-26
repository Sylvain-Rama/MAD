from dataclasses import dataclass
import math


@dataclass
class Position:
    x: float
    y: float

    @classmethod
    def from_angle_radius(cls, angle, radius):
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        return cls(x, y)

    def distance_to_core(self) -> float:
        dist = math.hypot(self.x, self.y)
        return 1e-10 if dist == 0 else dist

    def altitude(self, radius: float) -> float:
        r = self.distance_to_core()
        return max(0.0, r - radius)

    def __repr__(self) -> str:
        return f"Position {self.x:4.2f}, {self.y:4.2f}"

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Velocity:
    vx: float
    vy: float

    def __repr__(self) -> str:
        return f"Velocity {self.vx:4.2f}, {self.vy:4.2f}"
