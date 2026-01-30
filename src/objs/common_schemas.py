from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray


@dataclass
class MovableObject:
    position: NDArray[np.floating]
    velocity: NDArray[np.floating]

    active: bool = True

    @property
    def magnitude(self) -> np.floating:
        return np.sqrt(np.sum(np.square(self.position)))

    @property
    def norm(self) -> NDArray[np.floating]:
        return self.position / self.magnitude

    @classmethod
    def from_list(cls, pos: list[float], vel: list[float]) -> "MovableObject":
        return cls(np.array(pos), np.array(vel))

    def __repr__(self):
        a = "active" if self.active else "inactive"
        return f"Movable at {self.position}, velocity {self.velocity}, {a}."


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


if __name__ == "__main__":
    obj = MovableObject(position=np.array([3, 3, 0]), velocity=np.array([0, 0, 0]))

    print(obj.magnitude)
    print(obj.norm)
