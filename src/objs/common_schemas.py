from dataclasses import dataclass
import math


@dataclass
class Position:
    x: float
    y: float

    def distance_to_core(self):
        return math.hypot(self.x, self.y)

    def altitude(self, radius):
        r = self.distance_to_core()
        return max(0.0, r - radius)

    def __repr__(self):
        return f"Position {self.x:4.2f}, {self.y:4.2f}"


@dataclass
class Velocity:
    vx: float
    vy: float

    def __repr__(self):
        return f"Velocity {self.vx:4.2f}, {self.vy:4.2f}"
