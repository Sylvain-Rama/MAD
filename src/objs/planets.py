import numpy as np
import matplotlib.pyplot as plt
from common_schemas import MovableObject
from numpy.typing import NDArray
from constants import G, EARTH_SETTINGS
from dataclasses import dataclass, asdict


@dataclass
class PlanetConfig:
    position: list[float]
    radius: float
    mass: float
    spin_rate: float
    velocity: list[float] | None = None
    name: str = "Planet"

    @property
    def to_dict(self):
        return asdict(self)


class Planet(MovableObject):
    def __init__(self, config: PlanetConfig):
        super().__init__(config.position, config.velocity, config.mass)
        self.radius = config.radius
        self.spin_rate = config.spin_rate
        self.name = config.name

    def atmosphere(self, obj: MovableObject, drag_coeff: float) -> float:
        altitude = obj.magnitude - self.radius
        # rho = math.exp(-altitude / self.atmosphere_height)
        # v = math.hypot(vel.vx, vel.vy)
        # drag_x = -drag_coeff * rho * v * vel.vx
        # drag_y = -drag_coeff * rho * v * vel.vy
        return 0.0  # drag_x, drag_y

    def __repr__(self):
        return f"Planet {self.name} at {self.position}, mass {self.mass}, radius {self.radius}."


if __name__ == "__main__":
    earth = Planet(PlanetConfig(**EARTH_SETTINGS))
    obj = MovableObject(position=[earth.radius, 0, 0])

    print(earth)
    print(f"Gravity at surface: {earth.gravity(obj):4.2f}")
