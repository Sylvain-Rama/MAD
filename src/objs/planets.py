import numpy as np
from objs.common_schemas import MovableObject
from objs.constants import G
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

    @property
    def escape_velocity(self):
        return np.sqrt(2 * G * self.mass / self.radius)

    @property
    def gravity_at_surface(self):
        surf_obj = MovableObject(position=[self.radius, 0, 0])

        return self.gravity(surf_obj)

    def atmosphere(self, obj: MovableObject, drag_coeff: float) -> float:
        altitude = obj.magnitude - self.radius
        # rho = math.exp(-altitude / self.atmosphere_height)
        # v = math.hypot(vel.vx, vel.vy)
        # drag_x = -drag_coeff * rho * v * vel.vx
        # drag_y = -drag_coeff * rho * v * vel.vy
        return 0.0  # drag_x, drag_y

    def __repr__(self):
        return f"Planet {self.name} at {self.position}, mass {self.mass}, radius {self.radius}."
