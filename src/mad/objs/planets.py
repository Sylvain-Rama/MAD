import numpy as np
from numpy.typing import NDArray
from mad.objs.common_schemas import MovableObject
from mad.objs.constants import G
from dataclasses import dataclass, asdict


@dataclass
class PlanetConfig:
    position: list[float]
    radius: float
    mass: float
    spin_rate: float
    velocity: list[float] | None = None
    name: str = "Planet"
    rho0: float = 1.225
    atmosphere_height: float = 8000.0

    @property
    def to_dict(self):
        return asdict(self)


class Planet(MovableObject):
    def __init__(self, config: PlanetConfig):
        super().__init__(config.position, config.velocity, config.mass, config.name)
        self.radius = config.radius
        self.spin_rate = config.spin_rate
        self.atmosphere_height = config.atmosphere_height
        self.rho0 = config.rho0
        self.mu = self.mass * G

    @property
    def escape_velocity(self):
        return np.sqrt(2 * G * self.mass / self.radius)

    @property
    def gravity_at_surface(self):
        surface_pos = np.zeros_like(self.position)
        surface_pos[0] = self.radius
        surf_obj = MovableObject(position=list(surface_pos))

        return self.gravity(surf_obj)[0]

    def atmosphere_rho(self, obj: MovableObject) -> float:
        rho = 0.0
        alt = max(np.float(0.0), self.distance(obj) - self.radius)  # type: ignore[no-matching-overload]

        if alt > 0:
            rho = self.rho0 * np.exp(-alt / self.atmosphere_height)

        return rho

    def gravity(self, other: "MovableObject") -> NDArray:
        r_vec = other.position - self.position
        dist = np.linalg.norm(r_vec)
        if dist < 1e-6:
            return np.zeros_like(other.position)

        return -self.mu * r_vec / dist**3

    def __repr__(self):
        return f"Planet {self.name} at {self.position}, mass {self.mass}, radius {self.radius}."
