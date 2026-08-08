"""This module defines the Planet class, which represents a celestial body with mass, radius, and other properties.
The Planet class will calculate gravitational forces, drag forces, and distances between objects on its surface."""

import numpy as np
from numpy.typing import NDArray
from mad.objs.base import MovableObj, BallisticObj
from mad.configs import G
from dataclasses import dataclass, asdict


@dataclass
class PlanetConfig:
    position: list[float]
    radius: float
    mass: float
    velocity: list[float] | None = None
    name: str = "Planet"
    rho0: float = 1.225
    atmosphere_height: float = 8000.0

    @property
    def to_dict(self):
        return asdict(self)

    def create(self) -> "Planet":
        return Planet(self)


class Planet(MovableObj):
    def __init__(self, config: PlanetConfig):
        super().__init__(config.position, config.velocity, config.name)
        self.mass = config.mass
        self.radius = config.radius
        self.atmosphere_height = config.atmosphere_height
        self.rho0 = config.rho0
        self.mu = self.mass * G

    # Helpers for the user to know the planet's properties without having to calculate them manually.
    @property
    def escape_velocity(self):
        return np.sqrt(2 * self.mu / self.radius)

    @property
    def orbital_velocity(self):
        return np.sqrt(self.mu / self.radius)

    @property
    def gravity_at_surface(self):
        surface_pos = np.zeros_like(self.position)
        surface_pos[0] = self.radius
        surf_obj = MovableObj(position=surface_pos)

        # We take the first element as this is where distance = planet.radius.
        return np.linalg.norm(self.gravity(surf_obj)[0])

    def __repr__(self):
        return (
            f"Planet {self.name} at {self.position}\n"
            f"Mass {self.mass:.2e} kg, Radius {self.radius / 1000} km.\n"
            f"Gravity at surface: {self.gravity_at_surface:.2f} m/s^2\n"
            f"Orbital velocity: {self.orbital_velocity:.2f} m/s\n"
            f"Escape velocity: {self.escape_velocity:.2f} m/s"
        )

    def drag(self, obj: BallisticObj) -> NDArray:
        # Returns the drag acceleration vector (m/s^2) on the object, if it is in the atmosphere.
        drag = np.zeros_like(obj.velocity)
        alt = max(0.0, float(np.linalg.norm(obj.position - self.position)) - self.radius)

        if alt > 0:
            rho = self.rho0 * np.exp(-alt / self.atmosphere_height)
            v_mag = np.linalg.norm(obj.velocity)
            if v_mag > 0:
                drag = -0.5 * rho * obj.Cd * obj.area * v_mag * obj.velocity / obj.mass

        return drag

    def gravity(self, other: MovableObj) -> NDArray:
        # Returns the gravity acceleration vector (m/s^2) on the other object.
        r_vec = other.position - self.position
        dist = np.linalg.norm(r_vec)
        if dist < 1e-6:
            return np.zeros_like(other.position)

        return -self.mu * r_vec / dist**3

    def surface_distance(self, obj1: MovableObj, obj2: MovableObj) -> float:
        # Will give the linear distance (m) between 2 objects placed at the surface.

        cos_angle = np.dot(obj1.position, obj2.position) / (
            np.linalg.norm(obj1.position) * np.linalg.norm(obj2.position)
        )

        angle = np.arccos(np.clip(cos_angle, -1, 1))

        return self.radius * angle
