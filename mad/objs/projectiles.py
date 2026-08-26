"""This module defines the Projectile class, which represents a simple ballistic object that can be launched
and will be affected by gravity and drag forces.
The Projectile class is a subclass of BallisticObj and is initialized with a ProjectileConfig dataclass
that contains its properties such as mass, reference radius, drag coefficient, and initial position and velocity.
"""

from dataclasses import dataclass, asdict
from collections.abc import Collection
import numpy as np
from numpy.typing import NDArray
from mad.objs.base import Body
from mad.objs.planets import Planet
from mad.utils.logger import SourceLogger
from mad.objs.battle_computers import ComputerCommand

logger = SourceLogger()


@dataclass
class ProjectileConfig:
    mass: float  # kg
    name: str = "Projectile"
    ref_radius: float = 0.01  # m
    Cd: float = 0.47  # sphere

    def __post_init__(self):
        self.area = np.pi * self.ref_radius**2

    @property
    def to_dict(self):
        return asdict(self)

    def create(
        self,
        position: list[float] | NDArray,
        velocity: list[float] | NDArray | None = None,
        t: float = 0.0,
        reference_planet: Planet | None = None,
        gravity_bodies: Collection[Planet] | None = None,
    ) -> "Projectile":
        return Projectile(self, position, velocity, t, reference_planet, gravity_bodies)


class Projectile(Body):
    def __init__(
        self,
        config: ProjectileConfig,
        position: list[float] | NDArray,
        velocity: list[float] | NDArray | None = None,
        t: float = 0.0,
        reference_planet: Planet | None = None,
        gravity_bodies: Collection[Planet] | None = None,
    ):
        super().__init__(
            position,
            velocity,
            config.name,
            config.mass,
            config.area,
            config.Cd,
            reference_planet=reference_planet,
            gravity_bodies=gravity_bodies,
        )
        self.config = config
        self.t = t

    def accelerations(self, planet: Planet | None = None) -> NDArray:
        primary_planet = self._primary_planet(planet)
        if primary_planet is None:
            return np.zeros_like(self.velocity)
        if self.distance(primary_planet) <= primary_planet.radius:
            logger["Projectile"].info(f"{self.t:<.2f}s - {self.name} landed on the ground!")
            self.active = False
            return np.zeros_like(self.velocity)

        gravity_acc = self._gravity_acceleration(primary_planet)
        drag_acc = primary_planet.drag(self)

        return gravity_acc + drag_acc

    def update(self, dt: float, command: ComputerCommand | None = None) -> list[Body] | None:
        self.t += dt
        # Nothing to update internally: it's a rock...
        return None
