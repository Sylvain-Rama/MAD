"""This module defines the Projectile class, which represents a simple ballistic object that can be launched
and will be affected by gravity and drag forces.
The Projectile class is a subclass of BallisticObj and is initialized with a ProjectileConfig dataclass
that contains its properties such as mass, reference radius, drag coefficient, and initial position and velocity.
"""

from dataclasses import dataclass, asdict
import numpy as np
from numpy.typing import NDArray
from mad.objs.base import BallisticObj
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

    def create(self, position: NDArray, velocity: NDArray | None, t: float = 0.0) -> "Projectile":
        return Projectile(self, position, velocity, t)


class Projectile(BallisticObj):
    def __init__(
        self,
        config: ProjectileConfig,
        position: list[float] | NDArray,
        velocity: list[float] | NDArray | None = None,
        t: float = 0.0,
    ):
        super().__init__(position, velocity, config.name, config.mass, config.area, config.Cd)
        self.config = config
        self.t = t

    def accelerations(self, planet) -> NDArray:
        if self.distance(planet) <= planet.radius:
            logger["Projectile"].info(f"{self.t:<.2f}s - {self.name} landed on the ground!")
            self.active = False
            return np.zeros_like(self.velocity)

        gravity_acc = planet.gravity(self)
        drag_acc = planet.drag(self)

        return gravity_acc + drag_acc

    def update(self, dt: float, command: ComputerCommand | None = None) -> None:
        self.t += dt
        # Nothing to update internally: it's a rock...
        return None
