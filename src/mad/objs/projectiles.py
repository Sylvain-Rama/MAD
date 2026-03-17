from dataclasses import dataclass, asdict
import numpy as np
from numpy.typing import NDArray
from mad.objs.planets import Planet, SimulationInterface
from mad.objs.common_schemas import MovableObject, History
from mad.logger import SourceLogger

logger = SourceLogger()


@dataclass
class ProjectileConfig:
    position: list[float] | NDArray  # m
    mass: float  # kg
    velocity: list[float] | NDArray | None = None  # m / s
    name: str = "Projectile"
    area: float = 0.01  # m^2
    Cd: float = 0.47  # sphere

    @property
    def to_dict(self):
        return asdict(self)


class Projectile(SimulationInterface, MovableObject):
    def __init__(self, config: ProjectileConfig, t: float = 0.0):
        super().__init__(config.position, config.velocity, config.name)
        self.mass = config.mass
        self.area = config.area
        self.Cd = config.Cd
        self.config = config
        self.history = History(position=[self.position.tolist()], velocity=[self.velocity.tolist()], time=[t])
        self.t = t

    def accelerations(self, planet) -> NDArray:

        if self.distance(planet) <= planet.radius:
            logger["Projectile"].info(f"{self.name} landed on the ground!")
            self.active = False
            return np.zeros_like(self.velocity)

        gravity_acc = planet.gravity(self)
        drag_acc = planet.drag(self)

        return gravity_acc + drag_acc

    def integrate(self, dt: float, planet: Planet) -> None:
        # Velocity Verlet for solver.
        a0 = self.accelerations(planet)
        self.position += self.velocity * dt + 0.5 * a0 * dt**2
        a1 = self.accelerations(planet)

        self.velocity += 0.5 * (a0 + a1) * dt

        self.history.update(self.t, self.position.tolist(), self.velocity.tolist())

    def update(self, dt: float):
        self.t += dt
        # Nothing to update internally: it's a rock...
        return None
