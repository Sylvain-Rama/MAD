from dataclasses import dataclass
from mad.objs.base import Payload
from mad.objs.projectiles import ProjectileConfig
from mad.logger import SourceLogger
import numpy as np
from numpy.typing import NDArray

logger = SourceLogger()


@dataclass
class SatelliteConfig:
    mass: float  # kg
    ref_radius: float  # m
    Cd: float = 0.47  # sphere
    name: str = "Satellite"

    def __post_init__(self):
        self.area = np.pi * self.ref_radius**2

    def create(self, position: NDArray, velocity: NDArray, t: float) -> "Sputnik":
        return Sputnik(config=self, position=position, velocity=velocity, t=t)


class Sputnik(Payload):
    def __init__(
        self,
        config: "ProjectileConfig | SatelliteConfig",
        position: NDArray | None = None,
        velocity: NDArray | None = None,
        t: float = 0.0,
    ):
        if isinstance(config, SatelliteConfig):
            if position is None:
                raise ValueError("position must be provided when using SatelliteConfig.")
            pos = position
            vel = velocity
        else:
            pos = position if position is not None else np.array(config.position)
            vel = velocity if velocity is not None else config.velocity
        Payload.__init__(self, pos, vel, config.name, config.mass, config.area, config.Cd, t)
        self.config = config

    def accelerations(self, planet) -> NDArray:

        if self.distance(planet) <= planet.radius:
            logger["Satellite"].info(f"{self.name} landed on the ground!")
            self.active = False
            return np.zeros_like(self.velocity)

        gravity_acc = planet.gravity(self)

        return gravity_acc

    def update(self, dt: float):
        self.t += dt
        # Sputniks are indestructible and unaffected by drag, we only need to beep from time to time.
        if self.t // 60 == 0:
            logger["Satellite"].info(f"{self.name} -- Beep Beep!")

        return None
