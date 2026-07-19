"""Satellites are payloads that can be launched into orbit and will be affected by gravity and drag forces."""

from dataclasses import dataclass
from mad.objs.base import BallisticObj
from mad.objs.projectiles import ProjectileConfig
from mad.objs.battle_computers import ComputerCommand
from mad.guidances import Guidance, GuidanceManager
from mad.utils.logger import SourceLogger
import numpy as np
from copy import deepcopy
from numpy.typing import NDArray

logger = SourceLogger()


@dataclass
class SatelliteConfig:
    mass: float  # kg
    ref_radius: float  # m
    guidance: Guidance | GuidanceManager
    Cd: float = 0.47  # sphere
    name: str = "Satellite"

    def __post_init__(self):
        self.area = np.pi * self.ref_radius**2

    def create(self, position: NDArray, velocity: NDArray, t: float) -> "Satellite":
        return Satellite(config=self, position=position, velocity=velocity, t=t)


class Satellite(BallisticObj):
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
        BallisticObj.__init__(self, pos, vel, config.name, config.mass, config.area, config.Cd)  # type: ignore
        self.t = t
        self.config = config
        if getattr(config, "guidance", None) is not None:
            self.guidance = deepcopy(config.guidance)

    def accelerations(self, planet) -> NDArray:
        # Typically, we can ignore drag for satellites.
        if self.distance(planet) <= planet.radius:
            logger["Satellite"].info(f"{self.t:<.2f}s - {self.name} landed on the ground!")
            self.active = False
            return np.zeros_like(self.velocity)

        gravity_acc = planet.gravity(self)

        return gravity_acc

    def update(self, dt: float, command: ComputerCommand | None = None) -> None:
        self.t += dt

        return None


class Sputnik(Satellite):
    def update(self, dt: float, command: ComputerCommand | None = None) -> None:
        self.t += dt
        # Sputnik beeps from time to time.
        if self.t % 4000 < dt:
            logger["Satellite"].info(f"{self.t:<.2f}s - {self.name} -- Beep Beep!")

        return None


@dataclass
class SputnikConfig(SatelliteConfig):
    name: str = "Sputnik"

    def create(self, position: NDArray, velocity: NDArray, t: float) -> "Sputnik":
        logger["Satellite"].info(f"{t:<.2f}s - {self.name} released into orbit -- Beep Beep!")
        return Sputnik(config=self, position=position, velocity=velocity, t=t)
