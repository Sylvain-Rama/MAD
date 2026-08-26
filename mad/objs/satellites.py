"""Satellites are payloads that can be launched into orbit and will be affected by gravity and drag forces."""

from dataclasses import dataclass
from mad.objs.base import Body
from mad.objs.planets import Planet
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


class Satellite(Body):
    def __init__(
        self,
        config: SatelliteConfig,
        position: NDArray,
        velocity: NDArray | None = None,
        t: float = 0.0,
    ):
        super().__init__(
            position=position,
            velocity=velocity,
            name=config.name,
            mass=config.mass,
            area=config.area,
            Cd=config.Cd,
            guidance=deepcopy(config.guidance) if getattr(config, "guidance", None) is not None else None,
            t=t,
        )
        self.t = t
        self.config = config

    def _on_impact(self, planet: Planet) -> None:
        logger["Satellite"].info(f"{self.t:<.2f}s - {self.name} landed on the ground!")

    def _drag_acceleration(self, planet: Planet) -> NDArray:
        # Typically, we can ignore drag for satellites.
        return np.zeros_like(self.velocity)

    def update(self, dt: float, command: ComputerCommand | None = None) -> list[Body] | None:
        self.t += dt

        return None


class Sputnik(Satellite):
    def update(self, dt: float, command: ComputerCommand | None = None) -> list[Body] | None:
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
