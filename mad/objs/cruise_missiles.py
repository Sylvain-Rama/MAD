"""Cruise missiles are designed to fly at low altitudes and deliver a payload to a target.
This module defines the CruiseMissile class, which is a type of guided missile."""

from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Collection
import numpy as np
from numpy.typing import NDArray
from mad.objs.base import Body
from mad.objs.planets import Planet
from mad.objs.battle_computers import ComputerCommand
from mad.utils.logger import SourceLogger

from mad.guidances import Guidance, GuidanceManager, GuidanceStates

logger = SourceLogger()


@dataclass
class CruiseMissileConfig:
    mass: float  # kg
    ref_radius: float  # m
    Cd: float
    guidance: "Guidance | GuidanceManager"
    thrust_acc: float = 50.0  # m/s² — acceleration to reach cruise speed quickly
    name: str = "CruiseMissile"

    max_range_m: float = 1_000_000.0  # m
    yield_kt: float = 0.0  # kt — default to conventional warhead

    def __post_init__(self):
        self.area = np.pi * self.ref_radius**2

    def create(
        self,
        position: NDArray,
        velocity: NDArray | None = None,
        t: float = 0.0,
        reference_planet: Planet | None = None,
        gravity_bodies: Collection[Planet] | None = None,
    ) -> "CruiseMissile":
        return CruiseMissile(
            config=self,
            position=position,
            velocity=velocity,
            t=t,
            reference_planet=reference_planet,
            gravity_bodies=gravity_bodies,
        )


class _CruiseEngine:
    def __init__(self, thrust_acc: float):
        self._thrust_acc = float(thrust_acc)

    @property
    def thrust_acc(self) -> float:
        return self._thrust_acc

    def update(self, body, dt: float, command: ComputerCommand | None = None) -> None:
        return None


class CruiseMissile(Body):
    def __init__(
        self,
        config: CruiseMissileConfig,
        position: NDArray,
        velocity: NDArray | None = None,
        t: float = 0.0,
        reference_planet: Planet | None = None,
        gravity_bodies: Collection[Planet] | None = None,
    ):
        super().__init__(
            position=position,
            velocity=velocity,
            name=config.name,
            mass=config.mass,
            area=config.area,
            Cd=config.Cd,
            guidance=deepcopy(config.guidance) if config.guidance is not None else None,
            engine=_CruiseEngine(config.thrust_acc),
            t=t,
            reference_planet=reference_planet,
            gravity_bodies=gravity_bodies,
        )
        self.config = config
        self.guidance_results = self.guidance.get_guidance(self, t) if self.guidance is not None else None
        self.total_distance_traveled = 0.0
        self.motor_active = True

    @property
    def burned_fraction(self) -> float:
        return 1.0

    @property
    def has_thrust(self) -> bool:
        return True

    @property
    def thrust_acc(self) -> float:
        return self.engine.thrust_acc if self.engine is not None else 0.0

    def _update_config(self):
        """Update the missile's configuration based on guidance results."""
        if self.guidance_results is None:
            return
        if self.guidance_results.modify_config is not None:
            for attr, value in self.guidance_results.modify_config.items():
                if hasattr(self.config, attr):
                    setattr(self.config, attr, value)
                    logger["Missile"].info(
                        f"{self.t:<.2f}s - {self.name} changed config attribute {attr} to {value} at {self.t:.2f}."
                    )
                else:
                    logger["Missile"].warning(
                        f"{self.t:<.2f}s - {self.name} has no config attribute {attr} to change at {self.t:.2f}."
                    )

    def update(self, dt: float, command: ComputerCommand | None = None) -> list[Body] | None:
        self.t += dt
        self.total_distance_traveled += float(np.linalg.norm(self.velocity)) * dt
        if self.total_distance_traveled >= self.config.max_range_m:
            self.motor_active = False

        if self.guidance is not None and hasattr(self.guidance, "get_guidance"):
            self.guidance_results = self.guidance.get_guidance(self, self.t)
            self._update_config()
            if self.guidance_results is not None and self.guidance_results.state == GuidanceStates.DETONATE:
                self.detonate()
        return None

    def _on_impact(self, planet: Planet) -> None:
        if self.guidance is not None:
            target_distance = planet.surface_distance(self, self.guidance.target)
            logger["Missile"].info(
                f"{self.t:<.2f}s - {self.name} hit the ground at {target_distance:.2f} m from target!"
            )
        self.detonate()

    def _thrust_acceleration(self) -> NDArray:
        if not self.motor_active or self.guidance_results is None:
            return np.zeros_like(self.velocity)

        # Guidance returns a fractional vector (components already scaled relative to thrust_acc),
        # so multiply directly without renormalizing to preserve the absolute radial/tangential magnitudes.
        d = self.guidance_results.direction
        if np.linalg.norm(d) > 1e-8:
            return self.thrust_acc * d
        return np.zeros_like(self.velocity)

    def detonate(self):
        logger["Missile"].info(
            f"{self.t:<.2f}s - Warhead {self.name} detonated with yield {self.config.yield_kt:.2f} kt."
        )
        self.active = False

    def degrade(self):
        """Degrade the missile, e.g. when being intercepted."""
        logger["Missile"].info(f"{self.t:<.2f}s - {self.name} degraded.")
        self.active = False
