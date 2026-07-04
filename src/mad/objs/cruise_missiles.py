"""Cruise missiles are designed to fly at low altitudes and deliver a payload to a target.
This module defines the CruiseMissile class, which is a type of guided missile."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from mad.objs import BallisticObj, GuidedObj, Planet
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

    def create(self, position: NDArray, velocity: NDArray | None = None, t: float = 0.0) -> "CruiseMissile":
        return CruiseMissile(config=self, position=position, velocity=velocity, t=t)


class CruiseMissile(BallisticObj, GuidedObj):
    def __init__(self, config: CruiseMissileConfig, position: NDArray, velocity: NDArray | None = None, t: float = 0.0):
        BallisticObj.__init__(
            self,
            position=position,
            velocity=velocity,
            name=config.name,
            mass=config.mass,
            area=config.area,
            Cd=config.Cd,
        )
        self.config = config
        self.guidance = config.guidance
        self.guidance_results = self.guidance.get_guidance(self, t)
        self.t = t
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
        return self.config.thrust_acc

    def update(self, dt: float) -> None:
        self.total_distance_traveled += float(np.linalg.norm(self.velocity)) * dt
        if self.total_distance_traveled >= self.config.max_range_m:
            self.motor_active = False
        self.t += dt
        self.guidance_results = self.guidance.get_guidance(self, self.t)
        if self.guidance_results.state == GuidanceStates.DETONATE:
            self.detonate()
        return None

    def accelerations(self, planet: Planet) -> NDArray:
        if self.distance(planet) <= planet.radius:
            logger["Missile"].info(f"{self.t:<.2f}s - {self.name} hit the ground.")
            self.active = False
            return np.zeros_like(self.velocity)

        gravity = planet.gravity(self)
        drag = planet.drag(self)
        thrust = np.zeros_like(self.velocity)

        if self.motor_active:
            # Apply guidance direction directly. The guidance returns a fractional vector
            # (components scaled relative to thrust_acc), so multiply directly without
            # renormalizing to preserve the absolute radial/tangential magnitudes.

            d = self.guidance_results.direction
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-8:
                thrust += self.thrust_acc * d

        return gravity + drag + thrust

    def detonate(self):
        logger["Missile"].info(
            f"{self.t:<.2f}s - Warhead {self.name} detonated with yield {self.config.yield_kt:.2f} kt."
        )
        self.active = False

    def degrade(self):
        """Degrade the missile, e.g. when being intercepted."""
        logger["Missile"].info(f"{self.t:<.2f}s - {self.name} degraded.")
        self.active = False
