"""Base classes for all objects in the simulation.

The canonical runtime object for the physics loop is ``Body``: a single shared
simulated body with position, velocity, mass, drag, and optional guidance/engine
behaviour attached via composition.

``Body`` is the shared physical runtime object; ``MovableObj`` remains the
lightweight geometric base for non-physical entities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from collections.abc import Collection
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mad.objs.engines import Engine
from mad.utils.base_utils import to_vec3, normalize
from mad.utils.logger import SourceLogger

if TYPE_CHECKING:
    from mad.objs.battle_computers import ComputerCommand
    from mad.guidances.base_guidances import Guidance, GuidanceManager, GuidanceResults
    from mad.objs.planets import Planet

logger = SourceLogger()


class MovableObj:
    """MovableObj is the base class for any object that has a position and velocity, and can move in the simulation.
    It provides basic functionalities such as distance calculation and normalization of the position vector.
    It does not have any mass, area or drag coefficient, and is not affected by gravity or drag.
    It is only a geometric point that can move in space.
    Parameters:
    - position: initial position of the object in meters (m)
    - velocity: initial velocity of the object in meters per second (m/s)
    - name: name of the object (string)
    """

    _id_counter: int = 0

    def __init__(
        self,
        position: list[float] | NDArray,
        velocity: list[float] | NDArray | None = None,
        name: str = "MovableObject",
    ):

        self.position = to_vec3(position)
        if velocity is not None:
            self.velocity = to_vec3(velocity)  # m/s
        else:
            self.velocity = np.zeros_like(self.position)
        self.active: bool = True
        self.name = name
        self._id = self.__class__.__name__ + f"_{self.name}_" + str(MovableObj._id_counter)
        MovableObj._id_counter += 1

    @property
    def pos_norm(self) -> NDArray:
        return normalize(self.position)

    @property
    def vel_norm(self) -> NDArray:
        return normalize(self.velocity)

    def distance(self, other: "MovableObj") -> np.floating:
        return np.linalg.norm(self.position - other.position)

    def los(self, other: "MovableObj") -> NDArray:
        return other.position - self.position

    def __repr__(self):
        a = "active" if self.active else "inactive"
        return f"{self.name} at {self.position}, velocity {self.velocity}, {a}."

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MovableObj):
            return False
        return self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)

    def degrade(self) -> None:
        """Mark the object inactive when it is degraded or destroyed."""
        self.active = False


class Body(MovableObj):
    """Shared implementation for all dynamic physical bodies in the simulation."""

    def __init__(
        self,
        position: list[float] | NDArray,
        velocity: list[float] | NDArray | None = None,
        name: str = "Body",
        mass: float = 1.0,
        area: float = 0.01,
        Cd: float = 0.47,
        guidance: Guidance | GuidanceManager | None = None,
        engine: Engine | None = None,
        t: float = 0.0,
        reference_planet: "Planet | None" = None,
        gravity_bodies: Collection["Planet"] | None = None,
    ):
        super().__init__(position=position, velocity=velocity, name=name)
        self._mass = mass
        self._area = area
        self.Cd = Cd
        self.guidance = guidance
        self.engine = engine
        self.t = t
        self._reference_planet = reference_planet
        self.gravity_bodies: frozenset["Planet"] = frozenset(
            gravity_bodies if gravity_bodies is not None else (() if reference_planet is None else (reference_planet,))
        )
        self.guidance_results: GuidanceResults | None = None

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Mass must be positive.")
        self._mass = value

    @property
    def area(self) -> float:
        return self._area

    @area.setter
    def area(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Area must be positive.")
        self._area = value

    def integrate(self, dt: float, planet: "Planet | None" = None) -> None:
        """Advance position and velocity by one time step using Velocity Verlet integration."""
        a0 = self.accelerations(planet)
        self.position += self.velocity * dt + 0.5 * a0 * dt**2
        a1 = self.accelerations(planet)
        self.velocity += 0.5 * (a0 + a1) * dt

    @property
    def reference_planet(self) -> "Planet | None":
        """Compatibility alias for the body's current reference planet."""
        return self._reference_planet

    @reference_planet.setter
    def reference_planet(self, value: "Planet | None") -> None:
        self._reference_planet = value

    def bind_environment(
        self,
        reference_planet: "Planet | None" = None,
        gravity_bodies: Collection["Planet"] | None = None,
    ) -> None:
        """Optionally set local context and replace the fixed gravity sources."""
        if reference_planet is not None:
            self._reference_planet = reference_planet
        if gravity_bodies is not None:
            self.gravity_bodies = frozenset(gravity_bodies)

    def set_reference_planet(self, planet: "Planet | None") -> None:
        """Change local drag, impact, and reference-geometry context."""
        self.reference_planet = planet

    def _gravity_acceleration(self, planet: "Planet | None" = None) -> NDArray:
        gravity_bodies = getattr(self, "gravity_bodies", frozenset())
        if not gravity_bodies and planet is not None:
            gravity_bodies = frozenset((planet,))
        gravity = np.zeros_like(self.velocity)
        for body in gravity_bodies:
            gravity += body.gravity(self)
        return gravity

    @property
    def has_thrust(self) -> bool:
        return self.engine is not None and self.engine.has_thrust

    @property
    def burned_fraction(self) -> float:
        return float(self.engine.burned_fraction) if self.engine is not None else 0.0

    @property
    def thrust_acc(self) -> float:
        return float(self.engine.thrust_acc(self)) if self.engine is not None else 0.0

    def update(self, dt: float, command: "ComputerCommand | None" = None) -> list["Body"] | None:
        self.t += dt
        guidance = self.guidance
        if guidance is not None and hasattr(guidance, "get_guidance"):
            self.guidance_results = guidance.get_guidance(self, self.t)
        if self.engine is not None:
            self.engine.update(self, dt, command)
        return None

    def accelerations(self, planet: "Planet | None" = None) -> NDArray:
        """Template method: subclasses customize impact/drag/thrust via the hooks below instead of overriding this."""
        reference_planet = self.reference_planet if planet is None else planet
        if reference_planet is None:
            raise RuntimeError("Body must have a reference planet before acceleration can be calculated.")

        if self.distance(reference_planet) <= reference_planet.radius:
            self._on_impact(reference_planet)
            self.active = False
            return np.zeros_like(self.velocity)

        gravity = self._gravity_acceleration(reference_planet)
        drag = self._drag_acceleration(reference_planet)
        thrust = self._thrust_acceleration()

        return gravity + drag + thrust

    def _on_impact(self, planet: "Planet") -> None:
        """Hook called when the body reaches the surface of `planet`, before it is deactivated."""
        return None

    def _drag_acceleration(self, planet: "Planet") -> NDArray:
        return planet.drag(self)

    def _thrust_acceleration(self) -> NDArray:
        """Resolve engine thrust along the guidance direction, honoring any requested magnitude cap."""
        engine_thrust = self.thrust_acc
        guidance_result = self.guidance_results
        if engine_thrust <= 0.0 or guidance_result is None:
            return np.zeros_like(self.velocity)

        direction = guidance_result.direction
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 1e-8:
            return np.zeros_like(self.velocity)

        desired_acc = guidance_result.magnitude
        acc = min(engine_thrust, desired_acc) if desired_acc is not None else engine_thrust
        return acc * direction / direction_norm


@runtime_checkable
class ReleasableConfig(Protocol):
    """Protocol for config objects that can produce a Body via a factory method.

    Any dataclass with a ``name`` field and a ``create()`` method satisfies this
    protocol and can be used as a missile payload config.  This includes
    ``RVConfig``, ``RocketConfig``, ``CruiseMissileConfig``, and any other
    config whose ``create()`` returns a ``Body``.
    """

    name: str

    def create(
        self,
        position: NDArray,
        velocity: NDArray | None,
        t: float,
        reference_planet: "Planet | None" = None,
        gravity_bodies: Collection["Planet"] | None = None,
    ) -> "Body":
        """Instantiate and return the deployed object at the given state."""
        ...
