"""Base classes for all objects in the simulation.

The canonical runtime object for the physics loop is ``Body``: a single shared
simulated body with position, velocity, mass, drag, and optional guidance/engine
behaviour attached via composition.

Compatibility aliases such as ``BallisticObj`` remain available for older code
paths and external consumers during the migration to the simpler model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from collections.abc import Collection
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from mad.utils.base_utils import to_vec3, normalize
from mad.utils.logger import SourceLogger

if TYPE_CHECKING:
    from mad.objs.battle_computers import ComputerCommand
    from mad.guidances.base_guidances import Guidance, GuidanceManager, GuidanceResults

logger = SourceLogger()

if TYPE_CHECKING:
    from mad.objs.planets import Planet


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


class BallisticObj(MovableObj):
    """Compatibility base class for physics-bearing objects.

    ``Body`` is the preferred canonical implementation for new or refactored
    objects. ``BallisticObj`` is retained so existing object factories and tests
    continue to work while the project migrates to the simplified composition-based
    model.
    """

    def __init__(
        self,
        position: list[float] | NDArray,
        velocity: list[float] | NDArray | None = None,
        name: str = "BallisticObject",
        mass: float = 1.0,
        area: float = 0.01,
        Cd: float = 0.47,
    ):
        MovableObj.__init__(self, position, velocity, name)
        self._mass = mass
        self._area = area
        self.Cd = Cd

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def area(self) -> float:
        return self._area

    @mass.setter
    def mass(self, value: float):
        if value <= 0:
            raise ValueError("Mass must be positive.")
        self._mass = value

    @area.setter
    def area(self, value: float):
        if value <= 0:
            raise ValueError("Area must be positive.")
        self._area = value

    def integrate(self, dt: float, planet: "Planet | None" = None) -> None:
        """Advance position and velocity by one time step using Velocity Verlet integration."""
        a0 = self.accelerations(planet)
        self.position += self.velocity * dt + 0.5 * a0 * dt**2
        a1 = self.accelerations(planet)
        self.velocity += 0.5 * (a0 + a1) * dt

    def accelerations(self, planet: "Planet | None" = None) -> NDArray:
        raise NotImplementedError("BallisticObj subclasses must implement accelerations().")


class Body(BallisticObj):
    """Simplified shared implementation for all dynamic bodies in the simulation."""

    def __init__(
        self,
        position: list[float] | NDArray,
        velocity: list[float] | NDArray | None = None,
        name: str = "Body",
        mass: float = 1.0,
        area: float = 0.01,
        Cd: float = 0.47,
        guidance: Guidance | GuidanceManager | None = None,
        engine: Any | None = None,
        t: float = 0.0,
        planet: "Planet | None" = None,
        gravity_bodies: Collection["Planet"] | None = None,
    ):
        super().__init__(position=position, velocity=velocity, name=name, mass=mass, area=area, Cd=Cd)
        self.guidance = guidance
        self.engine: Any = engine
        self.t = t
        self.reference_planet = planet
        self.gravity_bodies: frozenset["Planet"] = frozenset(
            gravity_bodies if gravity_bodies is not None else (() if planet is None else (planet,))
        )
        self.guidance_results: GuidanceResults | None = None

    @property
    def planet(self) -> "Planet | None":
        """Compatibility alias for the body's current reference planet."""
        return self.reference_planet

    @planet.setter
    def planet(self, value: "Planet | None") -> None:
        self.reference_planet = value

    def bind_environment(
        self,
        planet: "Planet | None" = None,
        gravity_bodies: Collection["Planet"] | None = None,
    ) -> None:
        """Optionally set local context and replace the fixed gravity sources."""
        if planet is not None:
            self.reference_planet = planet
        if gravity_bodies is not None:
            self.gravity_bodies = frozenset(gravity_bodies)
        self._bind_unconfigured_guidance_planets()

    def set_reference_planet(self, planet: "Planet | None") -> None:
        """Change local drag, impact, and reference-geometry context."""
        self.reference_planet = planet

    def set_planet(self, planet: "Planet | None") -> None:
        """Compatibility wrapper for binding a single primary planet."""
        self.set_reference_planet(planet)

    def _bind_unconfigured_guidance_planets(self) -> None:
        """Initialize missing guidance planet references without changing configured ones."""
        if self.guidance is None:
            return
        guidances = getattr(self.guidance, "guidances", (self.guidance,))
        for guidance in guidances:
            if getattr(guidance, "planet", None) is None:
                guidance.planet = self.reference_planet

    def _primary_planet(self, planet: "Planet | None" = None) -> "Planet | None":
        """Return the configured reference planet, or a legacy call-site fallback."""
        return getattr(self, "reference_planet", None) or planet

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
        return self.engine is not None and getattr(self.engine, "thrust_acc", None) is not None

    @property
    def burned_fraction(self) -> float:
        if self.engine is None:
            return 0.0
        burned = getattr(self.engine, "burned_fraction", None)
        return float(burned) if burned is not None else 0.0

    @property
    def thrust_acc(self) -> float:
        if self.engine is None:
            return 0.0
        thrust_value: Any = getattr(self.engine, "thrust_acc", None)
        if callable(thrust_value):
            try:
                thrust_func = cast(Any, thrust_value)
                return float(thrust_func(self))
            except TypeError:
                return 0.0
        return float(thrust_value) if thrust_value is not None else 0.0

    def update(self, dt: float, command: "ComputerCommand | None" = None) -> list["Body"] | None:
        self.t += dt
        guidance = self.guidance
        if guidance is not None and hasattr(guidance, "get_guidance"):
            self.guidance_results = guidance.get_guidance(self, self.t)
        if self.engine is not None and hasattr(self.engine, "update"):
            self.engine.update(self, dt, command)
        return None

    def accelerations(self, planet: "Planet | None" = None) -> NDArray:
        reference_planet = self.reference_planet if planet is None else planet
        if reference_planet is None:
            raise RuntimeError("Body must have a reference planet before acceleration can be calculated.")

        if self.distance(reference_planet) <= reference_planet.radius:
            self.active = False
            return np.zeros_like(self.velocity)

        gravity = self._gravity_acceleration()
        drag = reference_planet.drag(self)
        thrust = np.zeros_like(self.velocity)

        if self.engine is not None:
            engine_thrust = self.thrust_acc
            if engine_thrust > 0.0 and self.guidance is not None:
                guidance_result = self.guidance_results
                if guidance_result is not None:
                    direction = guidance_result.direction
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-8:
                        thrust = engine_thrust * direction / direction_norm

        return gravity + drag + thrust


@runtime_checkable
class ReleasableConfig(Protocol):
    """Protocol for config objects that can produce a BallisticObj via a factory method.

    Any dataclass with a ``name`` field and a ``create()`` method satisfies this
    protocol and can be used as a missile payload config.  This includes
    ``RVConfig``, ``RocketConfig``, ``CruiseMissileConfig``, and any other
    config whose ``create()`` returns a ``BallisticObj``.
    """

    name: str

    def create(self, position: NDArray, velocity: NDArray | None, t: float) -> "BallisticObj":
        """Instantiate and return the deployed object at the given state."""
        ...
