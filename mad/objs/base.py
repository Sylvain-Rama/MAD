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
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from mad.utils.base_utils import to_vec3, normalize
from mad.utils.logger import SourceLogger

logger = SourceLogger()

if TYPE_CHECKING:
    from mad.objs.planets import Planet
    from mad.objs.battle_computers import ComputerCommand


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

    def __repr__(self):
        a = "active" if self.active else "inactive"
        return f"{self.name} at {self.position}, velocity {self.velocity}, {a}."

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MovableObj):
            return False
        return self._id == other._id


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

    def integrate(self, dt: float, planet: "Planet") -> None:
        """Advance position and velocity by one time step using Velocity Verlet integration."""
        a0 = self.accelerations(planet)
        self.position += self.velocity * dt + 0.5 * a0 * dt**2
        a1 = self.accelerations(planet)
        self.velocity += 0.5 * (a0 + a1) * dt

    def degrade(self):
        """Mark the object inactive when it is degraded or destroyed."""
        self.active = False


class Body(BallisticObj):
    """Simplified shared implementation for all dynamic bodies in the simulation.

    This keeps the existing physics-heavy API of BallisticObj while allowing
    optional guidance/engine strategies to be attached without requiring a deep
    inheritance hierarchy for every concrete object type.
    """

    def __init__(
        self,
        position: list[float] | NDArray,
        velocity: list[float] | NDArray | None = None,
        name: str = "Body",
        mass: float = 1.0,
        area: float = 0.01,
        Cd: float = 0.47,
        guidance: object | None = None,
        engine: object | None = None,
        t: float = 0.0,
    ):
        super().__init__(position=position, velocity=velocity, name=name, mass=mass, area=area, Cd=Cd)
        self.guidance = guidance
        self.engine = engine
        self.t = t
        self.guidance_results = None

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
        thrust = getattr(self.engine, "thrust_acc", None)
        if callable(thrust):
            return float(thrust(self))
        return float(thrust) if thrust is not None else 0.0

    def update(self, dt: float, command: object | None = None) -> list["BallisticObj"] | None:
        self.t += dt
        if self.guidance is not None and hasattr(self.guidance, "get_guidance"):
            self.guidance_results = self.guidance.get_guidance(self, self.t)
        if self.engine is not None and hasattr(self.engine, "update"):
            self.engine.update(self, dt, command)
        return None

    def accelerations(self, planet: "Planet") -> NDArray:
        if self.distance(planet) <= planet.radius:
            self.active = False
            return np.zeros_like(self.velocity)

        gravity = planet.gravity(self)
        drag = planet.drag(self)
        thrust = np.zeros_like(self.velocity)

        if self.engine is not None:
            engine_thrust = self.thrust_acc
            if engine_thrust > 0.0 and self.guidance is not None and hasattr(self.guidance, "get_guidance"):
                try:
                    guidance_result = self.guidance.get_guidance(self, self.t)
                except TypeError:
                    guidance_result = None

                if guidance_result is not None and getattr(guidance_result, "direction", None) is not None:
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
