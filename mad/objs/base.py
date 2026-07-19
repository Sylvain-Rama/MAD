"""Base classes for all objects in the simulation.
This includes MovableObj, BallisticObj, GuidedObj, and ReleasableConfig.
See objs/projectiles.py for the implementation of projectiles and missiles.
The base classes provide basic functionalities such as position, velocity, mass, area, drag coefficient, and guidance interfaces.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from mad.utils.base_utils import to_vec3
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
    def normalize(self) -> NDArray[np.floating]:
        norm = np.linalg.norm(self.position)
        if norm < 1e-8:
            return np.zeros_like(self.position)
        return self.position / norm

    def distance(self, other: "MovableObj") -> np.floating:
        return np.linalg.norm(self.position - other.position)

    def __repr__(self):
        a = "active" if self.active else "inactive"
        return f"{self.name} at {self.position}, velocity {self.velocity}, {a}."

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MovableObj):
            return False
        return self._id == other._id

    def degrade(self):
        """Degrade the object, e.g. when impacted."""
        logger["Projectile"].info(f"{self.name} degraded.")
        self.active = False


class SimulationInterface(ABC):
    """Abstract interface for any object that can be simulated inside a planet environment.
    Subclasses must implement `update` and `accelerations`; `integrate` has a no-op default
    that subclasses are expected to override."""

    # Declared here so type checkers know concrete subclasses (via MovableObj) provide these.
    position: NDArray
    velocity: NDArray
    name: str

    def __init__(self):
        self.active: bool = True
        self.t = 0.0

    @abstractmethod
    def update(self, dt: float, command: ComputerCommand | None = None) -> list["BallisticObj"] | None:
        """Update internal state. May return a list of new BallisticObj spawned during the step
        (e.g. a separated stage or released payload)."""
        self.t += dt

    @abstractmethod
    def accelerations(self, planet: "Planet") -> NDArray:
        """Return the total acceleration vector (gravity + thrust + drag + …) in m/s²."""
        pass

    def integrate(self, dt: float, planet: "Planet") -> None:
        """Advance position and velocity by one time step using Velocity Verlet integration."""
        a0 = self.accelerations(planet)
        self.position += self.velocity * dt + 0.5 * a0 * dt**2
        a1 = self.accelerations(planet)
        self.velocity += 0.5 * (a0 + a1) * dt

    def degrade(self):
        """Degrade the object, e.g. when it reaches the kill radius. By default, just mark it as inactive."""
        self.active = False


class BallisticObj(MovableObj, SimulationInterface):
    """
    BallisticObj is a MovableObj with mass, area and drag coefficient, which can be used for projectiles and missiles.
    It does not have any guidance or propulsion, and is only affected by gravity and drag.
    Parameters:
    - position: initial position of the object in meters (m)
    - velocity: initial velocity of the object in meters per second (m/s)
    - name: name of the object (string)
    - mass: mass of the object in kg
    - area: cross-sectional area of the object in m^2
    - Cd: drag coefficient of the object (dimensionless)
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
        super().__init__(position, velocity, name)
        self._mass = mass
        self._area = area
        self.Cd = Cd

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def area(self) -> float:
        return self._area


class GuidedObj(ABC):
    """Abstract mixin for simulation objects that receive guidance commands.

    Pair with MovableObj (or BallisticObj) in the class MRO.  Concrete
    subclasses must implement `burned_fraction` and `thrust_acc`, which are
    the two properties that guidance laws and thrust computations depend on.
    """

    @property
    @abstractmethod
    def burned_fraction(self) -> float:
        """Fraction of propellant consumed, in [0, 1]."""
        ...

    @property
    @abstractmethod
    def thrust_acc(self) -> float:
        """Maximum available propulsion acceleration (m/s²)."""
        ...


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
