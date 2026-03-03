import numpy as np
from numpy.typing import NDArray
from mad.objs.common_schemas import MovableObject
from mad.objs.constants import G
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod


@dataclass
class PlanetConfig:
    position: list[float]
    radius: float
    mass: float
    spin_rate: float
    velocity: list[float] | None = None
    name: str = "Planet"
    rho0: float = 1.225
    atmosphere_height: float = 8000.0

    @property
    def to_dict(self):
        return asdict(self)


class Planet(MovableObject):
    def __init__(self, config: PlanetConfig):
        super().__init__(config.position, config.velocity, config.name)
        self.mass = config.mass
        self.radius = config.radius
        self.spin_rate = config.spin_rate
        self.atmosphere_height = config.atmosphere_height
        self.rho0 = config.rho0
        self.mu = self.mass * G

    @property
    def escape_velocity(self):
        return np.sqrt(2 * G * self.mass / self.radius)

    @property
    def gravity_at_surface(self):
        surface_pos = np.zeros_like(self.position)
        surface_pos[0] = self.radius
        surf_obj = MovableObject(position=list(surface_pos))

        return self.gravity(surf_obj)[0]

    def drag(self, obj: MovableObject) -> NDArray:
        drag = np.zeros_like(obj.velocity)
        alt = max(0.0, self.distance(obj) - self.radius)  # type: ignore[no-matching-overload]

        if alt > 0:
            rho = self.rho0 * np.exp(-alt / self.atmosphere_height)
            v_mag = np.linalg.norm(obj.velocity)
            if v_mag > 0:
                drag = -0.5 * rho * obj.Cd * obj.area * v_mag * obj.velocity / obj.mass

        return drag

    def gravity(self, other: MovableObject) -> NDArray:
        r_vec = other.position - self.position
        dist = np.linalg.norm(r_vec)
        if dist < 1e-6:
            return np.zeros_like(other.position)

        return -self.mu * r_vec / dist**3

    def __repr__(self):
        return f"Planet {self.name} at {self.position}, mass {self.mass}, radius {self.radius}."


class SimulationInterface(ABC):

    @abstractmethod
    def update(self, dt: float) -> MovableObject | None:
        """This abstract method is dedicated to the update of the object itself.
        It can return other Movable objects to be able to spawn elements in the simulation."""
        pass

    @abstractmethod
    def accelerations(self, planet: Planet) -> NDArray:
        """Abstract method dedicated to the computation of accelerations: gravity, thrust, drag, etc..."""
        pass

    def integrate(self, dt: float, planet: Planet) -> None:
        """This abstract method is dedicated to the update of the object position / velocity according to the selected planet."""
        pass

    def step(self, dt: float, planet: Planet) -> MovableObject | None:
        """Convenience method if we want to run the object quickly through simulation."""
        obj = self.update(dt)
        self.integrate(dt, planet)

        return obj if obj else None
