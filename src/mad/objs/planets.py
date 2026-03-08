import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib as mpl
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

        # We take the first element as this is where distance = planet.radius.
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

    def surface_distance(self, obj1: MovableObject, obj2: MovableObject) -> float:
        # Will give the linear distance (m) between 2 objects placed at the surface.

        cos_angle = np.dot(obj1.position, obj2.position) / (
            np.linalg.norm(obj1.position) * np.linalg.norm(obj2.position)
        )

        angle = np.arccos(np.clip(cos_angle, -1, 1))

        return self.radius * angle

    def create_random_point(self, altitude: float = 10, name="SurfaceObj") -> MovableObject:
        # Create a random object at the surface (+ altitude) of the planet.

        v = np.random.normal(size=self.position.shape[0])
        v /= np.linalg.norm(v)

        return MovableObject(position=(self.radius + altitude) * v, name=name)

    def create_random_point_at_distance(self, obj: MovableObject, distance: float, name="RangedObj") -> MovableObject:
        # Create a new random object at set distance from another point on the planet.

        u = obj.norm
        sigma = distance / self.radius

        # random orthogonal direction
        v = np.random.normal(size=self.position.shape[0])
        v -= np.dot(v, u) * u
        v /= np.linalg.norm(v)

        point = np.cos(sigma) * u + np.sin(sigma) * v

        return MovableObject(position=self.radius * point, name=name)

    def plot_2D_with_points(self, points: list[MovableObject] | None, ax=None) -> mpl.figure.Figure | None:
        plot_fig = False
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 4))
            plot_fig = True

        planet_body = mpl.patches.Circle(
            self.position, radius=self.radius, ec="black", fill=False, label=self.name, ls="--"
        )
        ax.add_patch(planet_body)

        if points is not None:
            for point in points:
                ax.scatter(x=point.position[0], y=point.position[1], s=50, label=point.name)

        ax.set_aspect("equal")
        ax.legend()
        ax.grid()

        return fig if plot_fig else None  # type: ignore

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
