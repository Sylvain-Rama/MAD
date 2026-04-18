import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.patches
from mad.objs.common_schemas import MovableObj, DraggableObj
from mad.configs.physics import G
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


class Planet(MovableObj):
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
        return np.sqrt(2 * self.mu / self.radius)

    @property
    def orbital_velocity(self):
        return np.sqrt(self.mu / self.radius)

    @property
    def gravity_at_surface(self):
        surface_pos = np.zeros_like(self.position)
        surface_pos[0] = self.radius
        surf_obj = MovableObj(position=surface_pos)

        # We take the first element as this is where distance = planet.radius.
        return self.gravity(surf_obj)[0]

    def __repr__(self):
        return (
            f"Planet {self.name} at {self.position}\n"
            f"Mass {self.mass:.2e} kg, Radius {self.radius / 1000} km.\n"
            f"Gravity at surface: {self.gravity_at_surface:.2f} m/s^2\n"
            f"Orbital velocity: {self.orbital_velocity:.2f} m/s\n"
            f"Escape velocity: {self.escape_velocity:.2f} m/s"
        )

    def drag(self, obj: DraggableObj) -> NDArray:
        drag = np.zeros_like(obj.velocity)
        alt = max(0.0, float(np.linalg.norm(obj.position - self.position)) - self.radius)

        if alt > 0:
            rho = self.rho0 * np.exp(-alt / self.atmosphere_height)
            v_mag = np.linalg.norm(obj.velocity)
            if v_mag > 0:
                drag = -0.5 * rho * obj.Cd * obj.area * v_mag * obj.velocity / obj.mass

        return drag

    def gravity(self, other: MovableObj) -> NDArray:
        r_vec = other.position - self.position
        dist = np.linalg.norm(r_vec)
        if dist < 1e-6:
            return np.zeros_like(other.position)

        return -self.mu * r_vec / dist**3

    def surface_distance(self, obj1: MovableObj, obj2: MovableObj) -> float:
        # Will give the linear distance (m) between 2 objects placed at the surface.

        cos_angle = np.dot(obj1.position, obj2.position) / (
            np.linalg.norm(obj1.position) * np.linalg.norm(obj2.position)
        )

        angle = np.arccos(np.clip(cos_angle, -1, 1))

        return self.radius * angle

    def create_2D_point(self, altitude: float = 10, name="SurfaceObj") -> MovableObj:
        # Create a random object at the 2D surface (+ altitude) of the planet.

        v = np.random.normal(size=2)
        v /= np.linalg.norm(v)

        return MovableObj(position=(self.radius + altitude) * v, name=name)

    def create_2D_point_at_distance(self, obj: MovableObj, distance_km: float, name="RangedObj") -> MovableObj:
        # Create a new random object at set distance from another point on the planet.2D mode for easy plot.

        u = obj.normalize[:2]
        sigma = (distance_km * 1000) / self.radius

        # random orthogonal direction
        v = np.random.normal(size=2)
        v -= np.dot(v, u) * u
        v /= np.linalg.norm(v)

        point = np.cos(sigma) * u + np.sin(sigma) * v

        return MovableObj(position=self.radius * point, name=name)

    def plot_2D_with_points(
        self, points: list[MovableObj] | None, ax=None, display_planet=True
    ) -> matplotlib.figure.Figure | None:
        # 2D plot of the planet. If using point in 2D, they will appear at the circumference.
        plot_fig = False
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 4))
            plot_fig = True

        if display_planet:
            if points is not None and len(points) == 2:
                theta1 = np.degrees(
                    np.arctan2(points[0].position[1] - self.position[1], points[0].position[0] - self.position[0])
                )
                theta2 = np.degrees(
                    np.arctan2(points[1].position[1] - self.position[1], points[1].position[0] - self.position[0])
                )
                planet_body = matplotlib.patches.Arc(
                    (float(self.position[0]), float(self.position[1])),
                    2 * self.radius,
                    2 * self.radius,
                    angle=0,
                    theta1=min(theta1, theta2),
                    theta2=max(theta1, theta2),
                    ec="black",
                    label=self.name,
                    ls="--",
                )
            else:
                planet_body = matplotlib.patches.Circle(
                    (float(self.position[0]), float(self.position[1])),
                    radius=self.radius,
                    ec="black",
                    fill=False,
                    label=self.name,
                    ls="--",
                )
            ax.add_patch(planet_body)

        if points is not None:
            for point in points:
                ax.scatter(x=point.position[0], y=point.position[1], s=50, label=point.name)

        ax.set_aspect("equal")
        ax.legend()
        ax.grid()

        return fig if plot_fig else None  # type: ignore


class SimulationInterface(ABC):
    """SimulationInterface is an abstract class that defines the interface for any object that can be simulated in the planet environment.
    It requires the implementation of the update, accelerations and integrate methods, which are used to update the object's state,
    compute the accelerations and integrate the equations of motion, respectively. It is used to ensure that all objects that can be
    simulated in the planet environment have a consistent interface and can be easily integrated into the simulation loop.
    """

    @abstractmethod
    def update(self, dt: float) -> list[MovableObj] | None:
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
