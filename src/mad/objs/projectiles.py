from dataclasses import dataclass, asdict
import numpy as np
from mad.objs.planets import Planet
from mad.objs.common_schemas import MovableObject, History


@dataclass
class ProjectileConfig:
    position: list[float]  # m
    mass: float  # kg
    velocity: list[float] | None = None  # m / s
    name: str = "Projectile"
    area: float = 0.01  # m^2
    Cd: float = 0.47  # sphere

    @property
    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        if not self.velocity:
            self.velocity = [0.0] * len(self.position)


class Projectile(MovableObject):
    def __init__(self, config: ProjectileConfig):
        super().__init__(config.position, config.velocity, config.name)
        self.mass = config.mass
        self.area = config.area
        self.Cd = config.Cd
        self.config = config
        self.history = History(position=[config.position], velocity=[config.velocity])

    def accelerations(self, planet):
        total_acc = np.zeros_like(self.velocity)

        gravity_acc = planet.gravity(self)
        total_acc += gravity_acc

        v_mag = np.linalg.norm(self.velocity)

        if v_mag > 0:
            rho = planet.atmosphere_rho(self)
            drag_acc = -0.5 * rho * self.Cd * self.area * v_mag * self.velocity / self.mass
            total_acc += drag_acc

        return total_acc

    def step(self, dt: float, planet: Planet):

        if self.distance(planet) <= planet.radius:
            print(f"{self.name} landed on the ground!")
            self.active = False

        else:
            acc = self.accelerations(planet)

            self.velocity += acc * dt
            self.position += self.velocity * dt

            # Velocity Verlet for solver.
            a0 = self.accelerations(planet)
            self.position += self.velocity * dt + 0.5 * a0 * dt**2
            a1 = self.accelerations(planet)

            self.velocity += 0.5 * (a0 + a1) * dt

            self.history.position.append(self.position.tolist())
            self.history.velocity.append(self.velocity.tolist())
