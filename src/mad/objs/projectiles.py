from dataclasses import dataclass, asdict
import numpy as np
from mad.objs.planets import Planet
from mad.objs.common_schemas import MovableObject, History


@dataclass
class ProjectileConfig:
    position: list[float]
    mass: float
    velocity: list[float] | None = None
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
        super().__init__(config.position, config.velocity, config.mass, config.name)
        self.area = config.area
        self.Cd = config.Cd
        self.config = config
        self.history = History(position=[config.position], velocity=[config.velocity])

    def accelerations(self, planet):
        dist = self.distance(planet)
        total_acc = np.zeros_like(self.velocity)

        if dist > planet.radius:
            gravity_acc = planet.gravity(self)
            total_acc += gravity_acc

            v_mag = np.linalg.norm(self.velocity)

            if v_mag > 0:
                rho = planet.atmosphere_rho(self)
                drag_acc = -0.5 * rho * self.Cd * self.area * v_mag * self.velocity / self.mass
                total_acc += drag_acc

        else:
            print(f"{self.name} landed on the ground!")
            self.active = False

        return total_acc

    def step(self, dt: float, planet: Planet):

        acc = self.accelerations(planet)

        self.velocity += acc * dt
        self.position += self.velocity * dt

        self.history.position.append(self.position.tolist())
        self.history.velocity.append(self.velocity.tolist())
