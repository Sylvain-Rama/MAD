import math
import numpy as np
import matplotlib.pyplot as plt
from common_schemas import MovableObject
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class Planet:
    radius: float = 6371000.0  # m
    mass: float = 5.972e24  # kg
    spin_rate: float = 7.0882359e-5  # spin rate of plannet
    G: float = 6.67408e-11  # gravitational constant, m^3/(kg*s^2)

    def __post_init__(self):
        # gravitational parameter, we ignore the mass of the secondary object
        self.mu = self.G * self.mass

    def gravity(self, obj: MovableObject) -> NDArray[np.floating]:
        r = obj.magnitude
        g = -self.mu * obj.position / r**3

        return g

    def atmosphere(self, obj: MovableObject, drag_coeff: float) -> float:
        altitude = obj.magnitude - self.radius
        # rho = math.exp(-altitude / self.atmosphere_height)
        # v = math.hypot(vel.vx, vel.vy)
        # drag_x = -drag_coeff * rho * v * vel.vx
        # drag_y = -drag_coeff * rho * v * vel.vy
        return 0.0  # drag_x, drag_y


if __name__ == "__main__":
    earth = Planet()
    pos_surface = MovableObject.from_list([0, earth.radius, 0], [0, 0, 0])
    g = earth.gravity(pos_surface)
    print(f"Gravity at surface: {g}")
