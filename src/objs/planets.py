import math
from common_schemas import Position, Velocity


class Planet:
    def __init__(self, radius: float, mu: float, atmosphere_height: float, drag_coeff: float):
        self.radius = radius
        self.mu = mu
        self.atmosphere_height = atmosphere_height
        self.drag_coeff = drag_coeff

        if any([arg <= 0 for arg in [radius, mu, atmosphere_height, drag_coeff]]):
            raise ValueError("None of the arguments should be <= 0.")

    def gravity(self, pos: Position) -> tuple[float, float]:
        r = pos.distance_to_core()
        gx = -self.mu * pos.x / (r**3)
        gy = -self.mu * pos.y / (r**3)
        return gx, gy

    def atmosphere(self, pos: Position, vel: Velocity, drag_coeff: float) -> tuple[float, float]:
        altitude = pos.altitude(self.radius)
        rho = math.exp(-altitude / self.atmosphere_height)
        v = math.hypot(vel.vx, vel.vy)
        drag_x = -drag_coeff * rho * v * vel.vx
        drag_y = -drag_coeff * rho * v * vel.vy
        return drag_x, drag_y
