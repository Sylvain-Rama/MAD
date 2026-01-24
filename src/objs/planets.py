import math


class Planet:
    def __init__(self, radius: float, mu: float, atmosphere_height: float):
        self.radius = radius
        self.mu = mu
        self.atmosphere_height = atmosphere_height

    @staticmethod
    def distance_to_core(x, y):
        return math.hypot(x, y)

    def altitude(self, x, y):
        r = self.distance_to_core(x, y)
        return max(0.0, r - self.radius)

    def gravity(self, x: float, y: float):
        r = self.distance_to_core(x, y)
        gx = -self.mu * x / (r**3)
        gy = -self.mu * y / (r**3)
        return gx, gy

    def atmosphere(self, x, y, vx, vy, drag_coeff):
        altitude = self.altitude(x, y)
        rho = math.exp(-altitude / self.atmosphere_height)
        v = math.hypot(vx, vy)
        drag_x = -drag_coeff * rho * v * vx
        drag_y = -drag_coeff * rho * v * vy
        return drag_x, drag_y
