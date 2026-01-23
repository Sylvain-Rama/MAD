import math


class Target:
    def __init__(self, surface_angle_deg, planet_radius):
        a = math.radians(surface_angle_deg)
        self.angle = a
        self.x = planet_radius * math.cos(a)
        self.y = planet_radius * math.sin(a)
