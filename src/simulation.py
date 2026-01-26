import math
import matplotlib.pyplot as plt

from objs.missiles import Missile
from objs.targets import Target
from objs.planets import Planet


class Simulation:
    def __init__(self, dt=0.005, max_steps=8000):

        self.dt = dt
        self.max_steps = max_steps

        self.missiles = []
        self.targets = []
        self.planet: Planet | None = None

    def add_target(self, angle: float = 120):
        if not self.planet:
            raise ValueError("Please add a planet first.")
        self.targets.append(Target(surface_angle_deg=angle, planet_radius=self.planet.radius))

    def add_planet(self, planet: Planet):
        self.planet = planet

    def add_missile(self, missile: Missile):
        self.missiles.append(missile)

    def run(self):
        for _ in range(self.max_steps):
            for missile in [missile for missile in self.missiles if missile.alive]:
                if not missile.alive:
                    continue
                missile.step(self.dt, self.planet)

    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 6))

        planet = plt.Circle((0, 0), self.planet.radius, fill=False)
        ax.add_artist(planet)

        for target in self.targets:
            ax.scatter(target.x, target.y, marker="X", s=100, c="red", label="Target")

        max_x = min_x = max_y = min_y = 0
        for missile in self.missiles:
            xs = [item[0] for item in missile.history]
            ys = [item[1] for item in missile.history]
            ax.plot(xs, ys, label=missile.config.name)
            max_x = max(max_x, max(xs), self.planet.radius)
            min_x = min(min_x, min(xs), -self.planet.radius)

            max_y = max(max_y, max(ys), self.planet.radius)
            min_y = min(min_y, min(ys), -self.planet.radius)

            ax.scatter(xs[0], ys[0])

        ax.set_aspect("equal")
        padding = 20
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)

        ax.grid(True)
        ax.legend()
        plt.show()
