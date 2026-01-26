import math
import numpy as np
import matplotlib.pyplot as plt
from objs.common_schemas import Position, Velocity


class Planet:
    def __init__(self, radius: float, mu: float, atmosphere_height: float):
        self.radius = radius
        self.mu = mu
        self.atmosphere_height = atmosphere_height

        if any([arg <= 0 for arg in [radius, mu, atmosphere_height]]):
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

    def show_characteristics(self, grid_n=100):

        limit = (self.radius + self.atmosphere_height) * 2 + 10
        xs = np.linspace(-limit, limit, grid_n)
        ys = np.linspace(-limit, limit, grid_n)

        grid_x, grid_y = np.meshgrid(xs, ys)

        gravity_vals = []
        atm_vals = []
        for posx, posy in zip(np.ravel(grid_x), np.ravel(grid_y)):
            pos = Position(float(posx), float(posy))
            vel = Velocity(1, 1)
            gravity_vals.append(self.gravity(pos))
            atm_vals.append(self.atmosphere(pos, vel, drag_coeff=0.02))

        gravity_vals = np.abs(np.reshape(gravity_vals, (grid_n, grid_n, 2)))
        atm_vals = np.abs(np.reshape(atm_vals, (grid_n, grid_n, 2)))

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
        ax = ax.flatten()

        ax[0].imshow(np.sum(gravity_vals, axis=2), cmap="inferno")
        ax[1].imshow(np.sum(atm_vals, axis=2), cmap="inferno")

        planet = plt.Circle((grid_n // 2, grid_n // 2), self.radius, fill=False, color="r")
        ax[0].add_artist(planet)
        planet = plt.Circle((grid_n // 2, grid_n // 2), self.radius, fill=False, color="r")
        ax[1].add_artist(planet)

        return fig


if __name__ == "__main__":
    planet = Planet(radius=20, mu=200, atmosphere_height=5)

    fig = planet.show_characteristics()

    plt.show()
