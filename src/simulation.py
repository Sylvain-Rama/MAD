import math
import matplotlib.pyplot as plt

from objs.missiles import Missile
from objs.targets import Target


class Simulation:
    def __init__(self, planet_radius=100.0, mu=2.0e5, atmosphere_height=80.0, dt=0.005, max_steps=80000):
        self.R = planet_radius
        self.mu = mu
        self.atmosphere_height = atmosphere_height
        self.dt = dt
        self.max_steps = max_steps

        self.missiles = []
        self.target = None

    # =========================
    # Gravité et atmosphère
    # =========================
    def gravity(self, x, y):
        r = math.hypot(x, y)
        gx = -self.mu * x / (r**3)
        gy = -self.mu * y / (r**3)
        return gx, gy

    def atmosphere(self, x, y, vx, vy, drag_coeff):
        r = math.hypot(x, y)
        altitude = max(0.0, r - self.R)
        rho = math.exp(-altitude / self.atmosphere_height)
        v = math.hypot(vx, vy)
        drag_x = -drag_coeff * rho * v * vx
        drag_y = -drag_coeff * rho * v * vy
        return drag_x, drag_y

    # =========================
    # Définir la cible
    # =========================
    def set_target(self, surface_angle_deg):
        self.target = Target(surface_angle_deg, self.R)

    # =========================
    # Ajouter un missile
    # =========================
    def add_missile(
        self,
        surface_angle_deg,
        thrust,
        burn_time,
        drag_coeff,
        target_altitude,
        cruise_thrust_ratio=0.6,
        cruise_duration=2.0,
        name="Missile",
    ):
        if self.target is None:
            raise ValueError("Définir une cible avant d'ajouter un missile.")

        a = math.radians(surface_angle_deg)
        x = self.R * math.cos(a)
        y = self.R * math.sin(a)

        missile = Missile(
            position=(x, y),
            thrust=thrust,
            burn_time=burn_time,
            drag_coeff=drag_coeff,
            target_altitude=target_altitude,
            target=self.target,
            cruise_thrust_ratio=cruise_thrust_ratio,
            cruise_duration=cruise_duration,
            terminal_guidance=True,
            name=name,
        )

        self.missiles.append(missile)

    # =========================
    # Boucle de simulation
    # =========================
    def run(self):
        for _ in range(self.max_steps):
            any_alive = False
            for missile in self.missiles:
                if not missile.alive:
                    continue
                missile.step(self.dt, self.gravity, self.atmosphere, self.R)
                if math.hypot(missile.x, missile.y) <= self.R:
                    missile.alive = False
                else:
                    any_alive = True
            if not any_alive:
                break

    # =========================
    # Affichage
    # =========================
    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        # Planète
        planet = plt.Circle((0, 0), self.R, fill=False)
        ax.add_artist(planet)

        # Cible
        if self.target is not None:
            ax.scatter(self.target.x, self.target.y, marker="X", s=100, c="red", label="Cible")

        # Missiles
        for missile in self.missiles:
            ax.plot(missile.xs, missile.ys, label=missile.name)
            ax.scatter(missile.xs[0], missile.ys[0])

        ax.set_aspect("equal")
        ax.set_xlim(-260, 260)
        ax.set_ylim(-260, 260)
        ax.set_title("Simulation missiles balistiques avec guidage vers la cible")
        ax.grid(True)
        ax.legend()
        plt.show()
