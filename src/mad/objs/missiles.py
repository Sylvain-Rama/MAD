from mad.objs.projectiles import Projectile

import numpy as np


class MissileStage:
    def __init__(
        self,
        dry_mass: float,
        propellant_mass: float,
        thrust: float,  # N
        burn_time: float,  # s
        area: float,
        Cd: float,
        payload=None,  # another stage or None
    ):
        self.dry_mass = dry_mass
        self.propellant_mass = propellant_mass
        self.initial_propellant = propellant_mass

        self.thrust = thrust
        self.burn_time = burn_time
        self.burn_rate = propellant_mass / burn_time if burn_time > 0 else 0.0

        self.area = area
        self.Cd = Cd

        self.payload = payload
        self.active = True
        self.time_burning = 0.0

        @property
        def mass(self):
            payload_mass = self.payload.mass if self.payload else 0.0
            return self.dry_mass + self.propellant_mass + payload_mass

        def update(self, dt: float):
            if not self.active:
                return

            if self.propellant_mass > 0:
                dm = self.burn_rate * dt
                self.propellant_mass = max(0.0, self.propellant_mass - dm)
                self.time_burning += dt
            else:
                self.active = False

        def thrust_force(self, direction: np.ndarray) -> np.ndarray:
            if self.propellant_mass > 0:
                return self.thrust * direction
            return np.zeros_like(direction)
