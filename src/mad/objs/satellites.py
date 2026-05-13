from mad.objs import BallisticObj, ProjectileConfig
from mad.logger import SourceLogger
import numpy as np
from numpy.typing import NDArray

logger = SourceLogger()


class Sputnik(BallisticObj):
    def __init__(self, config: ProjectileConfig, t: float = 0.0):
        super().__init__(config.position, config.velocity, config.name, config.mass, config.area, config.Cd)
        self.config = config
        self.t = t

    def accelerations(self, planet) -> NDArray:

        if self.distance(planet) <= planet.radius:
            logger["Projectile"].info(f"{self.name} landed on the ground!")
            self.active = False
            return np.zeros_like(self.velocity)

        gravity_acc = planet.gravity(self)
        
        return gravity_acc

    def update(self, dt: float):
        self.t += dt
        # Sputniks are indestructible and unaffected by drag, we only need to beep from time to time.
        if self.t // 60 == 0:  
            logger["Projectile"].info(f"{self.name} -- Beep Beep!")

        return None