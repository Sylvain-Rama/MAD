from mad.logger import SourceLogger
from mad.objs.planets import Planet
from mad.objs.missiles import BallisticMissile
from mad.objs.projectiles import Projectile

logger = SourceLogger()


def run_simple_simulation(
    moving_objs: list[BallisticMissile | Projectile], planet: Planet, dt: float = 0.1, max_time: float = 3600.0
) -> list[BallisticMissile | Projectile]:
    """Run a simple simulation of the given objects moving under the influence of the planet's gravity and atmospheric drag.
    The objects must have their initial position and velocity set. The simulation runs until max_time
    or until all objects are inactive (e.g. impacted or ran out of propellant). Returns the list
    of objects with their final states after the simulation."""
    active_objs = moving_objs[:]
    t = 0.0
    while (t < max_time) and any(obj.active for obj in active_objs):

        for obj in active_objs[:]:
            if not obj.active:
                continue
            _ = obj.update(dt)
            obj.integrate(dt, planet)

        t += dt

    return active_objs
