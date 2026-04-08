from mad.logger import SourceLogger
from mad.objs.planets import Planet
from mad.objs.missiles import BallisticMissile
from mad.objs.projectiles import Projectile

logger = SourceLogger()


def run_simulation(
    moving_objs: list[BallisticMissile | Projectile], planet: Planet, dt: float = 0.1, max_time: float = 3600.0
) -> list[BallisticMissile | Projectile]:
    active_objs = moving_objs[:]
    for t in range(int(max_time / dt)):
        new_objects = []

        for obj in active_objs[:]:

            if not obj.active:
                continue

            sim_update = obj.update(dt)

            if sim_update is not None:
                new_objects.append(sim_update)
                logger["Simulation"].info(f"{sim_update.name} added to Simulation.")

            if not obj.active:
                continue
            obj.integrate(dt, planet)

        active_objs.extend(new_objects)

    return active_objs
