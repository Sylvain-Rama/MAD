import numpy as np
from time import time
from collections import defaultdict
from mad.logger import SourceLogger
from mad.objs.planets import Planet
from mad.objs.base import MovableObj, SimulationInterface
from mad.utils import extract_history

logger = SourceLogger()


class Simulation:
    def __init__(self, max_time: float = 3600.0, dt: float = 1.0):
        self.max_time = max_time
        self.dt = dt

    def apply_collisions(self, objs: list[MovableObj], collisions: list[tuple[int, int]]) -> None:
        """Mark both objects in each colliding pair as inactive (in-place)."""
        for i, j in collisions:
            objs[i].active = False
            objs[j].active = False

    def run(
        self,
        moving_objs: list[SimulationInterface],
        planet: Planet,
    ) -> None:
        """Run a simple simulation of the given objects moving under the influence of the planet's gravity and atmospheric drag.
        The objects must have their initial position and velocity set. The simulation runs until max_time
        or until all objects are inactive (e.g. impacted). The results are stored in self.results.

        If collision_radius > 0, voxel-based broad-phase collision detection is run each step via
        build_voxel_grid / detect_collisions / apply_collisions."""
        active_objs = moving_objs[:]
        t = 0.0
        start = time()
        logger["Simulation"].info("Starting simulation.")
        while (t < self.max_time) and any(obj.active for obj in active_objs):
            new_objects: list[SimulationInterface] = []

            # Update all active objects and collect any new objects they spawn (e.g. Payloads from missiles).
            for obj in active_objs:
                if not obj.active:
                    continue
                spawned = obj.update(self.dt)
                if spawned:
                    for s in spawned:
                        logger["Simulation"].info(f"{s.name} added to Simulation.")
                    new_objects.extend(spawned)

            if new_objects:
                logger["Simulation"].debug(f"{len(new_objects)} new objects spawned this step.")
                active_objs.extend(new_objects)

            # Integrate all active objects' positions and velocities according to planet's gravity and drag.
            for obj in active_objs:
                if not obj.active:
                    continue
                obj.integrate(self.dt, planet)

            t += self.dt

        self.results = extract_history(active_objs, planet)
        stop = time()
        logger["Simulation"].info(f"Simulation ended at {t:.2f}s. Took {stop - start:.2f} s of real time.")


# Convenience function for quick simulations without collision detection or logging.
def run_simple_simulation(
    moving_objs: list[SimulationInterface], planet: Planet, dt: float = 0.1, max_time: float = 3600.0
) -> list[SimulationInterface]:
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
