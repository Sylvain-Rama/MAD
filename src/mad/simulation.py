import numpy as np
from time import time
from collections import defaultdict
from mad.logger import SourceLogger
from mad.objs.planets import Planet
from mad.objs.common_schemas import BallisticObj, GuidedObj
from mad.utils import extract_history

logger = SourceLogger()


class Simulation:
    def __init__(self, voxel_size: float = 50_000.0, max_time: float = 3600.0, dt: float = 1.0):
        self.voxel_size = voxel_size
        self.max_time = max_time
        self.dt = dt

        self.neighbours = (-1, 0, 1)

    def build_voxel_grid(self, objs: list[BallisticObj]) -> dict[tuple[int, ...], list[int]]:
        """Partition active objects into a voxel grid and return a mapping of voxel key → list of object indices.

        Each object is assigned to the voxel whose integer key is floor(position / voxel_size).
        Only active objects are inserted. Objects without a `position` attribute are skipped.

        Parameters
        ----------
        objs:       list of simulation objects (only active ones are indexed)
        """
        grid: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for idx, obj in enumerate(objs):
            if not obj.active:
                continue
            pos = getattr(obj, "position", None)
            if pos is None:
                continue
            key = tuple(int(np.floor(c / self.voxel_size)) for c in pos)
            grid[key].append(idx)
        return grid

    def _neighbour_keys(self, key: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Return the 26 face/edge/corner neighbours of a 3-D voxel key plus the key itself."""
        x, y, z = key
        return [(x + dx, y + dy, z + dz) for dx in self.neighbours for dy in self.neighbours for dz in self.neighbours]

    def detect_collisions(
        self,
        objs: list[BallisticObj],
        grid: dict[tuple[int, ...], list[int]],
        collision_radius: float,
    ) -> list[tuple[int, int]]:
        """Return all colliding pairs using the pre-built voxel grid.

        For every occupied voxel the function checks the object against objects in
        the same voxel and its 26 neighbours, performing an exact distance test only
        for those candidates. Each pair is reported at most once.

        Parameters
        ----------
        objs:             list of simulation objects (same list passed to build_voxel_grid)
        grid:             voxel grid returned by build_voxel_grid
        collision_radius: two objects are considered colliding when their centres are
                        within this distance (metres)

        Returns
        -------
        List of (i, j) index pairs with i < j for every detected collision.
        """
        r2 = collision_radius**2
        seen: set[tuple[int, int]] = set()
        collisions: list[tuple[int, int]] = []

        for key, bucket in grid.items():
            candidates: list[int] = []
            for nkey in self._neighbour_keys(key):
                candidates.extend(grid.get(nkey, []))

            for i in bucket:
                pos_i = objs[i].position
                for j in candidates:
                    if j <= i:
                        continue
                    pair = (i, j)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    delta = objs[j].position - pos_i
                    if float(np.dot(delta, delta)) <= r2:
                        collisions.append(pair)

        return collisions

    def apply_collisions(self, objs: list[BallisticObj], collisions: list[tuple[int, int]]) -> None:
        """Mark both objects in each colliding pair as inactive (in-place)."""
        for i, j in collisions:
            objs[i].active = False
            objs[j].active = False

    def run(
        self,
        moving_objs: list[BallisticObj],
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
            new_objects: list[BallisticObj] = []

            for obj in active_objs[:]:
                if not obj.active:
                    continue
                spawned = obj.update(self.dt)
                if spawned:
                    for s in spawned:
                        logger["Simulation"].info(f"{s.name} added to Simulation.")
                    new_objects.extend(spawned)
                if not obj.active:
                    continue
                obj.integrate(self.dt, planet)

            active_objs.extend(new_objects)

            t += self.dt

        self.results = extract_history(active_objs, planet)
        stop = time()
        logger["Simulation"].info(f"Simulation ended at {t:.2f}s. Took {stop - start:.2f} s of real time.")


# Convenience function for quick simulations without collision detection or logging.
def run_simple_simulation(
    moving_objs: list[BallisticObj], planet: Planet, dt: float = 0.1, max_time: float = 3600.0
) -> list[BallisticObj]:
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
