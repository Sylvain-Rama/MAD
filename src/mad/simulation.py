from __future__ import annotations

from time import time
from mad.logger import SourceLogger
from mad.objs.planets import Planet
from mad.objs.base import MovableObj, SimulationInterface
from mad.utils import extract_history

logger = SourceLogger()


class HistoryCollector:
    """Records object state at each simulation step at the loop level.

    Parameters
    ----------
    fields:
        Names of fields to capture each step.  Supported values:

        - ``"t"``        – object time (float, from ``obj.t``)
        - ``"position"`` – 3-D position vector stored as a list ``[x, y, z]``
        - ``"velocity"`` – 3-D velocity vector stored as a list ``[x, y, z]``
        - ``"gamma"``    – guidance pitch angle from ``obj.guidance_results.gamma``
                           (``None`` when the object has no active guidance)

    Results are stored per object, keyed by ``obj._id``.  Each value is a dict
    of column-name → list, directly usable as a ``pd.DataFrame`` constructor
    argument::

        collector = HistoryCollector(["t", "position", "velocity", "gamma"])
        sim.run(objs, planet, collector=collector)
        df = pd.DataFrame(collector.to_dict()["Payload_RV1_0"])
    """

    def __init__(self, fields: list[str]) -> None:
        self.fields = list(fields)
        self._data: dict[str, dict[str, list]] = {}

    def _columns_for(self, field: str) -> list[str]:
        return [field]

    def record(self, objs: list) -> None:
        """Capture the current state of all active objects in *objs*."""
        for obj in objs:
            if not obj.active:
                continue
            key = obj._id
            if key not in self._data:
                self._data[key] = {col: [] for f in self.fields for col in self._columns_for(f)}
            entry = self._data[key]
            for f in self.fields:
                if f == "t":
                    entry["t"].append(getattr(obj, "t", 0.0))
                elif f == "position":
                    entry["position"].append(obj.position.tolist())
                elif f == "velocity":
                    entry["velocity"].append(obj.velocity.tolist())
                elif f == "gamma":
                    gr = getattr(obj, "guidance_results", None)
                    entry["gamma"].append(gr.gamma if gr is not None else None)

    def to_dict(self) -> dict[str, dict[str, list]]:
        """Return collected history as ``{obj._id: {column: [values, ...]}}``."""
        return self._data


class Simulation:
    def __init__(self, max_time: float = 3600.0, dt: float = 1.0):
        self.max_time = max_time
        self.dt = dt
        self.collector = HistoryCollector(["t", "position", "velocity", "gamma"])

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
        build_voxel_grid / detect_collisions / apply_collisions.

        If a *collector* is provided, it records the requested fields for every active object after
        each integration step (including the initial state before the first step)."""
        active_objs = moving_objs[:]
        t = 0.0
        start = time()
        logger["Simulation"].info("Starting simulation.")

        self.collector.record(active_objs)

        while (t < self.max_time) and any(obj.active for obj in active_objs):
            new_objects: list[SimulationInterface] = []

            # Update all active objects and collect any new objects they spawn (e.g. Payloads from missiles).
            for obj in active_objs:
                if not obj.active:
                    continue
                spawned = obj.update(self.dt)
                if spawned:
                    new_objects.extend(spawned)

            if new_objects:
                logger["Simulation"].info(f"New objects spawned this step: {[obj.name for obj in new_objects]}")
                active_objs.extend(new_objects)

            # Integrate all active objects' positions and velocities according to planet's gravity and drag.
            for obj in active_objs:
                if not obj.active:
                    continue
                obj.integrate(self.dt, planet)

            self.collector.record(active_objs)

            t += self.dt

        self.results = extract_history(active_objs, planet)
        stop = time()
        logger["Simulation"].info(f"Simulation ended at {t:.2f}s. Took {stop - start:.2f} s of real time.")


# Convenience function for quick simulations without collision detection or logging.
def run_simple_simulation(
    moving_objs: list[SimulationInterface], planet: Planet, dt: float = 0.1, max_time: float = 3600.0
) -> dict[str, dict[str, list]]:
    """Run a simple simulation of the given objects moving under the influence of the planet's gravity and atmospheric drag.
    The objects must have their initial position and velocity set. The simulation runs until max_time
    or until all objects are inactive (e.g. impacted or ran out of propellant). Returns the collected history as a dictionary.
    """
    active_objs = moving_objs[:]
    t = 0.0
    collector = HistoryCollector(["t", "position", "velocity", "gamma"])
    collector.record(active_objs)
    while (t < max_time) and any(obj.active for obj in active_objs):

        for obj in active_objs[:]:
            if not obj.active:
                continue
            _ = obj.update(dt)
            obj.integrate(dt, planet)

        collector.record(active_objs)
        t += dt

    return collector.to_dict()
