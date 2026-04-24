from mad.objs.base import MovableObj
import numpy as np
from collections import defaultdict
from mad.logger import SourceLogger
from mad.configs.physics import VOXEL_SIZE

logger = SourceLogger()


class CollisionDetector:
    """A simple collision detector based on a voxel grid.
    The grid partitions space into cubes of size `voxel_size`. Each active object is assigned to a voxel based on its position.
    Collisions are detected by checking each object against others in the same voxel and its 26 neighbours, using an exact distance test.
    """

    def __init__(self, voxel_size: float = VOXEL_SIZE):
        self.voxel_size = voxel_size
        self.neighbours = (-1, 0, 1)

    def build_voxel_grid(self, objs: list[MovableObj]) -> dict[tuple[int, ...], list[int]]:
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
            key = tuple(np.floor(obj.position / self.voxel_size, dtype=int))
            grid[key].append(idx)
        return grid

    def _neighbour_keys(self, key: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Return the 26 face/edge/corner neighbours of a 3-D voxel key plus the key itself."""
        x, y, z = key
        return [(x + dx, y + dy, z + dz) for dx in self.neighbours for dy in self.neighbours for dz in self.neighbours]

    def detect_collisions(
        self,
        objs: list[MovableObj],
        grid: dict[tuple[int, ...], list[int]],
        collision_radius: float = 500.0,
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
