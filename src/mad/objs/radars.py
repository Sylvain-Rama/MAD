import numpy as np
import itertools
from mad.objs.base import MovableObj
from mad.objs.planets import Planet
from mad.utils import to_voxel_key
from mad.configs.physics import VOXEL_SIZE

from dataclasses import dataclass, asdict


@dataclass
class RadarConfig:
    position: list[float]
    name: str = "Radar"
    range: float = 450_000.0  # m
    voxel_size: float = VOXEL_SIZE  # m, for determining which voxels to check for detections

    @property
    def to_dict(self):
        return asdict(self)


class Radar(MovableObj):
    def __init__(self, config: RadarConfig, planet: Planet):
        super().__init__(config.position, velocity=None, name=config.name)
        self.range = config.range
        self.voxel_size = config.voxel_size
        self.planet = planet

        self.detection_voxels = self.get_detection_voxels()

    def get_detection_voxels(self) -> set[tuple[int, ...]]:
        range_voxels = int(np.ceil(self.range / self.voxel_size))

        radar_key = np.array(to_voxel_key(self.position, voxel_size=self.voxel_size))

        offsets_1d = np.arange(-range_voxels, range_voxels + 1)
        offsets = np.array(list(itertools.product(offsets_1d, repeat=3)))  # (N, 3)
        candidate_keys = radar_key + offsets  # (N, 3)

        # World-space centre of each candidate voxel
        voxel_centers = (candidate_keys + 0.5) * self.voxel_size  # (N, 3)

        # Voxel centre within radar range & above planet surface
        dist_from_radar = np.linalg.norm(voxel_centers - self.position, axis=1)
        within_range = dist_from_radar <= self.range

        dist_from_planet = np.linalg.norm(voxel_centers - self.planet.position, axis=1)
        above_surface = dist_from_planet >= self.planet.radius

        valid_keys = candidate_keys[within_range & above_surface]
        return {tuple(key) for key in valid_keys}

    def detect(self, obj: MovableObj) -> bool:
        obj_voxel = to_voxel_key(obj.position, voxel_size=self.voxel_size)
        return obj_voxel in self.detection_voxels
