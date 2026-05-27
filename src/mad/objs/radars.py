import numpy as np
import itertools
from mad.objs import MovableObj
from mad.objs import Planet
from mad.utils.utils import to_voxel_key
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

    def get_detection_voxels(self) -> dict[tuple[int, ...], float]:
        range_voxels = int(np.ceil(self.range / self.voxel_size))

        self.radar_key = np.array(to_voxel_key(self.position, voxel_size=self.voxel_size))
        radar_key = self.radar_key

        offsets_1d = np.arange(-range_voxels, range_voxels + 1)
        self.max_value = np.max(np.abs(offsets_1d))
        all_offsets = np.array(list(itertools.product(offsets_1d, repeat=3)))  # (N, 3)
        candidate_keys = radar_key + all_offsets  # (N, 3)

        # World-space centre of each candidate voxel
        voxel_centers = (candidate_keys + 0.5) * self.voxel_size  # (N, 3)

        # Voxel centre within radar range & above planet surface
        dist_from_radar = np.linalg.norm(voxel_centers - self.position, axis=1)
        within_range = dist_from_radar <= self.range

        dist_from_planet = np.linalg.norm(voxel_centers - self.planet.position, axis=1)
        # Check if voxel is above planet surface, accounting for voxel diagonal
        # (worst case where radar is directly above the corner of the voxel)
        above_surface = dist_from_planet >= self.planet.radius - np.sqrt(3) * self.voxel_size / 2

        valid_mask = within_range & above_surface
        valid_keys = candidate_keys[valid_mask]
        valid_distances = dist_from_radar[valid_mask]
        strengths = 1 - valid_distances / self.range
        return {tuple(key): float(strength) for key, strength in zip(valid_keys, strengths)}

    def get_detection_strength(self, obj: MovableObj) -> float:
        obj_voxel = to_voxel_key(obj.position, voxel_size=self.voxel_size)
        return self.detection_voxels.get(tuple(obj_voxel), 0.0)

    def detect(self, obj: MovableObj) -> bool:
        obj_voxel = to_voxel_key(obj.position, voxel_size=self.voxel_size)
        return tuple(obj_voxel) in self.detection_voxels
