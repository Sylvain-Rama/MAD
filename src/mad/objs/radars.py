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
        above_surface = dist_from_planet >= self.planet.radius

        valid_mask = within_range & above_surface
        valid_keys = candidate_keys[valid_mask]
        offsets = valid_keys - radar_key
        strengths = 1 - np.max(np.abs(offsets), axis=1) / self.max_value
        return {tuple(key): float(strength) for key, strength in zip(valid_keys, strengths)}

    def get_detection_strength(self, obj: MovableObj) -> float:
        """Returns a value between 0 and 1 representing the strength of the radar detection for the given object, based on its distance from the radar and its radar cross section (area * Cd)."""
        obj_voxel = to_voxel_key(obj.position, voxel_size=self.voxel_size)
        return self.detection_voxels.get(tuple(obj_voxel), 0.0)

    def detect(self, obj: MovableObj) -> bool:
        obj_voxel = to_voxel_key(obj.position, voxel_size=self.voxel_size)
        return tuple(obj_voxel) in self.detection_voxels
