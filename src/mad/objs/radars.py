import numpy as np
import itertools
from mad.objs.base import MovableObj
from mad.objs.planets import Planet
from mad.utils import to_voxel_key
from mad.configs.physics import VOXEL_SIZE_KM

from dataclasses import dataclass, asdict


@dataclass
class RadarConfig:
    position: list[float]
    name: str = "Radar"
    range_km: float = 450.0  # km
    altitude_km: float = 0.0  # km, for line-of-sight calculations
    voxel_size_km: float = VOXEL_SIZE_KM  # km, for determining which voxels to check for detections

    @property
    def to_dict(self):
        return asdict(self)


class Radar(MovableObj):
    def __init__(self, config: RadarConfig, planet: Planet):
        super().__init__(config.position, velocity=None, name=config.name)
        self.range_km = config.range_km
        self.altitude_km = config.altitude_km
        self.voxel_size_km = config.voxel_size_km
        self.planet = planet

        self.detection_voxels = self.get_detection_voxels()

    def get_detection_voxels(self) -> list[tuple[int, ...]]:
        # Return the voxel keys that fall within the radar's detection range,
        # excluding voxels whose centres are below the planet surface.
        # An elevated radar extends the geometric horizon by sqrt(2 * R_planet * altitude).
        planet_radius_km = self.planet.radius / 1000.0
        horizon_km = np.sqrt(2.0 * planet_radius_km * self.altitude_km) if self.altitude_km > 0 else 0.0
        effective_range_km = self.range_km + horizon_km
        range_voxels = int(np.ceil(effective_range_km / self.voxel_size_km))

        radar_key = to_voxel_key(self.position)

        offsets_1d = np.arange(-range_voxels, range_voxels + 1)
        offsets = np.array(list(itertools.product(offsets_1d, repeat=3)))  # (N, 3)

        # World-space centre of each candidate voxel (metres)
        voxel_centers = (radar_key + offsets + 0.5) * self.voxel_size_km * 1000.0  # (N, 3)
        dists = np.linalg.norm(voxel_centers - self.planet.position, axis=1)  # (N,)

        valid = offsets[dists >= self.planet.radius]
        return [tuple(int(v) for v in row) for row in valid]

    def detect(self, obj: MovableObj) -> bool:
        obj_voxel = to_voxel_key(obj.position)
        return obj_voxel in self.detection_voxels
