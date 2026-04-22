from mad.objs.base import MovableObj
from mad.objs.planets import Planet

from dataclasses import dataclass, asdict

@dataclass
class RadarConfig:
    position: list[float]
    name: str = "Radar"
    range_km: float = 100_000.0  # m

    @property
    def to_dict(self):
        return asdict(self)


class Radar(MovableObj):
    def __init__(self, config: RadarConfig):
        super().__init__(config.position, velocity=None, name=config.name)
        self.range_km = config.range_km

    def detect(self, obj: MovableObj, planet: Planet) -> bool:
        # Simple line-of-sight check: if the object is above the horizon from the radar's perspective.
        # This is a very basic model and can be improved with more complex radar equations and atmospheric effects.
        radar_to_obj = obj.position - self.position
        radar_to_obj_dist = np.linalg.norm(radar_to_obj)
        if radar_to_obj_dist < 1e-8:
            return True  # Object is at the same position as radar

        radar_to_obj_dir = radar_to_obj / radar_to_obj_dist
        planet_to_radar = self.position - planet.position
        planet_to_radar_dist = np.linalg.norm(planet_to_radar)
        if planet_to_radar_dist < 1e-8:
            return False  # Radar is at the center of the planet, which is unrealistic

        planet_to_radar_dir = planet_to_radar / planet_to_radar_dist

        # Check if the angle between radar-to-object and planet-to-radar is less than 90 degrees
        # If it is greater than 90 degrees, the object is below the horizon and not detectable.
        dot_product = np.dot(radar_to_obj_dir, planet_to_radar_dir)
        return dot_product > 0