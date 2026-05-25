from dataclasses import dataclass, asdict, field
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING
from mad.objs.base import BallisticObj, GuidedObj, MovableObj, Payload, ReleasableConfig
from mad.objs.projectiles import ProjectileConfig, Projectile
from mad.objs.planets import Planet
from mad.logger import SourceLogger
from mad.configs.physics import G0



if TYPE_CHECKING:
    from mad.guidances import Guidance

logger = SourceLogger()


@dataclass
class CruiseMissileConfig:
    mass: float  # kg
    ref_radius: float  # m
    Cd: float
    name: str = "CruiseMissile"
    guidance: "Guidance | None" = None
    
    def __post_init__(self):
        self.area = np.pi * self.ref_radius**2

    def create(self, position: NDArray, velocity: NDArray, t: float) -> "CruiseMissile":
        return CruiseMissile(config=self, position=position, velocity=velocity, t=t)