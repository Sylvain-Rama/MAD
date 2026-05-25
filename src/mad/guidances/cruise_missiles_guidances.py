from mad.objs.base import MovableObj
from mad.guidances.base_guidances import Guidance, GuidableObj, GuidanceResults
from dataclasses import dataclass, asdict

import numpy as np
from numpy.typing import NDArray

from mad.logger import SourceLogger


logger = SourceLogger()

@dataclass
class CruiseGuidanceConfig:
    max_speed_m_s: float
    cruise_altitude_m: float
    max_range_m: float
    waypoints: list[MovableObj]

def define_trajectory(points:list[MovableObj], dt:float=1.0):
    pass


class CruiseWaypointGuidance(Guidance):
    def __init__(self, planet, target: MovableObj, config:CruiseGuidanceConfig):
        super().__init__(planet, target)
        self.cfg = config
        
        # Sign convention: +1 if local t_hat is prograde (toward target), -1 if retrograde.
        # Resolved once on the first get_guidance call.
        self._t_hat_sign: float | None = None

