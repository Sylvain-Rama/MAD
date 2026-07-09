import numpy as np
from dataclasses import dataclass

from mad.configs import B53_warhead, rod_of_god, V1

AVAILABLE_OBJECTS = {
    "B53_warhead": B53_warhead,
    "rod_of_god": rod_of_god,
    "V1": V1,
}


@dataclass
class SimParameters:
    altitudes_km: np.ndarray
    velocities_kms: np.ndarray
    gammas_deg: np.ndarray
    dt: float
    max_time: float


SIM_PARAMETERS = {
    "rod_of_god": {
        "altitudes_km": np.arange(500, 1201, 10),
        "velocities_kms": np.arange(8, 10, 0.2),
        "gammas_deg": np.arange(-50, 20, 2),
        "dt": 10.0,
        "max_time": 3600.0,
    },
    "B53_warhead_fine": {
        "altitudes_km": np.arange(0, 500, 10),
        "velocities_kms": np.arange(0.5, 8, 0.1),
        "gammas_deg": np.arange(20, 90, 1),
        "dt": 2.0,
        "max_time": 3600.0,
    },
    "V1": {
        "altitudes_km": np.arange(0.1, 1, 0.05),
        "velocities_kms": np.arange(0.1, 0.2, 0.01),
        "gammas_deg": np.arange(-10, 10, 1),
        "dt": 1.0,
        "max_time": 3600.0,
    },
}
