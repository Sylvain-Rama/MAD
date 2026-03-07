import numpy as np
from dataclasses import dataclass, asdict
from mad.objs.common_schemas import MovableObject
from mad.logger import SourceLogger
from mad.objs.constants import G0

logger = SourceLogger()


class Payload(MovableObject):
    mass: float  # kg
    area: float  # m^2
    yield_kt: float  # kt


@dataclass
class StageConfig:
    dry_mass: float  # kg
    propellant_mass: float  # kg
    thrust: float  # N
    Isp: float  # s
    area: float  # m^2
    Cd: float
    time_ECO: float  # s
    time_sep: float  # s
    payload: Payload | None = None
    name: str = "Stage"

    @property
    def to_dict(self):
        return asdict(self)


class MissileStage:
    def __init__(self, cfg: StageConfig):
        self.config = cfg
        self.dry_mass = cfg.dry_mass
        self.propellant_mass = cfg.propellant_mass

        self.thrust = cfg.thrust
        self.Isp = cfg.Isp

        self.area = cfg.area
        self.Cd = cfg.Cd

        self.exhaust_velocity = cfg.Isp * G0
        self.mass_flow_rate = cfg.thrust / self.exhaust_velocity

        self.active: bool = True
        self.payload = cfg.payload
        self.name = cfg.name
        self.t = 0.0

    @property
    def mass(self) -> float:
        payload_mass = self.payload.mass if self.payload else 0.0
        return self.dry_mass + self.propellant_mass + payload_mass

    def thrust_force(self) -> float:
        return self.thrust if self.propellant_mass > 0 else 0.0

    def update(self, dt: float) -> None:
        self.t += dt
        if not self.active:
            return

        if self.propellant_mass > 0:
            dm = self.mass_flow_rate * dt
            self.propellant_mass = max(0.0, self.propellant_mass - dm)
        else:
            logger["Missile"].info(f"{self.name} ran out of propellant at {self.t:.2f}.")
            self.active = False


@dataclass
class Guidance:
    cruise_altitude: float
    target: MovableObject


@dataclass
class MissileConfig:
    stages: list[MissileStage]
    position: list[float]
    name: str = "MultiStageMissile"
    guidance: Guidance | None = None

    @property
    def to_dict(self):
        return asdict(self)
