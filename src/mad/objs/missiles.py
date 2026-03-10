from dataclasses import dataclass, asdict
import numpy as np
from numpy.typing import NDArray
from mad.objs.common_schemas import MovableObject, History
from mad.objs.projectiles import ProjectileConfig, Projectile
from mad.objs.planets import Planet, SimulationInterface
from mad.objs.guidances import Guidance
from mad.logger import SourceLogger
from mad.objs.constants import G0

from copy import deepcopy

logger = SourceLogger()


class Payload(MovableObject):
    mass: float  # kg
    area: float  # m^2
    yield_kt: float  # kt


@dataclass
class StageConfig:
    dry_mass: float  # kg
    propellant_mass: float  # kg
    thrust: float  # N = kg * m / s^2
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
class BallisticConfig:
    stages: list[MissileStage]
    position: list[float]
    name: str = "MultiStageMissile"
    guidance: Guidance | None = None

    @property
    def to_dict(self):
        return asdict(self)


class BallisticMissile(SimulationInterface, MovableObject):
    def __init__(self, cfg: BallisticConfig, t=0.0):
        super().__init__(position=cfg.position, name=cfg.name)

        self.stages = cfg.stages
        self.guidance = cfg.guidance
        self.t = t
        self.history = History(time=[t], position=[self.position.tolist()], velocity=[self.velocity.tolist()])
        self.initial_mass = deepcopy(self.mass)
        self.final_mass = deepcopy(sum(stage.dry_mass for stage in self.stages))

    @property
    def mass(self):
        return sum(stage.mass for stage in self.stages)

    @property
    def area(self):
        return sum(stage.area for stage in self.stages)

    @property
    def Cd(self):
        return sum(stage.Cd for stage in self.stages)

    @property
    def deltav(self):
        dv_total = 0.0

        for i, stage in enumerate(self.stages):
            m0 = sum(s.mass for s in self.stages[i:])
            mf = m0 - stage.propellant_mass
            isp = stage.Isp
            dv = isp * G0 * np.log(m0 / mf)
            dv_total += dv

        return dv_total

    @property
    def burned_fraction(self) -> float:
        # Extremely imprecise, as it does not take into account we lose stages
        return np.clip((self.initial_mass - self.mass) / (self.initial_mass - self.final_mass), 0, 1)

    def ballistic_range(self, planet: Planet, gamma_deg: float = 30):
        # Helper to quickly determine the range of the missile.
        gamma = np.radians(gamma_deg)
        # Taking 0.8 to estimate for drag / gravity / steering losses
        deltav = 0.8 * self.deltav
        num = deltav**2 * np.sin(gamma) * np.cos(gamma)
        den = planet.mu / planet.radius - deltav**2 * np.sin(gamma) ** 2
        central_angle = 2 * np.arctan(num / den)

        return planet.radius * central_angle

    def __repr__(self):
        a = "active" if self.active else "inactive"
        return f"BallisticMissile {self.name}, deltaV {self.deltav} m/s, {a}."

    @property
    def thrust_acc(self):
        running_stage = self.stages[0]
        if not running_stage.active:
            return np.zeros_like(self.velocity)

        return running_stage.thrust / self.mass

    def update(self, dt: float) -> None | Projectile:
        self.t += dt
        running_stage = self.stages[0]
        running_stage.update(dt)

        if not running_stage.active:
            stage_cfg = ProjectileConfig(
                position=self.position.tolist(),
                velocity=self.velocity.tolist(),
                mass=running_stage.dry_mass,
                name=running_stage.name,
                area=running_stage.area,
                Cd=running_stage.Cd,
            )

            del self.stages[0]
            logger["Missile"].info(f"{self.name} - {running_stage.name} separated at {self.t:.2f}.")
            if len(self.stages) == 0:
                self.active = False
                logger["Missile"].info(f"{self.name} inactivated at {self.t:.2f}.")
            return Projectile(stage_cfg, t=deepcopy(self.t))
        else:
            return None

    def accelerations(self, planet: Planet) -> NDArray:
        gravity = planet.gravity(self)
        drag = planet.drag(self)

        direction = self.guidance.get_guidance(self) if self.guidance else np.ones_like(self.position)
        thrust = self.thrust_acc * direction

        return gravity + drag + thrust

    def integrate(self, dt: float, planet: Planet) -> None:
        # Velocity Verlet for solver.
        a0 = self.accelerations(planet)
        self.position += self.velocity * dt + 0.5 * a0 * dt**2
        a1 = self.accelerations(planet)

        self.velocity += 0.5 * (a0 + a1) * dt

        self.history.update(self.t, self.position.tolist(), self.velocity.tolist())
