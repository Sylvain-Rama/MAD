from dataclasses import dataclass, asdict
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING
from mad.objs.base import BallisticObj, GuidedObj, History, MovableObj
from mad.objs.projectiles import ProjectileConfig, Projectile
from mad.objs.planets import Planet
from mad.logger import SourceLogger
from mad.configs.physics import G0

from copy import deepcopy

if TYPE_CHECKING:
    from mad.guidances import Guidance

logger = SourceLogger()


@dataclass
class PayloadConfig:
    mass: float  # kg
    ref_radius: float  # m
    Cd: float
    name: str = "Payload"
    yield_kt: float = 0.0  # kt
    guidance: "Guidance | None" = None
    RCS_thrust: float = 500.0  # N, used for terminal guidance.

    def __post_init__(self):
        self.area = np.pi * self.ref_radius**2


class Payload(BallisticObj, GuidedObj):
    def __init__(self, config: PayloadConfig, position: NDArray, velocity=None, t=0.0):
        super().__init__(
            position=position,
            velocity=velocity,
            name=config.name,
            mass=config.mass,
            area=config.area,
            Cd=config.Cd,
        )
        self.yield_kt = config.yield_kt
        self.guidance = config.guidance
        self.t = t
        self.guidance_results = self.guidance.get_guidance(self, t) if self.guidance else None
        self.RCS_thrust = config.RCS_thrust  # N, typical for small thrusters

        self.history = History(
            time=[t],
            position=[self.position.tolist()],
            velocity=[self.velocity.tolist()],
            gamma=[self.guidance_results.gamma if self.guidance_results else None],
        )

    @property
    def thrust_acc(self) -> float:
        return self.RCS_thrust / self.mass

    @property
    def burned_fraction(self) -> float:
        # Payloads don't burn, but we can use this to smoothly transition from ballistic to terminal guidance.
        return 0.5

    def update(self, dt: float) -> None:
        self.t += dt
        self.guidance_results = self.guidance.get_guidance(self, self.t) if self.guidance else None

        return None

    def accelerations(self, planet: Planet) -> NDArray:

        if self.distance(planet) <= planet.radius:
            if self.guidance:
                distance_to_target = self.guidance.planet.surface_distance(self, self.guidance.target)
                logger["Missile"].info(f"Warhead {self.name} hit target at {distance_to_target/1000:.2f} km.")
            else:
                logger["Missile"].info(f"Warhead {self.name} detonated on the ground!")

            self.active = False
            return np.zeros_like(self.velocity)

        gravity = planet.gravity(self)
        drag = planet.drag(self)

        thrust = np.zeros_like(self.velocity)
        if self.guidance_results is not None:
            d = self.guidance_results.direction
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-8:
                desired_acc = self.guidance_results.magnitude
                acc = min(self.thrust_acc, desired_acc) if desired_acc is not None else self.thrust_acc
                thrust = acc * (d / d_norm)

        return gravity + drag + thrust

    def integrate(self, dt: float, planet: Planet) -> None:
        # Velocity Verlet for solver.
        a0 = self.accelerations(planet)
        self.position += self.velocity * dt + 0.5 * a0 * dt**2
        a1 = self.accelerations(planet)

        self.velocity += 0.5 * (a0 + a1) * dt

        self.history.update(
            self.t,
            self.position.tolist(),
            self.velocity.tolist(),
            self.guidance_results.gamma if self.guidance_results else None,
        )


@dataclass
class MissileStageConfig:
    thrust: float  # N = kg * m / s^2
    ref_radius: float  # m
    Cd: float = 1.08  # smooth, long cylinder.
    name: str = "MissileStage"
    dry_mass: float | None = None  # kg
    propellant_mass: float | None = None  # kg
    full_mass: float | None = None  # kg
    Isp: float | None = None  # s
    burn_time: float | None = None  # s, optional for now, can be computed from mass and thrust if not provided.

    def __post_init__(self):
        self.area = np.pi * self.ref_radius**2
        # Convenience methods to specify any two of dry_mass, propellant_mass and full_mass
        if self.full_mass is None:
            if self.dry_mass is not None and self.propellant_mass is not None:
                self.full_mass = self.dry_mass + self.propellant_mass
        elif self.dry_mass is None:
            if self.propellant_mass is not None:
                self.dry_mass = self.full_mass - self.propellant_mass
        elif self.propellant_mass is None:
            if self.dry_mass is not None:
                self.propellant_mass = self.full_mass - self.dry_mass
        elif abs((self.dry_mass + self.propellant_mass) - self.full_mass) >= 10:  # 10 kg tolerance for inconsistency
            # All three provided, check consistency.
            raise ValueError(f"Inconsistent masses for {self.name}.")

        if self.Isp is None and self.burn_time is not None:
            # Compute Isp from burn time if not provided.
            total_impulse = self.thrust * self.burn_time
            if self.propellant_mass is not None and self.propellant_mass > 0:
                self.Isp = total_impulse / (self.propellant_mass * G0)
            else:
                raise ValueError(f"Cannot compute Isp for {self.name} without propellant mass.")
        else:
            # We require Isp to be provided to compute burn time if not provided.
            if self.burn_time is None:
                raise ValueError(f"Burn time must be provided for {self.name} if Isp is not provided.")

    @property
    def to_dict(self):
        return asdict(self)


class MissileStage:
    def __init__(self, cfg: MissileStageConfig):
        self.config = cfg
        if cfg.dry_mass is None:
            raise ValueError(f"dry_mass could not be determined for {cfg.name}.")
        if cfg.propellant_mass is None:
            raise ValueError(f"propellant_mass could not be determined for {cfg.name}.")
        if cfg.Isp is None:
            raise ValueError(f"Isp must be provided for {cfg.name} to compute exhaust velocity.")

        self.dry_mass = cfg.dry_mass
        self.propellant_mass = cfg.propellant_mass
        self.thrust = cfg.thrust
        self.ref_radius = cfg.ref_radius
        self.Cd = cfg.Cd
        self.area = np.pi * self.ref_radius**2

        self.Isp = cfg.Isp
        self.exhaust_velocity = cfg.Isp * G0
        self.mass_flow_rate = cfg.thrust / self.exhaust_velocity

        self.active: bool = True
        self.name = cfg.name
        self.t = 0.0

    @property
    def mass(self) -> float:
        return self.dry_mass + self.propellant_mass

    @property
    def thrust_force(self) -> float:
        return self.thrust if self.propellant_mass > 0 else 0.0

    def update(self, dt: float) -> None:

        # TODO: Ability to keep the last stage active even if propellant is depleted,
        # to allow for coasting phases between stage separations and more flexible guidance.

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
class BallisticMissileConfig:
    stages: list[MissileStage]
    guidance: "Guidance | None" = None
    payload: PayloadConfig | None = None
    n_RVs: int = 1  # Number of reentry vehicles, used for terminal guidance.
    RV_separation_interval: float = 2.0  # Time between RV separations, in seconds.

    @property
    def to_dict(self):
        return asdict(self)


class BallisticMissile(BallisticObj, GuidedObj):
    def __init__(self, position, cfg: BallisticMissileConfig, velocity=None, name="BallisticMissile", t=0.0):
        # mass and area are computed properties on this class; bypass BallisticObj.__init__
        # to avoid storing unused _mass/_area defaults.
        MovableObj.__init__(self, position=position, velocity=velocity, name=name)

        self.stages = cfg.stages
        self.guidance = cfg.guidance
        self.payload = cfg.payload
        self.t = t
        self.n_RVs = cfg.n_RVs
        self.released_RVs = 0
        self.RV_separation_interval = cfg.RV_separation_interval
        self.last_RV_separation_time = 0.0

        self.initial_mass = deepcopy(self.mass)
        self.final_mass = deepcopy(
            sum(stage.dry_mass for stage in self.stages) + (self.payload.mass if self.payload else 0.0)
        )
        self.Cd = 1.08  # long cylinder, should be good enough for a first approximation
        self.guidance_results = self.guidance.get_guidance(self) if self.guidance else None

        self.history = History(
            time=[t],
            position=[self.position.tolist()],
            velocity=[self.velocity.tolist()],
            gamma=[self.guidance_results.gamma if self.guidance_results else None],
        )

    @property
    def mass(self):
        # We ignore the payload mass until the end, when it is released.
        # Allows us to not have to worry about the transition from missile to payload.
        return sum(stage.mass for stage in self.stages)

    @property
    def area(self):
        return self.stages[-1].area

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

    def ballistic_range(self, planet: Planet, gamma_rad: float = np.radians(45)) -> float:
        # Helper to quickly determine the range of the missile.
        # Taking 0.8 to estimate for drag / gravity / steering losses
        deltav = 0.8 * self.deltav
        num = deltav**2 * np.sin(gamma_rad) * np.cos(gamma_rad)
        den = planet.mu / planet.radius - deltav**2 * np.sin(gamma_rad) ** 2
        central_angle = 2 * np.arctan(num / den)

        return planet.radius * central_angle

    def __repr__(self):
        a = "active" if self.active else "inactive"
        return (
            f"BallisticMissile {self.name}, {a}.\n"
            f"Stages: {", ".join([x.name for x in self.stages])}.\n"
            f"Available deltaV: {self.deltav:.2f} m/s.\n"
            f"Burned Fraction: {self.burned_fraction:.2f}."
        )

    @property
    def thrust_acc(self) -> float:
        running_stage = self.stages[0]
        if not running_stage.active:
            return 0.0

        return running_stage.thrust_force / self.mass

    def update(self, dt: float) -> list[BallisticObj] | None:
        released_objects = []
        self.t += dt

        running_stage = self.stages[0]
        running_stage.update(dt)

        self.guidance_results = self.guidance.get_guidance(self, self.t) if self.guidance else None

        if self.guidance_results:
            if (
                self.guidance_results.state == "Release RV"
                and self.t - self.last_RV_separation_time > self.RV_separation_interval
                and self.payload
            ):
                payload_name = f"{self.payload.name}_{self.released_RVs + 1}"

                release_velocity = (
                    self.guidance_results.release_velocity
                    if self.guidance_results.release_velocity is not None
                    else self.velocity.copy()
                )

                payload = Payload(
                    config=self.payload,
                    position=self.position.copy(),
                    velocity=release_velocity,
                    t=deepcopy(self.t),
                )
                payload.name = payload_name
                released_objects.append(payload)
                logger["Missile"].info(f"{self.name} released payload {payload_name} at {self.t:.2f}.")
                self.released_RVs += 1
                self.last_RV_separation_time = deepcopy(self.t)

                # Cut thrust immediately on first RV release without deactivating stages
                # (stages must remain active to allow subsequent RV separations and keep the same mass).
                for stage in self.stages:
                    stage.thrust = 0.0
                    stage.mass_flow_rate = 0.0

        if self.released_RVs >= self.n_RVs:
            [setattr(stage, "active", False) for stage in self.stages]
            running_stage.active = False
            logger["Missile"].info(f"{self.name} has released all RVs at {self.t:.2f}. Stages deactivated.")

        if not running_stage.active:
            stage_cfg = ProjectileConfig(
                position=self.position.tolist(),
                velocity=self.velocity.tolist(),
                mass=running_stage.dry_mass,
                name=running_stage.name,
                ref_radius=running_stage.ref_radius,
                Cd=running_stage.Cd,
            )

            del self.stages[0]
            logger["Missile"].info(f"{self.name} - {running_stage.name} separated at {self.t:.2f}.")
            if len(self.stages) == 0:
                self.active = False
                logger["Missile"].info(f"{self.name} inactivated at {self.t:.2f}.")
            else:
                self.stages[0].t = self.t

            released_objects.append(Projectile(stage_cfg, t=deepcopy(self.t)))

        return released_objects if released_objects else None

    def accelerations(self, planet: Planet) -> NDArray:
        gravity = planet.gravity(self)
        drag = planet.drag(self)

        # If there is no thrust, no need to check for direction: we cannot act on it.
        if self.thrust_acc > 0:
            # If no guidance, we continue along the same direction.
            direction = self.guidance_results.direction if self.guidance_results else self.normalize
            direction = direction / np.linalg.norm(direction)
            thrust = self.thrust_acc * direction
        else:
            thrust = np.zeros_like(self.velocity)

        return gravity + drag + thrust

    def integrate(self, dt: float, planet: Planet) -> None:
        # Velocity Verlet for solver.
        a0 = self.accelerations(planet)
        self.position += self.velocity * dt + 0.5 * a0 * dt**2
        a1 = self.accelerations(planet)

        self.velocity += 0.5 * (a0 + a1) * dt

        self.history.update(
            self.t,
            self.position.tolist(),
            self.velocity.tolist(),
            self.guidance_results.gamma if self.guidance_results else None,
        )
