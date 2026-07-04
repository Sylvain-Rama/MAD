"""This module defines the ReentryVehicle, RocketStage, and Rocket classes, which represent different types of ballistic objects
that can be launched and will be affected by gravity and drag forces.
Rockets are defined by a list of RocketStage objects, a guidance system, and a list of payloads.
"""

from dataclasses import dataclass, asdict, field
import numpy as np
from numpy.typing import NDArray
from mad.objs import (
    BallisticObj,
    GuidedObj,
    MovableObj,
    Payload,
    ReleasableConfig,
    ProjectileConfig,
    Projectile,
    Planet,
)

from mad.guidances import GuidanceStates, Guidance, GuidanceManager
from mad.utils.logger import SourceLogger
from mad.configs import G0

from copy import deepcopy

logger = SourceLogger()


@dataclass
class RVConfig:
    mass: float  # kg
    ref_radius: float  # m
    Cd: float
    guidance: Guidance | GuidanceManager
    name: str = "ReentryVehicle"
    yield_kt: float = 0.0  # kt
    RCS_thrust: float = 500.0  # N, used for terminal guidance.

    def __post_init__(self):
        self.area = np.pi * self.ref_radius**2

    def create(self, position: NDArray, velocity: NDArray, t: float) -> "ReentryVehicle":
        return ReentryVehicle(config=self, position=position, velocity=velocity, t=t)


class ReentryVehicle(Payload, GuidedObj):
    """RVs are a special type of payload that can receive guidance commands and have a yield (kt) for detonation.
    They have a small RCS thruster for terminal guidance, which is used to steer the RV towards its target during the final phase of flight.
    We assume this thruster is not limited by propellant mass, and can be used for the entire flight.
    This is a simplification, but it allows us to focus on the guidance and detonation aspects of the RV without worrying about propellant management.
    """

    def __init__(self, config: RVConfig, position: NDArray, velocity=None, t=0.0):
        Payload.__init__(self, position, velocity, config.name, config.mass, config.area, config.Cd, t)
        self.yield_kt = config.yield_kt
        self.guidance = config.guidance
        self.guidance_results = self.guidance.get_guidance(self, t)
        self.RCS_thrust = config.RCS_thrust  # N, typical for small thrusters

    @property
    def has_thrust(self) -> bool:
        return self.RCS_thrust > 0

    @property
    def thrust_acc(self) -> float:
        return self.RCS_thrust / self.mass

    @property
    def burned_fraction(self) -> float:
        # Payloads don't burn, but we can use this to smoothly transition from ballistic to terminal guidance.
        return 0.5

    def update(self, dt: float) -> None:
        self.t += dt
        self.guidance_results = self.guidance.get_guidance(self, self.t)

        return None

    def accelerations(self, planet: Planet) -> NDArray:

        if self.distance(planet) <= planet.radius:
            if self.guidance:
                distance_to_target = self.guidance.planet.surface_distance(self, self.guidance.target)
                logger["Rocket"].info(
                    f"{self.t:<.2f}s - Warhead {self.name} hit target at {distance_to_target/1000:.2f} km."
                )
                self.detonate()
            else:
                self.detonate()

            self.active = False
            return np.zeros_like(self.velocity)

        gravity = planet.gravity(self)
        drag = planet.drag(self)

        thrust = np.zeros_like(self.velocity)
        if self.guidance_results.state != GuidanceStates.IDLE:
            d = self.guidance_results.direction
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-8:
                desired_acc = self.guidance_results.magnitude
                acc = min(self.thrust_acc, desired_acc) if desired_acc is not None else self.thrust_acc
                thrust = acc * d / d_norm

        return gravity + drag + thrust

    def detonate(self):
        logger["Rocket"].info(f"{self.t:<.2f}s - Warhead {self.name} detonated with yield {self.yield_kt:.2f} kt.")
        self.active = False


@dataclass
class RocketStageConfig:
    thrust: float  # N = kg * m / s^2
    ref_radius: float  # m
    Cd: float = 0.5  # Pointy end.
    name: str = "RocketStage"
    dry_mass: float | None = None  # kg
    propellant_mass: float | None = None  # kg
    full_mass: float | None = None  # kg
    Isp: float | None = None  # s
    burn_time: float | None = None  # s, optional for now, used to calculate Isp if needed.
    parallel: bool = False  # If True, this stage ignites simultaneously with the stage before it.
    separation_retrograde_dv: float = 0.0  # m/s retrograde delta-v applied to the stage hull at separation.

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

    @property
    def to_dict(self):
        return asdict(self)


class RocketStage:
    def __init__(self, cfg: RocketStageConfig):
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

        self.parallel: bool = cfg.parallel
        self.separation_retrograde_dv: float = cfg.separation_retrograde_dv
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

        self.t += dt
        if not self.active:
            return

        if self.propellant_mass > 0:
            dm = self.mass_flow_rate * dt
            self.propellant_mass = max(0.0, self.propellant_mass - dm)
        else:
            logger["Rocket"].info(f"{self.t:<.2f}s - {self.name} ran out of propellant at {self.t:.2f}.")
            self.active = False


@dataclass
class RocketConfig:
    stages: list[RocketStage]
    guidance: Guidance | GuidanceManager
    payloads: list[ReleasableConfig] = field(default_factory=list)
    payload_separation_interval: float = 2.0  # Time between payload separations, in seconds.

    @property
    def to_dict(self):
        return asdict(self)

    def create(self, position: NDArray, velocity: NDArray | None = None, t: float = 0.0) -> "Rocket":
        return Rocket(position=position, cfg=self, velocity=velocity, t=t)


class Rocket(BallisticObj, GuidedObj):
    def __init__(self, position, cfg: RocketConfig, velocity=None, name="Rocket", t=0.0):
        # mass and area are computed properties on this class; bypass BallisticObj.__init__
        # to avoid storing unused _mass/_area defaults.
        MovableObj.__init__(self, position=position, velocity=velocity, name=name)

        self.stages = cfg.stages
        self.guidance = cfg.guidance
        self.payloads: list[ReleasableConfig] = list(cfg.payloads)  # mutable copy; entries popped on release
        self.t = t
        self.n_payloads = len(self.payloads)  # initial count, used for burned_fraction
        self.released_payloads = 0
        self.payload_separation_interval = cfg.payload_separation_interval
        self.last_payload_separation_time = 0.0

        self.initial_mass = deepcopy(self.mass)
        # TODO: final_mass is used as the normalization bound for burned_fraction, which drives
        # the steering law.  Using only the first payload's mass preserves the original calibration
        # but is incorrect for heterogeneous or variable-count payload lists.
        self.final_mass = deepcopy(
            sum(stage.dry_mass for stage in self.stages) + (self.payloads[0].mass if self.payloads else 0.0)
        )
        self.Cd = 1.08  # long cylinder, should be good enough for a first approximation
        self.guidance_results = self.guidance.get_guidance(self) if self.guidance else None

        # Cached area/Cd for the coasting phase (all stages separated, payloads not yet released).
        # Use the first payload's properties so the drag-to-mass ratio is physically correct.
        self._coasting_area: float = (
            getattr(self.payloads[0], "area", self.stages[-1].area) if self.payloads else self.stages[-1].area
        )
        self._coasting_Cd: float = getattr(self.payloads[0], "Cd", self.Cd) if self.payloads else self.Cd

    @property
    def mass(self):
        # Payload masses are excluded: they only exist once released as independent objects.
        return sum(stage.mass for stage in self.stages)

    @property
    def area(self):
        return self.stages[-1].area if self.stages else self._coasting_area

    @property
    def has_thrust(self) -> bool:
        """True while at least one stage is still present (stages are removed upon propellant depletion)."""
        return bool(self.stages)

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
        if not self.stages:
            return 1.0
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
            f"Rocket {self.name}, {a}.\n"
            f"Stages: {", ".join([x.name for x in self.stages])}.\n"
            f"Available deltaV: {self.deltav:.2f} m/s.\n"
            f"Guidance: {self.guidance.__class__.__name__ if self.guidance else 'None'}.\n"
            f"Payloads: {', '.join(p.name for p in self.payloads) if self.payloads else 'None'}.\n"
        )

    @property
    def _active_burn_group(self) -> list["RocketStage"]:
        """Stage 0 plus any consecutive following stages whose ``parallel`` flag is True.

        All stages in this group burn simultaneously.  The group shrinks as
        individual stages deplete and separate; the next sequential stage only
        ignites once the group is completely exhausted.
        """
        if not self.stages:
            return []
        group = [self.stages[0]]
        for stage in self.stages[1:]:
            if stage.parallel:
                group.append(stage)
            else:
                break
        return group

    @property
    def thrust_acc(self) -> float:
        if not self.stages:
            return 0.0
        total_thrust = sum(s.thrust_force for s in self._active_burn_group if s.active)
        return total_thrust / self.mass if total_thrust > 0 else 0.0

    def update(self, dt: float) -> list[BallisticObj] | None:
        released_objects = []
        self.t += dt

        # Snapshot the active burn group before any mutations so that stage-removal
        # logic below can reliably tell which stages were already burning.
        burn_group = self._active_burn_group
        self.guidance_results = self.guidance.get_guidance(self, self.t)
        if self.guidance_results.state != GuidanceStates.IDLE:
            for stage in burn_group:
                stage.update(dt)

        if (
            self.guidance_results.state == GuidanceStates.RELEASE_PAYLOAD
            and self.t - self.last_payload_separation_time > self.payload_separation_interval
            and self.payloads
        ):
            next_cfg = self.payloads.pop(0)
            payload_name = f"{next_cfg.name}_{self.released_payloads + 1}"

            release_velocity = (
                self.guidance_results.release_velocity
                if self.guidance_results.release_velocity is not None
                else self.velocity.copy()
            )

            payload = next_cfg.create(
                position=self.position.copy(),
                velocity=release_velocity,
                t=deepcopy(self.t),
            )
            payload.name = payload_name
            released_objects.append(payload)
            logger["Rocket"].info(f"{self.t:<.2f}s - {self.name} released payload {payload_name} at {self.t:.2f}.")
            self.released_payloads += 1
            self.last_payload_separation_time = deepcopy(self.t)

            # Cut thrust immediately on first payload release without deactivating stages
            # (stages must remain active to allow subsequent payload separations and keep the same mass).
            for stage in self.stages:
                stage.thrust = 0.0
                stage.mass_flow_rate = 0.0

        if self.n_payloads > 0 and self.released_payloads >= self.n_payloads:
            for stage in self.stages:
                stage.active = False
            if not self.stages:
                # Coasting phase: no stages left, deactivate directly.
                self.active = False
            logger["Rocket"].info(
                f"{self.t:<.2f}s - {self.name} has released all payloads at {self.t:.2f}. Stages deactivated."
            )

        # Separate every depleted stage in the burn group (may be >1 for parallel stages).
        depleted = [s for s in burn_group if not s.active]
        for dep in depleted:
            speed = np.linalg.norm(self.velocity)
            if dep.separation_retrograde_dv > 0 and speed > 1e-6:
                sep_velocity = self.velocity - dep.separation_retrograde_dv * (self.velocity / speed)
            else:
                sep_velocity = self.velocity
            stage_cfg = ProjectileConfig(
                position=self.position.tolist(),
                velocity=sep_velocity.tolist(),
                mass=dep.dry_mass,
                name=dep.name,
                ref_radius=dep.ref_radius,
                Cd=dep.Cd,
            )
            self.stages.remove(dep)
            logger["Rocket"].info(f"{self.t:<.2f}s - {self.name} - {dep.name} separated at {self.t:.2f}.")
            released_objects.append(Projectile(stage_cfg, t=deepcopy(self.t)))

        if depleted:
            if len(self.stages) == 0:
                if not self.payloads:
                    self.active = False
                    logger["Rocket"].info(f"{self.t:<.2f}s - {self.name} inactivated at {self.t:.2f}.")
                else:
                    self.Cd = self._coasting_Cd
                    logger["Rocket"].info(f"{self.t:<.2f}s - {self.name} entering coast phase at {self.t:.2f}.")
            elif self.stages[0] not in burn_group:
                # A new sequential stage is now at the front; record when it started.
                self.stages[0].t = self.t
                logger["Rocket"].info(f"{self.t:<.2f}s - {self.name} - {self.stages[0].name} ignited at {self.t:.2f}.")

        return released_objects if released_objects else None

    def accelerations(self, planet: Planet) -> NDArray:
        if self.distance(planet) <= planet.radius:
            logger["Rocket"].info(f"{self.t:<.2f}s - {self.name} impacted the ground at {self.t:.2f}.")
            self.active = False
            return np.zeros_like(self.velocity)

        gravity = planet.gravity(self)

        drag = (
            planet.drag(self) if self.stages else np.zeros_like(self.velocity)
        )  # No drag if all stages separated and payloads not yet released.

        # If there is no thrust, no need to check for direction: we cannot act on it.
        if self.thrust_acc > 0:
            # If no guidance, we continue along the same direction.
            direction = self.guidance_results.direction if self.guidance_results else self.normalize
            direction = direction / np.linalg.norm(direction)
            thrust = self.thrust_acc * direction
        else:
            thrust = np.zeros_like(self.velocity)

        return gravity + drag + thrust
