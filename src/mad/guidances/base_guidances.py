from mad.objs import MovableObj, Planet

from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np
from numpy.typing import NDArray
from mad.utils.logger import SourceLogger

logger = SourceLogger()


GuidanceStates = Enum("GuidanceStates", ["IDLE", "POWERED", "COASTING", "TERMINAL", "RELEASE_PAYLOAD", "DETONATE"])


class GuidableObj(Protocol):
    """Structural interface expected by all Guidance implementations.

    Any object that exposes these attributes can be guided."""

    position: NDArray
    velocity: NDArray
    name: str

    @property
    def normalize(self) -> NDArray: ...

    @property
    def burned_fraction(self) -> float: ...

    @property
    def has_thrust(self) -> bool: ...

    @property
    def thrust_acc(self) -> float: ...

    def degrade(self):
        logger["Guidable"].info(f"{self.name} degraded.")


@dataclass
class GuidanceResults:
    direction: NDArray
    state: GuidanceStates
    gamma: float | None = None  # Optional angular velocity command for advanced guidance laws
    magnitude: float | None = None  # Optional desired acceleration magnitude (m/s²)
    release_velocity: NDArray | None = None  # Optimal RV release velocity vector (m/s)
    next_guidance: bool = False  # Whether to switch to the next guidance in the guidance list.


class Guidance(ABC):
    """Base class for all guidance laws.

    Provides shared geometry helpers (`central_angle`, `local_frame`,
    `optimal_gamma`, `gravity_turn_direction`) so subclasses only need to
    implement `get_guidance`.  All concrete subclasses start in the
    ``GuidanceStates.POWERED`` state; subclasses that need a different initial state
    should override it after calling ``super().__init__``.
    """

    def __init__(self, planet: Planet, target: MovableObj):
        self.planet = planet
        self.target = target
        self.state = GuidanceStates.POWERED
        # Sign convention: +1 if local t_hat from local_frame is prograde (toward target), -1 if retrograde.
        # Resolved once on the first _resolve_t_hat_sign call.
        self._t_hat_sign: float | None = None
        self.next_guidance: bool = False  # Whether to switch to the next guidance in the guidance list.

    @staticmethod
    def central_angle(missile: GuidableObj, target: MovableObj) -> NDArray:
        return np.arccos(np.clip(np.dot(missile.normalize, target.normalize), -1, 1))

    def local_frame(self, missile: GuidableObj) -> tuple[NDArray, NDArray]:
        r_hat = missile.normalize
        rt_hat = self.target.normalize

        t_hat = np.cross(np.cross(rt_hat, r_hat), r_hat)
        t_norm = np.linalg.norm(t_hat)
        if t_norm < 1e-8:
            return r_hat, np.zeros_like(r_hat)

        t_hat /= t_norm
        return r_hat, t_hat

    def optimal_gamma(self, missile: GuidableObj, sigma: NDArray) -> NDArray:
        v = np.linalg.norm(missile.velocity)
        gamma = np.arctan((v**2 - self.planet.mu / np.linalg.norm(missile.position)) / v**2 * np.tan(sigma / 2))
        return gamma

    def gravity_turn_direction(self, missile: GuidableObj, optimal_gamma: NDArray) -> NDArray:
        r_hat, t_hat = self.local_frame(missile)
        theta = optimal_gamma * missile.burned_fraction
        d = np.cos(theta) * r_hat + np.sin(theta) * t_hat
        return d / np.linalg.norm(d)

    def _resolve_t_hat_sign(self, r_hat: NDArray, t_hat: NDArray) -> float:
        """Return (and cache) the sign that makes ``t_hat`` point prograde toward ``self.target``.

        Returns +1.0 if ``t_hat`` already points in the prograde direction, -1.0 otherwise.
        The result is cached so the orientation is determined only on the first call.
        """
        if self._t_hat_sign is None:
            rt_hat = self.target.normalize
            prograde = rt_hat - np.dot(rt_hat, r_hat) * r_hat
            self._t_hat_sign = 1.0 if np.dot(prograde, t_hat) >= 0 else -1.0
        return self._t_hat_sign

    @abstractmethod
    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        pass

    def interrupt_guidance(self, planet: Planet, missile: GuidableObj, target: MovableObj, t: float = 0.0) -> None:
        """Interrupt the current guidance law and switch to the next one in the list.

        This is a convenience method that can be called by subclasses when they
        determine that their guidance is no longer appropriate (e.g., after
        burnout or when the target is no longer reachable).  It sets the
        ``next_guidance`` flag to True.
        """
        self.next_guidance = True


class GuidanceManager:
    """
    Manages a sequence of guidance laws, with their triggers and switches.
    """

    def __init__(self, guidances: list[Guidance]):
        self.guidances = guidances
        self.current_index = 0
        self.planet = guidances[0].planet
        self.target = guidances[0].target

    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        if self.current_index >= len(self.guidances):
            return GuidanceResults(
                direction=missile.velocity / np.linalg.norm(missile.velocity), state=GuidanceStates.IDLE
            )

        current_guidance = self.guidances[self.current_index]
        results = current_guidance.get_guidance(missile, t)

        if current_guidance.next_guidance:
            self.current_index += 1
            if self.current_index < len(self.guidances):
                logger["Guidance"].info(
                    f"Switching to guidance law {self.current_index}: {self.guidances[self.current_index].__class__.__name__}"
                )
                self.planet = self.guidances[self.current_index].planet
                self.target = self.guidances[self.current_index].target
            else:
                logger["Guidance"].info("No more guidance laws to switch to.")

        return results


class NoGuidance(Guidance):
    """No guidance: the missile continues on its current trajectory without any course correction."""

    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        return GuidanceResults(
            direction=missile.velocity / np.linalg.norm(missile.velocity),
            state=self.state,
            next_guidance=self.next_guidance,
        )


class GravityTurn(Guidance):
    """Gravity turn: the rocket starts vertically and gradually turns towards the target, following a smooth curve.
    The optimal curve is computed based on the current velocity and the central angle to the target."""

    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        sigma = self.central_angle(missile, self.target)
        gamma = self.optimal_gamma(missile, sigma)

        return GuidanceResults(
            direction=self.gravity_turn_direction(missile, gamma), state=self.state, next_guidance=self.next_guidance
        )


class ProportionalNavigation(Guidance):
    """Proportional Navigation (PN) terminal guidance for reentry vehicles.

    Commands acceleration perpendicular to the current velocity, proportional
    to the line-of-sight (LOS) rotation rate:

        a_cmd = N * v_c * los_rate_hat  (perpendicular to velocity)

    where N is the navigation gain (typically 3-5), v_c is the closing speed,
    and los_rate_hat is the unit LOS angular rate vector.

    Guidance is dormant until the payload is within `activation_altitude_km`
    OR within `activation_range_km` surface distance of the target. The LOS
    history is reset at activation to avoid a stale derivative.
    """

    def __init__(
        self,
        planet: Planet,
        target: MovableObj,
        N: float = 4.0,
        activation_altitude_km: float | None = 300.0,
        activation_range_km: float | None = None,
        altitude_gain: float = 0.005,
    ):
        super().__init__(planet, target)
        self.N = N
        self.activation_altitude_m = activation_altitude_km * 1000.0 if activation_altitude_km is not None else None
        self.activation_range_m = activation_range_km * 1000.0 if activation_range_km is not None else None
        self.altitude_gain = altitude_gain
        self._prev_los_hat: NDArray | None = None
        self._prev_t: float | None = None
        self._armed: bool = False

    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        los = self.target.position - missile.position
        los_norm = np.linalg.norm(los)
        if los_norm < 1e-8:
            return GuidanceResults(
                direction=np.zeros(3), state=GuidanceStates.DETONATE, next_guidance=self.next_guidance
            )
        los_hat = los / los_norm

        v_norm = np.linalg.norm(missile.velocity)
        if v_norm < 1e-8:
            return GuidanceResults(direction=np.zeros(3), state=self.state, next_guidance=self.next_guidance)
        v_hat = missile.velocity / v_norm

        # Check arming conditions; reset LOS history on the transition.
        if not self._armed:
            altitude = np.linalg.norm(missile.position) - self.planet.radius
            surface_range = self.planet.radius * self.central_angle(missile, self.target)
            armed_by_alt = self.activation_altitude_m is not None and altitude <= self.activation_altitude_m
            armed_by_range = self.activation_range_m is not None and surface_range <= self.activation_range_m
            if armed_by_alt or armed_by_range:
                self._armed = True
                self._prev_los_hat = None
                self._prev_t = None
                logger["Guidance"].info(
                    f"{t:<.2f}s - PN armed at altitude {altitude/1000:.1f} km, range {surface_range/1000:.1f} km."
                )
            else:
                return GuidanceResults(direction=np.zeros(3), state=self.state, next_guidance=self.next_guidance)

        # On first call after arming, seed the LOS history.
        if self._prev_los_hat is None or self._prev_t is None:
            self._prev_los_hat = los_hat.copy()
            self._prev_t = t
            return GuidanceResults(direction=np.zeros(3), state=self.state, next_guidance=self.next_guidance)

        dt = t - self._prev_t
        if dt < 1e-9:
            self._prev_los_hat = los_hat.copy()
            self._prev_t = t
            return GuidanceResults(direction=np.zeros(3), state=self.state, next_guidance=self.next_guidance)

        # LOS rate vector (rad/s): d(los_hat)/dt projected perpendicular to los_hat
        d_los_hat = (los_hat - self._prev_los_hat) / dt
        # Remove component along los_hat to get pure angular rate
        los_rate = d_los_hat - np.dot(d_los_hat, los_hat) * los_hat

        self._prev_los_hat = los_hat.copy()
        self._prev_t = t

        los_rate_norm = np.linalg.norm(los_rate)

        # We are on the object.
        if los_rate_norm < 1e-6:
            return GuidanceResults(
                direction=np.zeros(3), state=GuidanceStates.DETONATE, next_guidance=self.next_guidance
            )

        # Closing velocity (positive when approaching)
        v_c = -np.dot(missile.velocity, los_hat)

        # PN command: a = N * v_c * los_rate_hat, in the plane perpendicular to LOS.
        # Then project onto the plane perpendicular to velocity so RCS doesn't brake.
        a_cmd = self.N * v_c * los_rate / los_rate_norm
        a_cmd = a_cmd - np.dot(a_cmd, v_hat) * v_hat

        # Altitude matching: add a radial term proportional to the altitude error between
        # the missile and the target.  This lets the guidance correct vertical separation
        # (e.g. interceptor at cruise altitude vs low-flying target) in addition to the
        # standard lateral PN correction.  Re-project perpendicular to velocity afterwards.
        r_hat = missile.position / np.linalg.norm(missile.position)
        alt_error = np.linalg.norm(self.target.position) - np.linalg.norm(missile.position)
        a_cmd += self.altitude_gain * alt_error * r_hat
        a_cmd = a_cmd - np.dot(a_cmd, v_hat) * v_hat

        cmd_norm = np.linalg.norm(a_cmd)
        if cmd_norm < 1e-8:
            return GuidanceResults(direction=np.zeros(3), state=self.state, next_guidance=self.next_guidance)

        return GuidanceResults(
            direction=a_cmd / cmd_norm, state=self.state, magnitude=float(cmd_norm), next_guidance=self.next_guidance
        )
