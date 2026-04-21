from mad.objs.common_schemas import MovableObj

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np
from numpy.typing import NDArray
from mad.logger import SourceLogger
from mad.utils import load_ballistic_table

logger = SourceLogger()


class GuidableObj(Protocol):
    """Structural interface expected by all Guidance implementations.

    Any object that exposes these attributes can be guided — no concrete
    inheritance from BallisticMissile is required.  Both BallisticMissile
    and Payload satisfy this protocol."""

    position: NDArray
    velocity: NDArray
    name: str

    @property
    def normalize(self) -> NDArray: ...

    @property
    def burned_fraction(self) -> float: ...


@dataclass
class GuidanceResults:
    direction: NDArray
    state: str
    gamma: float | None = None  # Optional angular velocity command for advanced guidance laws
    magnitude: float | None = None  # Optional desired acceleration magnitude (m/s²)


class Guidance(ABC):
    """Base class for all guidance laws.

    Provides shared geometry helpers (`central_angle`, `local_frame`,
    `optimal_gamma`, `gravity_turn_direction`) so subclasses only need to
    implement `get_guidance`.  All concrete subclasses start in the
    ``"powered"`` state; subclasses that need a different initial state
    should override it after calling ``super().__init__``.
    """

    def __init__(self, planet, target: MovableObj):
        self.planet = planet
        self.target = target
        self.state = "powered"

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

    @abstractmethod
    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        pass


class GravityTurn(Guidance):
    """Gravity turn: the rocket starts vertically and gradually turns towards the target, following a smooth curve.
    The optimal curve is computed based on the current velocity and the central angle to the target."""

    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        sigma = self.central_angle(missile, self.target)
        gamma = self.optimal_gamma(missile, sigma)

        return GuidanceResults(direction=self.gravity_turn_direction(missile, gamma), state=self.state)


class TabulatedBallistic(Guidance):
    """The guidance returns an updated when the missile get in range to the target, according to the ballistic table.
    The ballistic table is a CSV file with columns: altitude_m, velocity_m_s, gamma_rad, range_rad.
    """

    def __init__(self, planet, target: MovableObj, ballistic_table_path: str):
        super().__init__(planet, target)
        self.ballistic_guidance = load_ballistic_table(ballistic_table_path) if ballistic_table_path else None

        # Sign convention: +1 if local t_hat is prograde (toward target), -1 if retrograde.
        # Resolved once on the first get_guidance call.
        self._t_hat_sign: float | None = None

    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:

        if self.ballistic_guidance is None:
            logger["Guidance"].error("Ballistic table not loaded. Cannot compute guidance.")
            return GuidanceResults(direction=np.zeros(3), state=self.state)

        r_hat, t_hat = self.local_frame(missile)

        # On first call, determine whether t_hat is prograde (+1) or retrograde (-1)
        # by comparing it against the projection of the target direction onto the tangential plane.
        if self._t_hat_sign is None:
            rt_hat = self.target.normalize
            prograde = rt_hat - np.dot(rt_hat, r_hat) * r_hat
            self._t_hat_sign = 1.0 if np.dot(prograde, t_hat) >= 0 else -1.0

        sigma = self.central_angle(missile, self.target)
        range_to_target = self.planet.radius * sigma

        altitude = np.linalg.norm(missile.position) - self.planet.radius
        velocity = np.linalg.norm(missile.velocity)

        # Find the closest entry in the ballistic table based on altitude, velocity and gamma.
        # Convert missile gamma to the table's prograde convention using the detected sign.
        table = self.ballistic_guidance.table

        v_r = np.dot(missile.velocity, r_hat)
        v_t = np.dot(missile.velocity, t_hat)
        missile_gamma = np.arctan2(v_r, self._t_hat_sign * v_t)

        query_point = np.array(
            [
                altitude / self.ballistic_guidance.alt_scale,
                velocity / self.ballistic_guidance.vel_scale,
                missile_gamma / self.ballistic_guidance.gam_scale,
            ]
        )
        k = min(3, len(table))
        dists, idxs = self.ballistic_guidance.kdtree.query(query_point, k=k)

        if dists[0] < 1e-12:
            # Exact match — no interpolation needed.
            optimal_range = table[idxs[0], 3] * self.planet.radius
            gamma = table[idxs[0], 2]
        else:
            weights = 1.0 / dists
            weights /= weights.sum()
            optimal_range = float(np.dot(weights, table[idxs, 3])) * self.planet.radius
            gamma = float(np.dot(weights, table[idxs, 2]))

        if range_to_target <= optimal_range:
            logger["Guidance"].debug(
                f"Old gamma: {missile_gamma:.4f} rad, optimal gamma: {gamma:.4f} rad, range to target: {range_to_target:.2f} m, optimal range: {optimal_range:.2f} m."
            )
            # Look up the gamma needed to reach the target range from current (alt, vel).
            # This is the steering target: different from the gamma-based lookup above which
            # only tells us what range the missile's *current* gamma would achieve.
            range_query = np.array(
                [
                    altitude / self.ballistic_guidance.alt_scale,
                    velocity / self.ballistic_guidance.vel_scale,
                    (range_to_target / self.planet.radius) / self.ballistic_guidance.range_scale,
                ]
            )
            r_dists, r_idxs = self.ballistic_guidance.kdtree_range.query(range_query, k=2)
            if r_dists[0] < 1e-12:
                gamma = table[r_idxs[0], 2]
            else:
                r_weights = 1.0 / r_dists
                r_weights /= r_weights.sum()
                gamma = float(np.dot(r_weights, table[r_idxs, 2]))

            self.state = "ballistic"

            logger["Guidance"].debug(f"New gamma from secondary KDTREE: {gamma:.4f} rad")

            logger["Guidance"].debug(
                f"Switch range error: {(range_to_target - optimal_range)/1000:.2f} km, gamma error: {missile_gamma - gamma:.2f} rad."
            )

        # Convert table gamma (prograde convention) back to the local t_hat convention
        # before passing to gravity_turn_direction.
        # 2: Aggressiveness factor to ensure the missile gets in range, was tuned empirically.
        theta = self._t_hat_sign * gamma * missile.burned_fraction * 2

        direction = np.cos(theta) * r_hat + np.sin(theta) * t_hat

        # direction = self.gravity_turn_direction(missile, self._t_hat_sign * gamma)
        return GuidanceResults(
            direction=direction,
            state=self.state,
            gamma=gamma,
        )


class RCSGuidance(Guidance):
    """Simple guidance that uses RCS thrusters to always point directly at the target, without any powered flight phase."""

    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        v_norm = np.linalg.norm(missile.velocity)
        if v_norm < 1e-8:
            return GuidanceResults(direction=np.zeros(3), state=self.state)
        v_hat = missile.velocity / v_norm

        los = self.target.position - missile.position

        # Remove the component along the current velocity so thrust is purely a
        # course correction (perpendicular to flight path).  This makes guidance
        # stable at any thrust level: higher thrust curves the trajectory more
        # sharply toward the target instead of causing downrange overshoot.
        correction = los - np.dot(los, v_hat) * v_hat
        norm = np.linalg.norm(correction)
        if norm < 1e-8:
            return GuidanceResults(direction=np.zeros(3), state=self.state)
        return GuidanceResults(direction=correction / norm, state=self.state)


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
        planet,
        target: MovableObj,
        N: float = 4.0,
        activation_altitude_km: float | None = 300.0,
        activation_range_km: float | None = None,
    ):
        super().__init__(planet, target)
        self.N = N
        self.activation_altitude_m = activation_altitude_km * 1000.0 if activation_altitude_km is not None else None
        self.activation_range_m = activation_range_km * 1000.0 if activation_range_km is not None else None
        self._prev_los_hat: NDArray | None = None
        self._prev_t: float | None = None
        self._armed: bool = False

    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        los = self.target.position - missile.position
        los_norm = np.linalg.norm(los)
        if los_norm < 1e-8:
            return GuidanceResults(direction=np.zeros(3), state=self.state)
        los_hat = los / los_norm

        v_norm = np.linalg.norm(missile.velocity)
        if v_norm < 1e-8:
            return GuidanceResults(direction=np.zeros(3), state=self.state)
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
                    f"PN armed at altitude {altitude/1000:.1f} km, range {surface_range/1000:.1f} km."
                )
            else:
                return GuidanceResults(direction=np.zeros(3), state=self.state)

        # On first call after arming, seed the LOS history.
        if self._prev_los_hat is None or self._prev_t is None:
            self._prev_los_hat = los_hat.copy()
            self._prev_t = t
            return GuidanceResults(direction=np.zeros(3), state=self.state)

        dt = t - self._prev_t
        if dt < 1e-9:
            self._prev_los_hat = los_hat.copy()
            self._prev_t = t
            return GuidanceResults(direction=np.zeros(3), state=self.state)

        # LOS rate vector (rad/s): d(los_hat)/dt projected perpendicular to los_hat
        d_los_hat = (los_hat - self._prev_los_hat) / dt
        # Remove component along los_hat to get pure angular rate
        los_rate = d_los_hat - np.dot(d_los_hat, los_hat) * los_hat

        self._prev_los_hat = los_hat.copy()
        self._prev_t = t

        los_rate_norm = np.linalg.norm(los_rate)
        if los_rate_norm < 1e-12:
            return GuidanceResults(direction=np.zeros(3), state=self.state)

        # Closing velocity (positive when approaching)
        v_c = -np.dot(missile.velocity, los_hat)

        # PN command: a = N * v_c * los_rate_hat, in the plane perpendicular to LOS.
        # Then project onto the plane perpendicular to velocity so RCS doesn't brake.
        a_cmd = self.N * v_c * los_rate / los_rate_norm
        a_cmd = a_cmd - np.dot(a_cmd, v_hat) * v_hat

        cmd_norm = np.linalg.norm(a_cmd)
        if cmd_norm < 1e-8:
            return GuidanceResults(direction=np.zeros(3), state=self.state)

        return GuidanceResults(direction=a_cmd / cmd_norm, state=self.state, magnitude=float(cmd_norm))
