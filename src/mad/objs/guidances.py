from mad.objs.common_schemas import MovableObj

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from mad.logger import SourceLogger
from mad.utils import load_ballistic_table

if TYPE_CHECKING:
    from mad.objs.missiles import BallisticMissile

logger = SourceLogger()


@dataclass
class GuidanceResults:
    direction: NDArray
    state: str


class Guidance(ABC):
    # Any Guidance class should return the direction as a NDArray of same shape of position or velocity.
    def __init__(self, planet, target: "MovableObj"):
        self.planet = planet
        self.target = target

    @staticmethod
    def central_angle(missile: "BallisticMissile", target: MovableObj) -> NDArray:
        return np.arccos(np.clip(np.dot(missile.normalize, target.normalize), -1, 1))

    def local_frame(self, missile: "BallisticMissile") -> tuple[NDArray, NDArray]:
        r_hat = missile.normalize
        rt_hat = self.target.normalize

        t_hat = np.cross(np.cross(rt_hat, r_hat), r_hat)
        t_norm = np.linalg.norm(t_hat)
        if t_norm < 1e-8:
            return r_hat, np.zeros_like(r_hat)

        t_hat /= t_norm
        return r_hat, t_hat

    def optimal_gamma(self, missile: "BallisticMissile", sigma: NDArray) -> NDArray:
        v = np.linalg.norm(missile.velocity)
        gamma = np.arctan((v**2 - self.planet.mu / np.linalg.norm(missile.position)) / v**2 * np.tan(sigma / 2))
        return gamma

    @abstractmethod
    def get_guidance(self, missile: "BallisticMissile", t: float = 0.0) -> GuidanceResults:
        pass


class GravityTurn(Guidance):
    """Gravity turn: the rocket starts vertically and gradually turns towards the target, following a smooth curve.
    The optimal curve is computed based on the current velocity and the central angle to the target."""

    def __init__(self, planet, target: "MovableObj"):
        super().__init__(planet, target)
        self.state = "powered"

    def gravity_turn_direction(
        self,
        missile: "BallisticMissile",
        optimal_gamma: NDArray,
    ):
        r_hat, t_hat = self.local_frame(missile)
        theta = optimal_gamma * missile.burned_fraction

        d = np.cos(theta) * r_hat + np.sin(theta) * t_hat
        return d / np.linalg.norm(d)

    def get_guidance(self, missile: "BallisticMissile", t: float = 0.0) -> GuidanceResults:
        sigma = self.central_angle(missile, self.target)
        gamma = self.optimal_gamma(missile, sigma)

        return GuidanceResults(direction=self.gravity_turn_direction(missile, gamma), state=self.state)


class ClosedFormBallistic(Guidance):
    """Closed-form ballistic guidance: the missile starts vertically and stops thrusting at the optimal point,
    then follows a ballistic trajectory to the target. The optimal point is computed based on the current velocity
    and the central angle to the target.
    """

    def __init__(self, planet, target: "MovableObj"):
        super().__init__(planet, target)
        self.state = "powered"

    def set_flight_phase(self, missile: "BallisticMissile", gamma: NDArray, sigma: NDArray, t: float) -> None:

        # Compute the optimal angle (and corresponding arc-length distance) to stop thrusting
        # based on the current velocity and central angle.
        r = np.linalg.norm(missile.position)
        v = np.linalg.norm(missile.velocity)
        optimal_angle = 2 * np.arctan(
            v**2 * np.sin(gamma) * np.cos(gamma) / (self.planet.mu / r - v**2 * np.sin(gamma) ** 2)
        )

        optimal_angle = np.linalg.norm(optimal_angle)
        optimal_distance = self.planet.radius * optimal_angle

        # Use only the tangential (horizontal) component of the remaining distance so that
        # radial ascent — which doesn't change the downrange position — cannot inflate the
        # metric and prevent the threshold from being reached.
        _, t_hat = self.local_frame(missile)
        tangential_distance = np.abs(np.dot(self.target.position - missile.position, t_hat))

        if tangential_distance <= optimal_distance:
            self.state = "ballistic"
            logger["Physics"].info(f"{missile.name} switched to ballistic phase at distance {optimal_distance:.2f} m.")
        elif sigma <= optimal_angle:
            self.state = "ballistic"
            logger["Physics"].info(
                f"{missile.name} switched to ballistic phase at central angle {np.degrees(sigma):.2f} deg."
            )

    def gravity_turn_direction(
        self,
        missile: "BallisticMissile",
        optimal_gamma: NDArray,
    ):
        r_hat, t_hat = self.local_frame(missile)
        theta = optimal_gamma * missile.burned_fraction

        d = np.cos(theta) * r_hat + np.sin(theta) * t_hat
        return d / np.linalg.norm(d)

    def get_guidance(self, missile: "BallisticMissile", t: float = 0.0) -> GuidanceResults:
        sigma = self.central_angle(missile, self.target)
        gamma = self.optimal_gamma(missile, sigma)
        direction = self.gravity_turn_direction(missile, gamma)
        self.set_flight_phase(missile, gamma, sigma, t)

        return GuidanceResults(direction=direction, state=self.state)


class RangeGuided(Guidance):
    """The guidance returns an updated when the missile get in range to the target, according to the ballistic table.
    The ballistic table is a CSV file with columns: altitude_m, velocity_m_s, gamma_rad, range_rad.
    """

    def __init__(self, planet, target: "MovableObj", ballistic_table_path: str):
        super().__init__(planet, target)
        self.state = "powered"
        if ballistic_table_path is not None:
            self.ballistic_table = load_ballistic_table(ballistic_table_path)
            self.alt_scale = np.ptp(self.ballistic_table[:, 0]) or 1.0
            self.vel_scale = np.ptp(self.ballistic_table[:, 1]) or 1.0
            self.gam_scale = np.ptp(self.ballistic_table[:, 2]) or 1.0
        else:
            self.ballistic_table = None
            self.alt_scale = 1.0
            self.vel_scale = 1.0
            self.gam_scale = 1.0
        # Sign convention: +1 if local t_hat is prograde (toward target), -1 if retrograde.
        # Resolved once on the first get_guidance call.
        self._t_hat_sign: float | None = None

    def gravity_turn_direction(
        self,
        missile: "BallisticMissile",
        optimal_gamma: NDArray,
    ):
        r_hat, t_hat = self.local_frame(missile)
        theta = optimal_gamma * missile.burned_fraction

        d = np.cos(theta) * r_hat + np.sin(theta) * t_hat
        return d / np.linalg.norm(d)

    def get_guidance(self, missile: "BallisticMissile", t: float = 0.0) -> GuidanceResults:

        if self.ballistic_table is None:
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
        table = self.ballistic_table

        v_r = np.dot(missile.velocity, r_hat)
        v_t = np.dot(missile.velocity, t_hat)
        missile_gamma = np.arctan2(v_r, self._t_hat_sign * v_t)

        idx = np.argmin(
            np.sqrt(
                ((table[:, 0] - altitude) / self.alt_scale) ** 2
                + ((table[:, 1] - velocity) / self.vel_scale) ** 2
                + ((table[:, 2] - missile_gamma) / self.gam_scale) ** 2
            )
        )
        optimal_range = table[idx, 3] * self.planet.radius
        optimal_gamma = table[idx, 2]

        if range_to_target <= optimal_range:
            self.state = "ballistic"
            logger["Guidance"].info(
                f"{missile.name} switched to ballistic phase at range {range_to_target:.2f} m (optimal: {optimal_range:.2f} m)."
            )

        # Convert table gamma (prograde convention) back to the local t_hat convention
        # before passing to gravity_turn_direction.
        direction = self.gravity_turn_direction(missile, self._t_hat_sign * optimal_gamma)
        return GuidanceResults(direction=direction, state=self.state)
