from mad.objs.common_schemas import MovableObject


from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from mad.logger import SourceLogger

if TYPE_CHECKING:
    from mad.objs.missiles import BallisticMissile

logger = SourceLogger()


class Guidance(ABC):
    # Any Guidance class should return the direction as a NDArray of same shape of position or velocity.
    def __init__(self, planet, target: "MovableObject"):
        self.planet = planet
        self.target = target

    @staticmethod
    def central_angle(missile: "BallisticMissile", target: MovableObject) -> NDArray:
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
        gamma = np.arctan(
            (missile.velocity**2 - self.planet.mu / np.linalg.norm(missile.position))
            / missile.velocity**2
            * np.tan(sigma / 2)
        )
        return gamma


    @abstractmethod
    def get_guidance(self, missile: "BallisticMissile") -> NDArray:
        pass


class GravityTurn(Guidance):
    """Gravity turn: the rocket starts vertically and gradually turns towards the target, following a smooth curve. 
    The optimal curve is computed based on the current velocity and the central angle to the target."""

    def __init__(self, planet, target: "MovableObject"):
        super().__init__(planet, target)

    def gravity_turn_direction(
        self,
        missile: "BallisticMissile",
        optimal_gamma: NDArray,
    ):
        r_hat, t_hat = self.local_frame(missile)
        theta = optimal_gamma * missile.burned_fraction

        d = np.cos(theta) * r_hat + np.sin(theta) * t_hat
        return d / np.linalg.norm(d)

    def get_guidance(self, missile: "BallisticMissile") -> NDArray:
        sigma = self.central_angle(missile, self.target)
        gamma = self.optimal_gamma(missile, sigma)

        return self.gravity_turn_direction(missile, gamma)


class ClosedFormBallistic(Guidance):
    """Closed-form ballistic guidance: the missile starts vertically and stops thrusting at the optimal point, 
    then follows a ballistic trajectory to the target. The optimal point is computed based on the current velocity 
    and the central angle to the target.
    """

    def __init__(self, planet, target: "MovableObject"):
            super().__init__(planet, target)
            self.state = "powered"
             

    def set_flight_phase(self, missile: "BallisticMissile", gamma: NDArray) -> None:
    
        # Compute the optimal distance to stop thrusting based on the current velocity and central angle.
        optimal_distance = self.planet.radius * 2 * np.arctan(
            (0.8 * missile.deltav) ** 2 * np.sin(gamma) * np.cos(gamma)
            / (self.planet.mu / self.planet.radius - (0.8 * missile.deltav) ** 2 * np.sin(gamma) ** 2)
        )

        optimal_distance = np.linalg.norm(optimal_distance)

        if self.state == "powered" and self.planet.surface_distance(missile, self.target) <= optimal_distance:
            self.state = "ballistic"
            logger["Physics"].info(f"{missile.name} switched to ballistic phase at distance {optimal_distance:.2f} m.")

    def get_guidance(self, missile: "BallisticMissile") -> NDArray:
        sigma = self.central_angle(missile, self.target)
        gamma = self.optimal_gamma(missile, sigma)
        self.set_flight_phase(missile, gamma)

        return gamma 