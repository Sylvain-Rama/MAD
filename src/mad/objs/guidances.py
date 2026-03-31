from mad.objs.common_schemas import MovableObject


from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from mad.logger import SourceLogger

logger = SourceLogger()


class Guidance(ABC):
    # Any Guidance class should return the direction as a NDArray of same shape of position or velocity.

    @abstractmethod
    def get_guidance(self, missile) -> NDArray:
        pass


class GravityTurn(Guidance):

    def __init__(self, planet, target: "MovableObject"):
        self.planet = planet
        self.target = target

    @staticmethod
    def central_angle(missile: MovableObject, target: MovableObject) -> NDArray:
        return np.arccos(np.clip(np.dot(missile.normalize, target.normalize), -1, 1))

    def local_frame(self, missile: "MovableObject") -> tuple[NDArray, NDArray]:
        r_hat = missile.normalize
        rt_hat = self.target.normalize

        t_hat = np.cross(np.cross(rt_hat, r_hat), r_hat)
        t_norm = np.linalg.norm(t_hat)
        if t_norm < 1e-8:
            return r_hat, np.zeros_like(r_hat)

        t_hat /= t_norm
        return r_hat, t_hat

    def optimal_gamma(self, missile: MovableObject, sigma: NDArray) -> NDArray:

        gamma = np.arctan(
            (missile.velocity**2 - self.planet.mu / np.linalg.norm(missile.position))
            / missile.velocity**2
            * np.tan(sigma / 2)
        )

        return gamma

    def gravity_turn_direction(
        self,
        missile: MovableObject,
        optimal_gamma: NDArray,
    ):

        r_hat, t_hat = self.local_frame(missile)

        # smooth rotation from vertical to target direction
        theta = optimal_gamma * missile.burned_fraction

        d = np.cos(theta) * r_hat + np.sin(theta) * t_hat
        return d / np.linalg.norm(d)

    def get_guidance(self, missile: MovableObject) -> NDArray:

        sigma = self.central_angle(missile, self.target)
        gamma = self.optimal_gamma(missile, sigma)

        return self.gravity_turn_direction(missile, gamma)
