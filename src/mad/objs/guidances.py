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


class ClosedFormBallistic(Guidance):

    def __init__(self, planet, target: "MovableObject"):
        self.planet = planet
        self.target = target

    @staticmethod
    def central_angle(missile: MovableObject, target: MovableObject) -> NDArray:
        r1 = missile.position / np.linalg.norm(missile.position)
        r2 = target.position / np.linalg.norm(target.position)

        return np.arccos(np.clip(np.dot(r1, r2), -1, 1))

    def local_frame(self, missile: "MovableObject") -> tuple[NDArray, NDArray]:
        r_hat = missile.position / np.linalg.norm(missile.position)

        # local horizontal (90° rotation)
        t_hat = np.array([-r_hat[1], r_hat[0]])

        # ensure tangent points toward target
        if np.dot(t_hat, missile.position - self.target.position) < 0:
            t_hat = -t_hat

        return r_hat, t_hat

    def optimal_gamma(self, missile: MovableObject, sigma: NDArray) -> NDArray:

        gamma = np.arctan(
            (missile.velocity**2 - self.planet.mu / np.linalg.norm(missile.position))
            / missile.velocity**2
            * np.tan(sigma / 2)
        )
        # gamma = np.clip(gamma, 0.0, np.pi / 2)

        return gamma

    def gravity_turn_direction(
        self,
        missile: MovableObject,
        optimal_gamma: NDArray,
    ):

        r_hat, t_hat = self.local_frame(missile)
        f = missile.burned_fraction

        if f < 0.1:
            theta = 0.0
        else:
            theta = optimal_gamma * (f - 0.1) / 0.9

        # smooth rotation from vertical to target direction
        # theta = optimal_gamma * missile.burned_fraction

        d = np.cos(theta) * r_hat + np.sin(theta) * t_hat
        return d / np.linalg.norm(d)

    def get_guidance(self, missile: MovableObject) -> NDArray:

        sigma = self.central_angle(missile, self.target)
        gamma = self.optimal_gamma(missile, sigma)

        return self.gravity_turn_direction(missile, gamma)
