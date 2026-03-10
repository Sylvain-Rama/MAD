from mad.objs.common_schemas import MovableObject


from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class Guidance(ABC):
    # Any Guidance class should return the direction as a NDArray of same shape of position or velocity.

    @abstractmethod
    def get_guidance(self, missile) -> NDArray:
        pass


class ClosedFormBallistic(Guidance):

    def __init__(self, planet, target: "MovableObject"):
        self.planet = planet
        self.target = target

    def local_frame(self, missile) -> tuple[NDArray[np.floating], NDArray[np.floating]]:

        r_hat = missile.norm
        target_hat = self.target.norm
        delta = self.target.position - missile.position

        if r_hat.size == 2:
            # 2D tangent: rotate 90° CCW
            t_hat = np.array([-r_hat[1], r_hat[0]])
            # Ensure t_hat points toward target
            if np.dot(t_hat, delta) < 0:
                t_hat = -t_hat
        else:
            # 3D tangent: great-circle tangent
            plane_normal = np.cross(r_hat, target_hat)
            t_hat = np.cross(plane_normal, r_hat)

        t_norm = np.linalg.norm(t_hat)
        if t_norm < 1e-8:
            return r_hat, np.zeros_like(r_hat)

        return r_hat, t_hat / t_norm

    def optimal_gamma(self, missile, sigma: float) -> float:

        v = np.linalg.norm(missile.velocity)
        r = np.linalg.norm(missile.position)
        return np.arctan((v**2 - self.planet.mu / r) / v**2 * np.tan(sigma / 2))

    def gravity_turn_direction(self, missile, optimal_gamma: float) -> NDArray[np.floating]:

        r_hat, t_hat = self.local_frame(missile)

        # Construct thrust vector along tangent + radial
        d = np.cos(optimal_gamma) * t_hat - np.sin(optimal_gamma) * r_hat
        return d / np.linalg.norm(d)

    def get_guidance(self, missile) -> NDArray[np.floating]:

        sigma = missile.central_angle(self.target)
        gamma = self.optimal_gamma(missile, sigma)
        return self.gravity_turn_direction(missile, gamma)
