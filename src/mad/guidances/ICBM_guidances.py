from mad.objs.base import MovableObj
from mad.guidances.base_guidances import Guidance, GuidableObj, GuidanceResults, GuidanceStates


import numpy as np

from mad.utils.logger import SourceLogger
from mad.utils.ballistic_tables import load_ballistic_table

logger = SourceLogger()


class TabulatedBallistic(Guidance):
    """The guidance returns an updated when the missile get in range to the target, according to the ballistic table.
    The ballistic table is a CSV file with columns: altitude_m, velocity_m_s, gamma_rad, range_rad.
    """

    def __init__(self, planet, target: MovableObj, ballistic_table_path: str):
        super().__init__(planet, target)
        self.ballistic_guidance = load_ballistic_table(ballistic_table_path) if ballistic_table_path else None

    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:

        if self.ballistic_guidance is None:
            logger["Guidance"].error("Ballistic table not loaded. Cannot compute guidance.")
            return GuidanceResults(direction=np.zeros(3), state=self.state)

        r_hat, t_hat = self.local_frame(missile)
        sign = self._resolve_t_hat_sign(r_hat, t_hat)

        sigma = self.central_angle(missile, self.target)
        range_to_target = self.planet.radius * sigma

        altitude = np.linalg.norm(missile.position) - self.planet.radius
        velocity = np.linalg.norm(missile.velocity)

        # Find the closest entry in the ballistic table based on altitude, velocity and gamma.
        # Convert missile gamma to the table's prograde convention using the detected sign.
        table = self.ballistic_guidance.table

        v_r = np.dot(missile.velocity, r_hat)
        v_t = np.dot(missile.velocity, t_hat)
        missile_gamma = np.arctan2(v_r, sign * v_t)

        query_point = np.array(
            [
                altitude / self.ballistic_guidance.alt_scale,
                velocity / self.ballistic_guidance.vel_scale,
                missile_gamma / self.ballistic_guidance.gam_scale,
            ]
        )
        k = min(5, len(table))
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

        release_velocity = None
        if range_to_target <= optimal_range:
            # TODO: Continue correction for final approach.
            # See coasting phase for last missile stage.

            self.state = GuidanceStates.RELEASE_PAYLOAD

            # Compute the optimal RV release velocity: same speed as the missile but
            # aligned to the table's optimal gamma so the RV follows the correct ballistic arc.
            v_mag = np.linalg.norm(missile.velocity)
            release_velocity = v_mag * (np.sin(gamma) * r_hat + sign * np.cos(gamma) * t_hat)

        # Convert table gamma (prograde convention) back to the local t_hat convention
        # before passing to gravity_turn_direction.
        # 2: Aggressiveness factor to ensure the missile gets in range, was tuned empirically.
        # Should disappear the day we have coasting phase.
        theta = sign * gamma * missile.burned_fraction * 2

        direction = np.cos(theta) * r_hat + np.sin(theta) * t_hat

        return GuidanceResults(
            direction=direction,
            state=self.state,
            gamma=gamma,
            release_velocity=release_velocity,
        )
