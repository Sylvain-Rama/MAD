"""Guidance laws for ICBMs & rockets."""

from mad.objs import MovableObj
from mad.guidances import Guidance, GuidableObj, GuidanceResults, GuidanceStates, GuidanceInterrupts
from mad.objs.planets import Planet
from typing import Callable


import numpy as np

from mad.utils.logger import SourceLogger
from mad.utils.ballistic_tables import load_ballistic_table

logger = SourceLogger()


class TabulatedBallistic(Guidance):
    """The guidance returns an updated when the missile get in range to the target, according to the ballistic table.
    The ballistic table is a CSV file with columns: altitude_m, velocity_m_s, gamma_rad, range_rad.
    """

    def __init__(
        self,
        reference_planet: Planet,
        target: MovableObj,
        ballistic_table_name: str,
        interrupt_fn: Callable[[GuidanceInterrupts], bool] | None = None,
    ):
        super().__init__(reference_planet, target, interrupt_fn=interrupt_fn)
        self.ballistic_guidance = load_ballistic_table(ballistic_table_name)

    def __repr__(self):
        max_alt, min_alt = max(self.ballistic_guidance.altitudes), min(self.ballistic_guidance.altitudes)
        max_vel, min_vel = max(self.ballistic_guidance.velocities), min(self.ballistic_guidance.velocities)
        max_gamma, min_gamma = max(self.ballistic_guidance.gammas), min(self.ballistic_guidance.gammas)
        return (
            f"TabulatedBallistic(table name={self.ballistic_guidance.name}, "
            f"altitude range=[{min_alt:.1f}, {max_alt:.1f}] m, "
            f"velocity range=[{min_vel:.1f}, {max_vel:.1f}] m/s, "
            f"degrees range=[{np.degrees(min_gamma):.3f}, {np.degrees(max_gamma):.3f}] deg)"
        )

    def _compute_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:

        if self.ballistic_guidance is None:
            logger["Guidance"].error("Ballistic table not loaded. Cannot compute guidance.")
            return GuidanceResults(direction=np.zeros(3), state=self.state)

        r_hat, t_hat = self.local_frame(missile)
        sign = self._resolve_t_hat_sign(r_hat, t_hat)

        sigma = self.central_angle(missile, self.target)
        range_to_target = self.reference_planet.radius * sigma

        altitude = np.linalg.norm(missile.position) - self.reference_planet.radius
        velocity = np.linalg.norm(missile.velocity)

        # Convert missile gamma to the table's prograde convention using the detected sign.
        v_r = np.dot(missile.velocity, r_hat)
        v_t = np.dot(missile.velocity, t_hat)
        missile_gamma = np.arctan2(v_r, sign * v_t)

        # Linearly interpolate the optimal range over the table's regular
        # (altitude, velocity, gamma) grid. Clamp to the grid envelope since gamma is
        # itself a grid axis, so the clamped query value doubles as the table's gamma.
        table = self.ballistic_guidance
        clamped_altitude = np.clip(altitude, table.altitudes[0], table.altitudes[-1])
        clamped_velocity = np.clip(velocity, table.velocities[0], table.velocities[-1])
        gamma = float(np.clip(missile_gamma, table.gammas[0], table.gammas[-1]))

        if velocity > table.velocities[-1] or velocity < table.velocities[0]:
            logger["Guidance"].warning(
                f"{missile.name}: velocity {velocity:.1f} m/s is outside the ballistic table's "
                f"range [{table.velocities[0]:.1f}, {table.velocities[-1]:.1f}] m/s; clamping to "
                f"{clamped_velocity:.1f} m/s. The table likely needs a wider velocity grid."
            )

        optimal_range = (
            float(table.range_interp([clamped_altitude, clamped_velocity, gamma])[0]) * self.reference_planet.radius
        )

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
