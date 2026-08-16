"""Guidance laws for satellites and launch vehicles."""

import numpy as np
from numpy.typing import NDArray
from typing import Callable

from mad.objs import MovableObj
from mad.guidances.base_guidances import (
    Guidance,
    GuidableObj,
    GuidanceResults,
    GuidanceStates,
    GuidanceInterrupts,
    GuidanceManager,
    StraightUp,
)
from mad.guidances.interrupt_guidances import interrupt_at_altitude
from mad.utils.logger import SourceLogger

logger = SourceLogger()


class _ProgradeTrackingGuidance(Guidance):
    """Base for orbital guidance laws that track the prograde (tangential) direction."""

    def __init__(
        self,
        planet,
        target: MovableObj | None = None,
        interrupt_fn: Callable[["GuidanceInterrupts"], bool] | None = None,
    ):
        super().__init__(planet, target=target, interrupt_fn=interrupt_fn)  # type: ignore[arg-type]
        self._prograde_hat: NDArray | None = None

    def _resolve_t_hat(self, missile: GuidableObj, r_hat: NDArray) -> NDArray:
        """Return the tangential unit vector that defines the pitch-over plane.

        Priority:
        1. Great-circle direction toward ``self.target`` (when a target is set).
        2. Horizontal component of the vehicle's current velocity.
        3. Eastward direction (first-call fallback when no horizontal speed yet).
        """
        if self.target is not None:
            _, t_hat = self.local_frame(missile)
            if np.linalg.norm(t_hat) > 1e-8:
                return self._resolve_t_hat_sign(r_hat, t_hat) * t_hat

        v_horiz = missile.velocity - np.dot(missile.velocity, r_hat) * r_hat
        v_horiz_mag = np.linalg.norm(v_horiz)

        if v_horiz_mag > 1.0:
            self._prograde_hat = v_horiz / v_horiz_mag
        elif self._prograde_hat is None:
            north = np.array([0.0, 0.0, 1.0])
            east = np.cross(north, r_hat)
            east_mag = np.linalg.norm(east)
            self._prograde_hat = east / east_mag if east_mag > 1e-8 else np.array([1.0, 0.0, 0.0])

        return self._prograde_hat  # type: ignore[return-value]


class CosinePitchProgram(_ProgradeTrackingGuidance):
    """Smoothly pitches the vehicle from vertical to horizontal using a cosine-ease schedule."""

    def __init__(
        self,
        planet,
        target: MovableObj | None = None,
        min_turn_altitude_m: float = 0.0,
        turn_end_altitude_m: float = 0.0,
        interrupt_fn: Callable[["GuidanceInterrupts"], bool] | None = None,
    ):
        super().__init__(planet, target=target, interrupt_fn=interrupt_fn)
        self.min_turn_altitude_m = min_turn_altitude_m
        self.turn_end_altitude_m = turn_end_altitude_m

    def _compute_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        r_hat = missile.pos_norm
        altitude = np.linalg.norm(missile.position) - self.planet.radius
        t_hat = self._resolve_t_hat(missile, r_hat)

        turn_range = self.turn_end_altitude_m - self.min_turn_altitude_m
        if turn_range > 0:
            progress = float(np.clip((altitude - self.min_turn_altitude_m) / turn_range, 0.0, 1.0))
        else:
            progress = 1.0

        theta = (np.pi / 2.0) * (1.0 - np.cos(np.pi * progress)) / 2.0
        direction = np.cos(theta) * r_hat + np.sin(theta) * t_hat
        return GuidanceResults(
            direction=direction / np.linalg.norm(direction),
            state=self.state,
            next_guidance=self.next_guidance,
        )


class OrbitalInsertion(_ProgradeTrackingGuidance):
    """Burns tangentially to reach orbital speed at the target perigee altitude, then releases the payload."""

    def __init__(
        self,
        planet,
        perigee_altitude_m: float,
        apogee_altitude_m: float | None = None,
        target: MovableObj | None = None,
        altitude_tol_m: float | None = None,
        interrupt_fn: Callable[["GuidanceInterrupts"], bool] | None = None,
    ):
        super().__init__(planet, target=target, interrupt_fn=interrupt_fn)
        self.perigee_altitude_m = perigee_altitude_m
        self.perigee_radius_m = planet.radius + perigee_altitude_m
        self.altitude_tol_m = altitude_tol_m if altitude_tol_m is not None else 0.05 * perigee_altitude_m

        # Target orbital speed at perigee.
        # Circular:    v = √(μ / r_p)
        # Elliptical:  v = √(μ · (2/r_p − 1/a)),  a = (r_p + r_a) / 2  [vis-viva]
        if apogee_altitude_m is not None:
            r_a = planet.radius + apogee_altitude_m
            semi_major_axis = (self.perigee_radius_m + r_a) / 2.0
            self._v_target = float(np.sqrt(planet.mu * (2.0 / self.perigee_radius_m - 1.0 / semi_major_axis)))
        else:
            self._v_target = float(np.sqrt(planet.mu / self.perigee_radius_m))

    def _compute_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        r_hat = missile.pos_norm
        altitude = np.linalg.norm(missile.position) - self.planet.radius
        t_hat = self._resolve_t_hat(missile, r_hat)
        v_horiz_mag = abs(np.dot(missile.velocity, t_hat))

        # Fallback: all propellant spent — release at best available point.
        # NOTE: use has_thrust (stages list empty) rather than burned_fraction, which
        # is an imprecise formula that can return >= 1.0 a few seconds early.
        # NOTE: 80 km is a rough "above the atmosphere" threshold. By that altitude
        # any v_r ≤ 0 crossing is almost certainly a genuine apogee rather than a
        # pitch-programme artefact.
        if not missile.has_thrust:
            v_r = np.dot(missile.velocity, r_hat)
            at_target_band = altitude >= self.perigee_altitude_m - self.altitude_tol_m
            at_apogee = v_r <= 0.0 and altitude > 80_000.0
            if at_target_band or at_apogee:
                logger["Guidance"].info(
                    f"{t:<.2f}s - {missile.name} All propellant spent at altitude {altitude / 1e3:.1f} km, "
                    f"v_horiz = {v_horiz_mag:.1f} m/s (target {self._v_target:.1f} m/s). Releasing payload."
                )
                self.state = GuidanceStates.RELEASE_PAYLOAD
                return GuidanceResults(
                    direction=np.zeros(3),
                    state=self.state,
                    release_velocity=missile.velocity.copy(),
                )

        if v_horiz_mag >= 0.99 * self._v_target:
            logger["Guidance"].info(
                f"{t:<.2f}s - {missile.name} Orbit insertion achieved at altitude {altitude / 1e3:.1f} km, "
                f"v_horiz = {v_horiz_mag:.1f} m/s (target {self._v_target:.1f} m/s)."
            )
            self.state = GuidanceStates.RELEASE_PAYLOAD
            return GuidanceResults(
                direction=np.zeros(3),
                state=self.state,
                release_velocity=missile.velocity.copy(),
            )

        return GuidanceResults(direction=t_hat.copy(), state=self.state, next_guidance=self.next_guidance)


def LEOInsertionGuidance(
    planet,
    perigee_altitude_m: float,
    apogee_altitude_m: float | None = None,
    target: MovableObj | None = None,
    min_turn_altitude_m: float = 0.0,
    turn_end_altitude_m: float | None = None,
    altitude_tol_m: float | None = None,
    interrupt_fn: Callable[["GuidanceInterrupts"], bool] | None = None,
) -> GuidanceManager:
    """Assemble a LEO insertion guidance sequence as a Class."""
    turn_end = turn_end_altitude_m if turn_end_altitude_m is not None else 0.8 * perigee_altitude_m
    tol = altitude_tol_m if altitude_tol_m is not None else 0.05 * perigee_altitude_m

    vertical_rise = StraightUp(
        planet=planet,
        target=target,  # type: ignore[arg-type]
        interrupt_fn=lambda i: interrupt_at_altitude(i, min_turn_altitude_m),
    )
    pitch_program = CosinePitchProgram(
        planet=planet,
        target=target,
        min_turn_altitude_m=min_turn_altitude_m,
        turn_end_altitude_m=turn_end,
        interrupt_fn=lambda i: interrupt_at_altitude(i, perigee_altitude_m - tol),
    )
    orbital_insertion = OrbitalInsertion(
        planet=planet,
        perigee_altitude_m=perigee_altitude_m,
        apogee_altitude_m=apogee_altitude_m,
        target=target,
        altitude_tol_m=tol,
        interrupt_fn=interrupt_fn,
    )
    return GuidanceManager([vertical_rise, pitch_program, orbital_insertion])


class RCSGuidance(Guidance):
    """Simple guidance that uses RCS thrusters to always point directly at the target, without any powered flight phase."""

    def _compute_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
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
