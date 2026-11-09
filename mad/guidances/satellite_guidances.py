"""Guidance laws for satellites and launch vehicles."""

import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, cast
from mad.objs.planets import Planet
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
        reference_planet: Planet,
        target: MovableObj | None = None,
        interrupt_fn: Callable[["GuidanceInterrupts"], bool] | None = None,
    ):
        super().__init__(reference_planet=reference_planet, target=target, interrupt_fn=interrupt_fn)  # type: ignore[arg-type]
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
        reference_planet: Planet,
        target: MovableObj | None = None,
        min_turn_altitude_m: float = 0.0,
        turn_end_altitude_m: float = 0.0,
        interrupt_fn: Callable[["GuidanceInterrupts"], bool] | None = None,
    ):
        super().__init__(reference_planet=reference_planet, target=target, interrupt_fn=interrupt_fn)
        self.min_turn_altitude_m = min_turn_altitude_m
        self.turn_end_altitude_m = turn_end_altitude_m

    def _compute_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        r_hat = missile.pos_norm
        altitude = np.linalg.norm(missile.position) - self.reference_planet.radius
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
        reference_planet: Planet,
        perigee_altitude_m: float,
        apogee_altitude_m: float | None = None,
        target: MovableObj | None = None,
        altitude_tol_m: float | None = None,
        interrupt_fn: Callable[["GuidanceInterrupts"], bool] | None = None,
    ):
        super().__init__(reference_planet=reference_planet, target=target, interrupt_fn=interrupt_fn)
        self.perigee_altitude_m = perigee_altitude_m
        self.perigee_radius_m = self.reference_planet.radius + perigee_altitude_m
        self.altitude_tol_m = altitude_tol_m if altitude_tol_m is not None else 0.05 * perigee_altitude_m

        # Target orbital speed at perigee.
        # Circular:    v = √(μ / r_p)
        # Elliptical:  v = √(μ · (2/r_p − 1/a)),  a = (r_p + r_a) / 2  [vis-viva]
        if apogee_altitude_m is not None:
            r_a = self.reference_planet.radius + apogee_altitude_m
            semi_major_axis = (self.perigee_radius_m + r_a) / 2.0
            self._v_target = float(
                np.sqrt(self.reference_planet.mu * (2.0 / self.perigee_radius_m - 1.0 / semi_major_axis))
            )
        else:
            self._v_target = float(np.sqrt(self.reference_planet.mu / self.perigee_radius_m))

    def _compute_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        r_hat = missile.pos_norm
        altitude = np.linalg.norm(missile.position) - self.reference_planet.radius
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

        if v_horiz_mag >= self._v_target:
            logger["Guidance"].info(
                f"{t:<.2f}s - {missile.name} Orbit insertion achieved at altitude {altitude / 1e3:.1f} km, "
                f"v_horiz = {v_horiz_mag:.1f} m/s (target {self._v_target:.1f} m/s)."
            )
            self.state = GuidanceStates.RELEASE_PAYLOAD
            release_velocity = missile.velocity.copy()
            tangential_speed = np.dot(release_velocity, t_hat)
            release_velocity += (np.sign(tangential_speed) * self._v_target - tangential_speed) * t_hat
            return GuidanceResults(
                direction=np.zeros(3),
                state=self.state,
                release_velocity=release_velocity,
            )

        return GuidanceResults(direction=t_hat.copy(), state=self.state, next_guidance=self.next_guidance)


class OrbitalInjectionGuidance(Guidance):
    """Inject a body from a circular parking orbit onto an outbound Hohmann transfer.

    The target planet's current distance from ``reference_planet`` is used as the
    transfer apoapsis. This is a patched-conic approximation: it does not
    propagate the target planet or solve a Lambert problem.

    The thrust is always prograde and the burn ends once the total speed relative to
    the reference planet reaches the vis-viva speed of the transfer ellipse, so the
    radial velocity built up during a finite burn is accounted for.

    If ``start_angle_rad`` is given, the body first waits in its parking orbit, without thrust,
    until it reaches this polar angle (counter-clockwise around +z, from +x, relative to the
    reference planet). The thrust of the first and of the last step is scaled so that the burn
    starts at that angle, and ends at the injection speed, regardless of the simulation step.
    """

    def __init__(
        self,
        reference_planet: Planet,
        target_planet: Planet,
        target: MovableObj | None = None,
        interrupt_fn: Callable[["GuidanceInterrupts"], bool] | None = None,
        start_angle_rad: float | None = None,
    ):
        super().__init__(
            reference_planet=reference_planet,
            target=target if target is not None else target_planet,
            interrupt_fn=interrupt_fn,
        )
        self.target_planet = target_planet
        self._target_orbit_radius = float(np.linalg.norm(target_planet.position - reference_planet.position))
        if self._target_orbit_radius <= reference_planet.radius:
            raise ValueError("target_planet must orbit outside the reference planet")

        self.start_angle_rad = start_angle_rad
        self._burning = start_angle_rad is None
        self._injection_complete = False

    @property
    def target_orbit_radius_m(self) -> float:
        """Distance from the reference planet to the target planet's orbit."""
        return self._target_orbit_radius

    def _prograde_direction(self, position: NDArray) -> NDArray:
        orbit_normal = np.array([0.0, 0.0, 1.0])
        radial = position / np.linalg.norm(position)
        if abs(np.dot(orbit_normal, radial)) > 1.0 - 1e-8:
            orbit_normal = np.array([0.0, 1.0, 0.0])
        tangent = np.cross(orbit_normal, radial)
        return tangent / np.linalg.norm(tangent)

    def _injection_speed(self, radius: float) -> float:
        semi_major_axis = (radius + self._target_orbit_radius) / 2.0
        return float(np.sqrt(self.reference_planet.mu * (2.0 / radius - 1.0 / semi_major_axis)))

    def _ignition_fraction(self, position: NDArray, relative_velocity: NDArray) -> float | None:
        """Fraction of the current step left once the start angle is reached, None if it is not reached during this step."""
        if self.start_angle_rad is None:
            return 1.0
        angular_step = float(np.cross(position, relative_velocity)[2] / np.dot(position, position)) * self.dt
        if angular_step <= 0.0:
            return None
        remaining = (self.start_angle_rad - np.arctan2(position[1], position[0])) % (2.0 * np.pi)
        return 1.0 - remaining / angular_step if remaining <= angular_step else None

    def _compute_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        position = missile.position - self.reference_planet.position
        position_norm = np.linalg.norm(position)
        if position_norm <= self.reference_planet.radius:
            raise ValueError("missile must be outside the reference planet")

        tangent = self._prograde_direction(position)
        injection_speed = self._injection_speed(position_norm)
        relative_velocity = missile.velocity - self.reference_planet.velocity
        speed = float(np.linalg.norm(relative_velocity))

        if self._injection_complete or (self._burning and speed >= injection_speed - 0.05):
            self._injection_complete = True
            self.state = GuidanceStates.COASTING
            self.next_guidance = True
            return GuidanceResults(direction=np.zeros_like(missile.velocity), state=self.state, next_guidance=True)

        magnitude = None
        if not self._burning:
            ignition_fraction = self._ignition_fraction(position, relative_velocity)
            if ignition_fraction is None:
                # Waiting in the parking orbit: the engine is off and does not burn propellant.
                return GuidanceResults(
                    direction=np.zeros_like(missile.velocity), state=GuidanceStates.IDLE, next_guidance=False
                )
            self._burning = True
            self.state = GuidanceStates.POWERED
            magnitude = ignition_fraction * missile.thrust_acc
        elif self.dt > 0.0 and speed > 0.0:
            # Last step: scale the thrust so that the speed ends the step at the injection speed.
            gravity = self.reference_planet.gravity(cast(Any, missile))
            end_radius = float(np.linalg.norm(position + relative_velocity * self.dt))
            speed_gain = (self._injection_speed(end_radius) - speed) / self.dt
            along_track = float(np.dot(tangent, relative_velocity / speed))
            required_acc = (speed_gain - float(np.dot(gravity, relative_velocity / speed))) / along_track
            if required_acc < missile.thrust_acc:
                magnitude = max(required_acc, 0.0)

        return GuidanceResults(
            direction=tangent, state=self.state, next_guidance=self.next_guidance, magnitude=magnitude
        )


class CircularOrbitAscent(_ProgradeTrackingGuidance):
    """Closed-loop ascent to a circular orbit that keeps the vehicle's propellant for later burns.

    Unlike ``OrbitalInsertion`` it never releases a payload: it hands over to the next guidance
    law (``next_guidance``) as soon as the horizontal speed reaches the local circular speed.

    The thrust is split between the local vertical and horizontal directions every step:

    - The vertical part makes the vehicle climb along a vertical-speed profile
      ``vr = sqrt(2 * braking_acc * (orbit_altitude - altitude))`` that reaches zero at the orbit altitude,
      and then holds that altitude by cancelling gravity minus centrifugal acceleration.
    - The rest of the thrust accelerates the vehicle horizontally. The thrust of the last step is scaled so that
      the horizontal speed ends exactly at the circular speed, regardless of the simulation step.

    As the vertical part is computed from the available thrust acceleration, the law works for any thrust-to-weight ratio.
    """

    def __init__(
        self,
        reference_planet: Planet,
        orbit_altitude_m: float,
        braking_acc: float = 4.0,
        gain: float = 0.3,
        target: MovableObj | None = None,
        interrupt_fn: Callable[["GuidanceInterrupts"], bool] | None = None,
    ):
        super().__init__(reference_planet=reference_planet, target=target, interrupt_fn=interrupt_fn)
        self.orbit_altitude_m = orbit_altitude_m
        self.braking_acc = braking_acc  # m/s², vertical deceleration used to reach the orbit altitude.
        self.gain = gain  # 1/s, tracking gain on the vertical-speed profile.

    def _compute_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        mu = self.reference_planet.mu
        radius = float(np.linalg.norm(missile.position))
        r_hat = missile.position / radius
        altitude = radius - self.reference_planet.radius
        t_hat = self._resolve_t_hat(missile, r_hat)
        radial_speed = float(np.dot(missile.velocity, r_hat))
        horizontal_speed = float(np.dot(missile.velocity, t_hat))

        if altitude > 0.5 * self.orbit_altitude_m and horizontal_speed >= np.sqrt(mu / radius) - 0.5:
            logger["Guidance"].info(
                f"{t:<.2f}s - {missile.name} Circular orbit reached at altitude {altitude / 1e3:.1f} km, "
                f"v_horiz = {horizontal_speed:.1f} m/s."
            )
            self.state = GuidanceStates.COASTING
            self.next_guidance = True
            return GuidanceResults(direction=np.zeros(3), state=self.state, next_guidance=True)

        thrust_acc = missile.thrust_acc
        if thrust_acc <= 0.0:
            return GuidanceResults(direction=np.zeros(3), state=self.state, next_guidance=self.next_guidance)

        desired_radial_speed = np.sqrt(2.0 * self.braking_acc * max(self.orbit_altitude_m - altitude, 0.0))
        radial_acc = self.gain * (desired_radial_speed - radial_speed) + mu / radius**2 - horizontal_speed**2 / radius
        radial_acc = float(np.clip(radial_acc, -0.9 * thrust_acc, 0.95 * thrust_acc))
        horizontal_acc = np.sqrt(thrust_acc**2 - radial_acc**2)

        # Last step: scale the thrust so that the horizontal speed ends the step at the circular speed.
        magnitude = None
        if self.dt > 0.0:
            required_acc = (np.sqrt(mu / radius) - horizontal_speed) / self.dt
            if required_acc < horizontal_acc:
                magnitude = thrust_acc * max(required_acc, 0.0) / horizontal_acc

        direction = (radial_acc * r_hat + horizontal_acc * t_hat) / thrust_acc
        return GuidanceResults(
            direction=direction, state=self.state, next_guidance=self.next_guidance, magnitude=magnitude
        )


def LEOAscentGuidance(
    reference_planet: Planet,
    orbit_altitude_m: float,
    vertical_rise_altitude_m: float = 1_000.0,
    braking_acc: float = 4.0,
    gain: float = 0.3,
    target: MovableObj | None = None,
    interrupt_fn: Callable[["GuidanceInterrupts"], bool] | None = None,
) -> GuidanceManager:
    """Assemble a launch-to-circular-orbit guidance sequence that keeps the payload on board.

    Same idea as ``LEOInsertionGuidance``, but the last law is ``CircularOrbitAscent``: it ends with a
    coast instead of a payload release, so a later guidance law (e.g. a trans-lunar injection) can keep using the rocket.
    """
    vertical_rise = StraightUp(
        reference_planet=reference_planet,
        target=target,  # type: ignore[arg-type]
        interrupt_fn=lambda i: interrupt_at_altitude(i, vertical_rise_altitude_m),
    )
    ascent = CircularOrbitAscent(
        reference_planet=reference_planet,
        orbit_altitude_m=orbit_altitude_m,
        braking_acc=braking_acc,
        gain=gain,
        target=target,
        interrupt_fn=interrupt_fn,
    )
    return GuidanceManager([vertical_rise, ascent])


def LEOInsertionGuidance(
    reference_planet: Planet,
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
        reference_planet=reference_planet,
        target=target,  # type: ignore[arg-type]
        interrupt_fn=lambda i: interrupt_at_altitude(i, min_turn_altitude_m),
    )
    pitch_program = CosinePitchProgram(
        reference_planet=reference_planet,
        target=target,
        min_turn_altitude_m=min_turn_altitude_m,
        turn_end_altitude_m=turn_end,
        interrupt_fn=lambda i: interrupt_at_altitude(i, perigee_altitude_m - tol),
    )
    orbital_insertion = OrbitalInsertion(
        reference_planet=reference_planet,
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
