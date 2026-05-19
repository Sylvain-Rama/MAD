from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from mad.objs.base import MovableObj
from mad.guidances.base_guidances import Guidance, GuidableObj, GuidanceResults
from mad.logger import SourceLogger

logger = SourceLogger()


class LEOInsertionState(StrEnum):
    VERTICAL_RISE = "vertical_rise"
    PITCH_PROGRAM = "pitch_program"
    ORBIT_INSERTION = "orbit_insertion"
    RELEASE_PAYLOAD = "release_payload"


class LEOInsertionGuidance(Guidance):
    """Guides a launch vehicle to a low-Earth orbit (circular or elliptical) at a desired perigee altitude.

    Three sequential phases
    -----------------------
    1. **vertical_rise** – thrust straight up (along ``r_hat``) until
       ``min_turn_altitude_m`` is reached.
    2. **pitch_program** – smoothly pitch from vertical to horizontal as the
       vehicle climbs from ``min_turn_altitude_m`` toward ``target_altitude_m``.
       The pitch angle follows a cosine-ease schedule
       ``θ = (π/2) · (1 − cos(π·p)) / 2``  where *p* ∈ [0, 1] is the altitude
       progress fraction, giving a smooth start and end to the turn.
    3. **orbit_insertion** – once the vehicle is within ``altitude_tol_m`` of
       ``target_altitude_m``, thrust purely in the tangential direction to reach
       the required perigee speed.  For a **circular** orbit this is
       ``v_circ = √(μ/r_p)``; for an **elliptical** orbit (when
       ``apogee_altitude_m`` is supplied) the vis-viva perigee speed is used:
       ``v_p = √(μ · (2/r_p − 1/a))`` where ``a = (r_p + r_a) / 2``.
       ``release_payload`` is declared when the horizontal speed reaches
       99 % of the target speed; the current velocity is set as
       ``release_velocity`` so the missile releases the satellite at the
       correct orbital velocity.

    Tangential direction
    --------------------
    When a ``target`` ``MovableObj`` is supplied, the tangential direction
    ``t_hat`` is derived from the great-circle direction toward it (the same
    ``local_frame`` used by ``GravityTurn``), which naturally encodes the
    desired orbital plane.  This makes polar and arbitrary-inclination orbits
    easy to specify: just place the target anywhere in the desired orbital
    plane.

    When no ``target`` is given, ``t_hat`` falls back to the horizontal
    component of the vehicle's own velocity (updated each step), or to an
    eastward direction on the first call before any horizontal speed builds up.

    Parameters
    ----------
    planet :
        The central body (must expose ``planet.radius`` and ``planet.mu``).
    target_altitude_m :
        Perigee altitude above the surface in **metres**.  The vehicle is
        released here (or as close to it as propellant allows).
    apogee_altitude_m :
        Apogee altitude in **metres** for an elliptical target orbit.
        When ``None`` (default) a circular orbit at ``target_altitude_m``
        is targeted.
    target :
        Optional reference point whose direction from the vehicle defines the
        desired orbital plane.  Pass any ``MovableObj`` (e.g. a ground target
        or a virtual waypoint) that lies in the intended orbital plane.
        If ``None``, the guidance defaults to the vehicle's prograde direction.
    min_turn_altitude_m :
        Altitude above the surface at which pitch-over begins (metres).
        Defaults to 1 000 m.
    turn_end_altitude_m :
        Altitude at which the pitch programme completes and the vehicle thrusts
        purely in the tangential (horizontal) direction (metres).  Setting this
        well below ``target_altitude_m`` greatly reduces gravity and steering
        losses: the vehicle pitches over early and then coasts / thrusts
        horizontally to gain orbital speed before reaching the target orbit.
        Defaults to 80 % of ``target_altitude_m``.
    altitude_tol_m :
        Half-width of the altitude band (metres) around ``target_altitude_m``
        in which orbit insertion is triggered.
        Defaults to 5 % of ``target_altitude_m``.
    """

    def __init__(
        self,
        planet,
        target_altitude_m: float,
        apogee_altitude_m: float | None = None,
        target: MovableObj | None = None,
        min_turn_altitude_m: float = 1_000.0,
        turn_end_altitude_m: float | None = None,
        altitude_tol_m: float | None = None,
    ):
        super().__init__(planet, target=target)  # type: ignore[arg-type]  # target may be None
        self.target_altitude_m = target_altitude_m
        self.target_radius_m = planet.radius + target_altitude_m
        self.min_turn_altitude_m = min_turn_altitude_m
        self.turn_end_altitude_m = turn_end_altitude_m if turn_end_altitude_m is not None else 0.8 * target_altitude_m
        self.altitude_tol_m = altitude_tol_m if altitude_tol_m is not None else 0.05 * target_altitude_m
        self.state = LEOInsertionState.VERTICAL_RISE

        # Target orbital speed at perigee.
        # Circular:    v = √(μ / r_p)
        # Elliptical:  v = √(μ · (2/r_p − 1/a)),  a = (r_p + r_a) / 2  [vis-viva]
        if apogee_altitude_m is not None:
            r_a = planet.radius + apogee_altitude_m
            semi_major_axis = (self.target_radius_m + r_a) / 2.0
            self._v_target = np.sqrt(planet.mu * (2.0 / self.target_radius_m - 1.0 / semi_major_axis))
        else:
            self._v_target = np.sqrt(planet.mu / self.target_radius_m)

        # Cached prograde unit vector; only used when target is None.
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
                return t_hat

        # Fall back to prograde from velocity.
        v_horiz = missile.velocity - np.dot(missile.velocity, r_hat) * r_hat
        v_horiz_mag = np.linalg.norm(v_horiz)

        if v_horiz_mag > 1.0:
            self._prograde_hat = v_horiz / v_horiz_mag
        elif self._prograde_hat is None:
            # No horizontal velocity yet; default to an eastward direction.
            north = np.array([0.0, 0.0, 1.0])
            east = np.cross(north, r_hat)
            east_mag = np.linalg.norm(east)
            self._prograde_hat = east / east_mag if east_mag > 1e-8 else np.array([1.0, 0.0, 0.0])

        return self._prograde_hat  # type: ignore[return-value]

    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        r = np.linalg.norm(missile.position)
        altitude = r - self.planet.radius
        r_hat = missile.normalize

        if altitude < self.min_turn_altitude_m:
            self.state = LEOInsertionState.VERTICAL_RISE
            return GuidanceResults(direction=r_hat.copy(), state=self.state)

        t_hat = self._resolve_t_hat(missile, r_hat)
        v_horiz_mag = abs(np.dot(missile.velocity, t_hat))

        # Fallback: all propellant spent — release payload when either:
        #   (a) the vehicle has reached the target altitude band, or
        #   (b) the vehicle is at/past apogee (radial velocity ≤ 0) above 80 km.
        # Case (b) ensures the payload is deployed at the highest possible point
        # when the rocket is underpowered to reach the nominal target altitude.
        # NOTE: use has_thrust (stages list empty) rather than burned_fraction, which
        # is an imprecise formula that can return >= 1.0 a few seconds early.
        if not missile.has_thrust:
            v_r = np.dot(missile.velocity, r_hat)
            at_target_band = altitude >= self.target_altitude_m - self.altitude_tol_m
            at_apogee = v_r <= 0.0 and altitude > 80_000.0
            if at_target_band or at_apogee:
                logger["Guidance"].info(
                    f"All propellant spent at altitude {altitude / 1e3:.1f} km, "
                    f"v_horiz = {v_horiz_mag:.1f} m/s (target {self._v_target:.1f} m/s). Releasing payload."
                )
                self.state = LEOInsertionState.RELEASE_PAYLOAD
                return GuidanceResults(
                    direction=np.zeros(3),
                    state=self.state,
                    release_velocity=missile.velocity.copy(),
                )

        if abs(altitude - self.target_altitude_m) <= self.altitude_tol_m:
            self.state = LEOInsertionState.ORBIT_INSERTION

            if v_horiz_mag >= 0.99 * self._v_target:
                logger["Guidance"].info(
                    f"Orbit insertion achieved at altitude {altitude / 1e3:.1f} km, "
                    f"v_horiz = {v_horiz_mag:.1f} m/s (target {self._v_target:.1f} m/s)."
                )
                self.state = LEOInsertionState.RELEASE_PAYLOAD
                return GuidanceResults(
                    direction=np.zeros(3),
                    state=self.state,
                    release_velocity=missile.velocity.copy(),
                )

            return GuidanceResults(direction=t_hat.copy(), state=self.state)

        self.state = LEOInsertionState.PITCH_PROGRAM

        # Cosine-ease pitch schedule: smooth start and end to the turn.
        #   p = 0  → θ = 0       (vertical)
        #   p = 1  → θ = π/2     (horizontal)
        # Progress saturates at 1 once turn_end_altitude_m is reached, after
        # which the vehicle thrusts purely tangentially until orbit insertion.
        progress = np.clip(
            (altitude - self.min_turn_altitude_m) / (self.turn_end_altitude_m - self.min_turn_altitude_m),
            0.0,
            1.0,
        )
        theta = (np.pi / 2.0) * (1.0 - np.cos(np.pi * progress)) / 2.0

        direction = np.cos(theta) * r_hat + np.sin(theta) * t_hat
        return GuidanceResults(direction=direction / np.linalg.norm(direction), state=self.state)


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
