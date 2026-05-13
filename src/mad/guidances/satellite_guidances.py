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
    """Guides a launch vehicle to a circular low-Earth orbit at a desired altitude.

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
       the target altitude, thrust purely in the tangential direction to reach
       circular-orbit speed ``v_circ = √(μ/r_target)``.
       ``release_payload`` is declared when the horizontal speed reaches
       99 % of ``v_circ``; the current velocity is set as ``release_velocity``
       so the missile releases the satellite at the correct orbital velocity.

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
        Desired circular-orbit altitude above the surface in **metres**.
    target :
        Optional reference point whose direction from the vehicle defines the
        desired orbital plane.  Pass any ``MovableObj`` (e.g. a ground target
        or a virtual waypoint) that lies in the intended orbital plane.
        If ``None``, the guidance defaults to the vehicle's prograde direction.
    min_turn_altitude_m :
        Altitude above the surface at which pitch-over begins (metres).
        Defaults to 1 000 m.
    altitude_tol_m :
        Half-width of the altitude band (metres) around ``target_altitude_m``
        in which orbit insertion is triggered.
        Defaults to 5 % of ``target_altitude_m``.
    """

    def __init__(
        self,
        planet,
        target_altitude_m: float,
        target: MovableObj | None = None,
        min_turn_altitude_m: float = 1_000.0,
        altitude_tol_m: float | None = None,
    ):
        super().__init__(planet, target=target)  # type: ignore[arg-type]  # target may be None
        self.target_altitude_m = target_altitude_m
        self.target_radius_m = planet.radius + target_altitude_m
        self.min_turn_altitude_m = min_turn_altitude_m
        self.altitude_tol_m = altitude_tol_m if altitude_tol_m is not None else 0.05 * target_altitude_m
        self.state = LEOInsertionState.VERTICAL_RISE

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
        v_circ = np.sqrt(self.planet.mu / self.target_radius_m)
        v_horiz_mag = abs(np.dot(missile.velocity, t_hat))


        if abs(altitude - self.target_altitude_m) <= self.altitude_tol_m:
            self.state = LEOInsertionState.ORBIT_INSERTION

            if v_horiz_mag >= 0.99 * v_circ:
                logger["Guidance"].info(
                    f"LEO orbit achieved at altitude {altitude / 1e3:.1f} km, "
                    f"v_horiz = {v_horiz_mag:.1f} m/s (target {v_circ:.1f} m/s)."
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
        progress = np.clip(
            (altitude - self.min_turn_altitude_m) / (self.target_altitude_m - self.min_turn_altitude_m),
            0.0,
            1.0,
        )
        theta = (np.pi / 2.0) * (1.0 - np.cos(np.pi * progress)) / 2.0

        direction = np.cos(theta) * r_hat + np.sin(theta) * t_hat
        return GuidanceResults(direction=direction / np.linalg.norm(direction), state=self.state)
