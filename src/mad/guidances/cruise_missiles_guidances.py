from mad.objs.base import MovableObj
from mad.guidances.base_guidances import Guidance, GuidableObj, GuidanceResults
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from mad.utils.logger import SourceLogger

logger = SourceLogger()


@dataclass
class CruiseGuidanceConfig:
    max_speed_m_s: float
    cruise_altitude_m: float
    max_range_m: float
    waypoints: list[MovableObj]


class CruiseWaypointGuidance(Guidance):
    """Waypoint guidance for cruise missiles using a cubic spline trajectory.

    A cubic spline is fitted through all waypoints (each projected radially to
    ``cruise_altitude_m``).  The first waypoint is the launch site; the last is
    the final target.  Between waypoints the missile stays near cruise altitude
    and does not exceed ``max_speed_m_s``.

    Guidance direction is the blend of:
      - a lateral component pointing toward a lookahead point on the spline, and
      - a radial component that corrects altitude error (proportional feedback).

    Thrust magnitude is set to zero once ``max_speed_m_s`` is reached.
    """

    # Lookahead distance along the spline (m).  Can be tuned per scenario.
    _LOOKAHEAD_M: float = 50_000.0

    def __init__(self, planet, target: MovableObj, config: CruiseGuidanceConfig):
        super().__init__(planet, target)
        self.cfg = config

        # Per-axis cubic splines and arc-length knot vector, built in _build_spline.
        self._spline_x: CubicSpline | None = None
        self._spline_y: CubicSpline | None = None
        self._spline_z: CubicSpline | None = None
        self._s_params: NDArray | None = None
        self._total_arc: float = 0.0

        # Monotonically non-decreasing progress along the spline (arc-length, m).
        self._progress_s: float = 0.0

        self._build_spline()

    def _build_spline(self) -> None:
        """Fit a cubic spline through waypoints projected to cruise altitude."""
        alt = self.cfg.cruise_altitude_m
        pts = np.array([(self.planet.radius + alt) * wp.normalize for wp in self.cfg.waypoints])  # (N, 3)

        # Arc-length parameterisation
        diffs = np.diff(pts, axis=0)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
        self._total_arc = float(s[-1])
        self._s_params = s

        self._spline_x = CubicSpline(s, pts[:, 0])
        self._spline_y = CubicSpline(s, pts[:, 1])
        self._spline_z = CubicSpline(s, pts[:, 2])

        logger["Guidance"].info(
            f"CruiseWaypointGuidance: spline built over {len(self.cfg.waypoints)} waypoints, "
            f"total arc length {self._total_arc / 1000:.1f} km."
        )

    def _eval_spline(self, s: float) -> NDArray:
        """Evaluate the spline at arc-length parameter s (clamped to valid range)."""
        assert self._spline_x is not None and self._spline_y is not None and self._spline_z is not None
        s = float(np.clip(s, 0.0, self._total_arc))
        return np.array(
            [
                float(self._spline_x(s)),
                float(self._spline_y(s)),
                float(self._spline_z(s)),
            ]
        )

    def _advance_progress(self, missile: GuidableObj) -> None:
        """Move _progress_s to the nearest spline point within a forward search window.

        The search is strictly forward (no backtracking) to handle the missile
        passing a waypoint cleanly.
        """
        assert self._spline_x is not None and self._spline_y is not None and self._spline_z is not None
        search_window = max(self._LOOKAHEAD_M, self._total_arc * 0.05)
        s_start = self._progress_s
        s_end = min(s_start + search_window, self._total_arc)

        candidates = np.linspace(s_start, s_end, 100)
        pts = np.column_stack(
            [
                self._spline_x(candidates),
                self._spline_y(candidates),
                self._spline_z(candidates),
            ]
        )
        dists = np.linalg.norm(pts - missile.position, axis=1)
        self._progress_s = float(candidates[int(np.argmin(dists))])

    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        # Terminal condition: missile is within 500 m of the target.
        dist_to_target = np.linalg.norm(missile.position - self.target.position)
        if dist_to_target < 500.0:
            if self.state != "terminal":
                self.state = "terminal"
                logger["Guidance"].info(f"{missile.name} reached terminal range ({dist_to_target:.0f} m from target).")
            return GuidanceResults(direction=np.zeros(3), state=self.state)

        # Update progress along the spline.
        self._advance_progress(missile)

        # Choose the lookahead point: LOOKAHEAD_M ahead of current progress,
        # clamped to the end of the spline.
        s_lookahead = min(self._progress_s + self._LOOKAHEAD_M, self._total_arc)
        lookahead_pt = self._eval_spline(s_lookahead)

        # Lateral direction: normalised vector toward lookahead.
        lateral = lookahead_pt - missile.position
        lat_norm = np.linalg.norm(lateral)
        if lat_norm > 1e-8:
            lateral = lateral / lat_norm
        else:
            lateral = np.zeros(3)

        # Altitude correction: proportional radial feedback.
        # alt_frac is dimensionless — it equals 1 when the error equals LOOKAHEAD_M.
        r_hat = missile.normalize
        current_alt = np.linalg.norm(missile.position) - self.planet.radius
        alt_error = self.cfg.cruise_altitude_m - current_alt
        alt_frac = np.clip(alt_error / self._LOOKAHEAD_M, -1.0, 1.0)

        # Blend: lateral component dominates; radial term corrects altitude.
        raw = lateral + alt_frac * r_hat
        raw_norm = np.linalg.norm(raw)
        if raw_norm < 1e-8:
            return GuidanceResults(direction=np.zeros(3), state=self.state)
        direction = raw / raw_norm

        # Speed limiter: cut thrust once max speed is reached.
        speed = float(np.linalg.norm(missile.velocity))
        magnitude: float | None = None if speed < self.cfg.max_speed_m_s else 0.0

        return GuidanceResults(direction=direction, state=self.state, magnitude=magnitude)
