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
    waypoints: list[MovableObj]


class CruiseWaypointGuidance(Guidance):
    """Waypoint guidance for cruise missiles using a cubic spline trajectory.

    A cubic spline is fitted through all waypoints.  The first waypoint is the launch site; the last is
    the final target.  Between waypoints the missile stays near cruise altitude
    and does not exceed ``max_speed_m_s``.

    Guidance direction is the blend of:
      - a lateral component pointing toward a lookahead point on the spline, and
      - a radial component that corrects altitude error (proportional feedback).

    Thrust magnitude is set to zero once ``max_speed_m_s`` is reached.
    """

    # Lookahead distance along the spline (m).  Can be tuned per scenario.
    _LOOKAHEAD_M: float = 50_000.0

    def __init__(self, planet, target: MovableObj, waypoints: list[MovableObj]):
        super().__init__(planet, target)

        # Per-axis cubic splines and arc-length knot vector, built in _build_spline.
        self._spline_x: CubicSpline | None = None
        self._spline_y: CubicSpline | None = None
        self._spline_z: CubicSpline | None = None
        self._s_params: NDArray | None = None
        self._total_arc: float = 0.0

        # Monotonically non-decreasing progress along the spline (arc-length, m).
        self._progress_s: float = 0.0
        self.waypoints = waypoints

        self._build_spline()

    # Spacing between densification points along great-circle arcs (m).
    _DENSIFY_STEP_M: float = 200_000.0

    def _build_spline(self) -> None:
        """Fit a cubic spline through waypoints densified along great-circle arcs.

        A plain Cartesian spline through sparse waypoints cuts through (or far above)
        the sphere for long ranges.  We insert intermediate points via SLERP so every
        chord is at most ``_DENSIFY_STEP_M`` of arc, keeping the spline near the
        sphere surface at ``cruise_altitude_m``.
        """

        r = self.planet.radius
        normals = [wp.normalize for wp in self.waypoints]

        # Densify each segment with SLERP-interpolated points at cruise altitude.
        dense_pts: list[NDArray] = []
        for i in range(len(normals) - 1):
            n0 = normals[i]
            n1 = normals[i + 1]
            cos_sigma = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
            sigma = np.arccos(cos_sigma)
            arc_len = r * sigma
            n_steps = max(2, int(np.ceil(arc_len / self._DENSIFY_STEP_M)) + 1)
            # Exclude the last point of each segment to avoid duplicates;
            # include it only on the final segment.
            end = n_steps if i == len(normals) - 2 else n_steps - 1
            for j in range(end):
                frac = j / (n_steps - 1)
                if sigma < 1e-10:
                    n = n0
                else:
                    n = (np.sin((1.0 - frac) * sigma) * n0 + np.sin(frac * sigma) * n1) / np.sin(sigma)
                dense_pts.append(r * n)

        pts = np.array(dense_pts)

        # Arc-length parameterisation
        diffs = np.diff(pts, axis=0)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
        self._total_arc = float(s[-1])
        self._s_params = s

        self._spline_x = CubicSpline(s, pts[:, 0])
        self._spline_y = CubicSpline(s, pts[:, 1])
        self._spline_z = CubicSpline(s, pts[:, 2])

        logger["Guidance"].info(
            f"CruiseWaypointGuidance: spline built over {len(self.waypoints)} waypoints "
            f"({len(pts)} dense points), total arc length {self._total_arc / 1000:.1f} km."
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
        # Terminal condition: missile is within 100 m of the target.
        dist_to_target = np.linalg.norm(missile.position - self.target.position)
        if dist_to_target < 100.0:
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

        # Tangential unit vector on the sphere surface pointing toward the lookahead point.
        r_hat, _ = self.local_frame(missile)
        lookahead_hat = lookahead_pt / np.linalg.norm(lookahead_pt)
        t_hat = np.cross(np.cross(r_hat, lookahead_hat), r_hat)
        t_norm = np.linalg.norm(t_hat)
        if t_norm > 1e-8:
            t_hat /= t_norm
        else:
            t_hat = np.zeros(3)

        return GuidanceResults(direction=t_hat, state=self.state)
