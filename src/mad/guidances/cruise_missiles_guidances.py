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
    max_speed_m_s: float = 300.0  # m/s — top cruise speed
    altitude_settling_time_s: float = 30.0  # desired altitude settling time (critical damping)
    cruise_altitude_m: float = 100.0


class CruiseWaypointGuidance(Guidance):
    """Waypoint guidance for cruise missiles using a cubic spline trajectory.

    A cubic spline is fitted through all waypoints.  The first waypoint is the launch site; the last is
    the final target. Between waypoints, guidance computes a tangential command
    toward a spline lookahead point and a radial command that regulates altitude.

        Guidance direction is the blend of:
      - a lateral component pointing toward a lookahead point on the spline, and
            - a radial component that corrects altitude error (PD feedback + gravity compensation).
    """

    # Lookahead distance along the spline (m).  Can be tuned per scenario.
    _LOOKAHEAD_M: float = 50_000.0

    def __init__(
        self,
        planet,
        target: MovableObj,
        config: CruiseGuidanceConfig,
    ):
        super().__init__(planet, target)

        self.config = config

        # Per-axis cubic splines and arc-length knot vector, built in _build_spline.
        self._spline_x: CubicSpline | None = None
        self._spline_y: CubicSpline | None = None
        self._spline_z: CubicSpline | None = None
        self._s_params: NDArray | None = None
        self._total_arc: float = 0.0

        # Monotonically non-decreasing progress along the spline (arc-length, m).
        self._progress_s: float = 0.0
        self.waypoints = self.config.waypoints

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

        r = self.planet.radius + self.config.cruise_altitude_m
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

        # Altitude hold: gravity compensation + critically-damped PD radial control.
        # Returns a fractional vector (components are fractions of thrust_acc) so
        # CruiseMissile can apply `thrust_acc * direction` directly without re-normalizing,
        # preserving the absolute radial and tangential acceleration magnitudes.
        pos_norm = float(np.linalg.norm(missile.position))
        current_alt = pos_norm - self.planet.radius
        alt_error = self.config.cruise_altitude_m - current_alt
        v_radial = float(np.dot(missile.velocity, r_hat))
        g_mag = self.planet.mu / max(pos_norm**2, 1e-9)
        omega_n = 4.0 / max(self.config.altitude_settling_time_s, 1.0)
        Kp = omega_n**2
        Kd = 2.0 * omega_n
        available_acc = max(float(getattr(missile, "thrust_acc", 0.0)), 1e-9)
        radial_acc = np.clip(g_mag + Kp * alt_error - Kd * v_radial, -available_acc, available_acc)
        radial_frac = radial_acc / available_acc  # in (-1, 1]

        # Include tangential thrust only when below the speed cap.
        speed = float(np.linalg.norm(missile.velocity))
        if speed < self.config.max_speed_m_s:
            cmd = t_hat + radial_frac * r_hat
        else:
            cmd = radial_frac * r_hat

        return GuidanceResults(direction=cmd, state=self.state)


class PurePursuit(Guidance):
    """Pure-pursuit guidance with altitude hold.

    The horizontal component points toward the target's current position
    (projected onto the local tangential plane).  The radial component
    applies a gravity-compensating critically-damped PD controller to
    maintain ``cruise_altitude_m`` above the planet surface.

    Parameters
    ----------
    planet:
        Planet object used for gravity and radius.
    target:
        Target to pursue.
    cruise_altitude_m:
        Desired altitude above the planet surface (metres).  Defaults to
        the missile's altitude at the first guidance call.
    altitude_settling_time_s:
        Desired altitude settling time in seconds (critical damping).
    terminal_range_m:
        Distance to the target (metres) at which the guidance switches from
        altitude-hold cruise to direct 3-D line-of-sight pursuit, allowing
        the missile to close on a target at any altitude.  The motor stays
        active during this phase (state becomes ``"homing"``, not
        ``"terminal"``).  Default 10 km.
    """

    def __init__(
        self,
        planet,
        target: MovableObj,
        cruise_altitude_m: float | None = None,
        altitude_settling_time_s: float = 30.0,
        terminal_range_m: float = 10_000.0,
    ):
        super().__init__(planet, target)
        self._cruise_altitude_m = cruise_altitude_m
        self.altitude_settling_time_s = altitude_settling_time_s
        self.terminal_range_m = terminal_range_m

    def get_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        # Initialise cruise altitude from the missile's current altitude on the first call.
        if self._cruise_altitude_m is None:
            self._cruise_altitude_m = float(np.linalg.norm(missile.position)) - self.planet.radius

        los = self.target.position - missile.position
        los_norm = np.linalg.norm(los)

        # Homing phase: switch to direct 3-D pursuit so the missile can close on a
        # target at a different altitude.  State is "homing", NOT "terminal", so that
        # CruiseMissile does not cut the motor — thrust must remain active to the end.
        if los_norm < self.terminal_range_m:
            if self.state != "homing":
                self.state = "homing"
                logger["Guidance"].info(f"{missile.name} entered homing phase ({los_norm:.0f} m from target).")
            if los_norm < 1e-8:
                return GuidanceResults(direction=np.zeros(3), state=self.state)
            return GuidanceResults(direction=los / los_norm, state=self.state)

        # Cruise phase: altitude-hold + horizontal pursuit.
        r_hat = missile.normalize

        # Horizontal pursuit: project LOS onto the local tangential plane.
        los_hat = los / los_norm
        los_tan = los_hat - np.dot(los_hat, r_hat) * r_hat
        t_norm = np.linalg.norm(los_tan)
        t_hat = los_tan / t_norm if t_norm > 1e-8 else np.zeros(3)

        # Altitude-hold: gravity compensation + critically-damped PD.
        pos_norm = float(np.linalg.norm(missile.position))
        current_alt = pos_norm - self.planet.radius
        alt_error = self._cruise_altitude_m - current_alt
        v_radial = float(np.dot(missile.velocity, r_hat))
        g_mag = self.planet.mu / max(pos_norm**2, 1e-9)
        omega_n = 4.0 / max(self.altitude_settling_time_s, 1.0)
        Kp = omega_n**2
        Kd = 2.0 * omega_n
        available_acc = max(float(getattr(missile, "thrust_acc", 0.0)), 1e-9)
        radial_acc = np.clip(g_mag + Kp * alt_error - Kd * v_radial, -available_acc, available_acc)
        radial_frac = radial_acc / available_acc  # in (-1, 1]

        cmd = t_hat + radial_frac * r_hat
        return GuidanceResults(direction=cmd, state=self.state)
