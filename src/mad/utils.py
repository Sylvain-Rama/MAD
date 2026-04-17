from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import TYPE_CHECKING
from mad.logger import SourceLogger

if TYPE_CHECKING:
    from mad.objs.planets import Planet


@dataclass
class BallisticTable:
    table: NDArray
    alt_scale: float
    vel_scale: float
    gam_scale: float


BALLISTIC_FIELD_NAMES = ["altitude_m", "velocity_m_s", "gamma_rad", "range_rad"]

logger = SourceLogger()


def to_vec3(arr: list | NDArray) -> NDArray:
    if not isinstance(arr, (list, np.ndarray)):
        raise TypeError(f"This vector expected a list or NDarray, got {type(arr)} instead.")
    if isinstance(arr, list):
        arr = np.asarray(arr)

    if arr.shape[0] >= 3:
        return arr[:3]

    out = np.zeros(3, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out


def load_ballistic_table(table_name: str) -> BallisticTable | None:
    """Load a ballistic table from a CSV file. The CSV file must have columns: altitude_m, velocity_m_s, gamma_rad, range_rad.
    The first row must be a header with exactly those column names. Returns a BallisticTable object."""

    # TODO: Use a proper path management solution and make this more robust to different environments.
    # For now, we assume the tables are in src/mad/tables and the script is run from the root of the repo.
    file_path = f"/app/src/mad/tables/{table_name}.csv"
    try:
        with open(file_path, newline="") as f:
            header = f.readline().strip().split(",")
        if header != BALLISTIC_FIELD_NAMES:
            logger["I/O"].error(f"Ballistic table must have columns {BALLISTIC_FIELD_NAMES}. Got {header} instead.")
            return None
        table = np.genfromtxt(file_path, delimiter=",", skip_header=1, filling_values=np.nan)
        n_before = len(table)
        table = table[~np.isnan(table).any(axis=1)]
        n_dropped = n_before - len(table)
        if n_dropped:
            logger["I/O"].warning(f"Dropped {n_dropped} row(s) with missing values from {file_path}.")

        alt_scale = np.ptp(table[:, 0]) or 1.0
        vel_scale = np.ptp(table[:, 1]) or 1.0
        gam_scale = np.ptp(table[:, 2]) or 1.0

        return BallisticTable(table=table, alt_scale=alt_scale, vel_scale=vel_scale, gam_scale=gam_scale)

    except Exception as e:
        logger["I/O"].error(f"Error loading ballistic table from {file_path}: {e}")
        return None


def extract_history(objs: list, planet: Planet) -> dict[str, dict[str, NDArray]]:

    results = {}
    for obj in objs:
        pos = np.asarray(obj.history.position)
        vel = np.asarray(obj.history.velocity)
        time = np.asarray(obj.history.time)
        gamma = np.asarray([x for x in obj.history.gamma if x is not None]) if obj.history.gamma else None

        posx, posz = pos[:, 0], pos[:, 1]

        r = np.linalg.norm(pos, axis=1)
        velout = np.linalg.norm(vel, axis=1)

        energy = 0.5 * velout**2 - planet.mu / r
        altitude = r - planet.radius

        results[obj.name] = {
            "time": time,
            "altitude": altitude,
            "velocity": velout,
            "posx": posx,
            "posz": posz,
            "energy": energy,
        }
        if gamma is not None:
            results[obj.name]["gamma"] = gamma

    return results
