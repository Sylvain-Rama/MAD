from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import TYPE_CHECKING
from scipy.spatial import KDTree
from mad.logger import SourceLogger
from mad.configs.physics import VOXEL_SIZE

if TYPE_CHECKING:
    from mad.simulation import HistoryCollector


@dataclass
class BallisticTable:
    table: NDArray
    alt_scale: float
    vel_scale: float
    gam_scale: float
    range_scale: float
    kdtree: KDTree  # Built from normalized (alt, vel, gamma) inputs
    kdtree_range: KDTree  # Built from normalized (alt, vel, range_rad) for range-based gamma lookup


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


def to_voxel_key(position: list | NDArray, voxel_size: float = VOXEL_SIZE) -> tuple[int, ...]:
    """Convert a position in metres to a voxel key (tuple of ints) based on the given voxel size in km."""
    if not isinstance(position, (list, np.ndarray)):
        raise TypeError(f"Position must be a list or NDArray, got {type(position)} instead.")
    pos_arr = np.asarray(position)
    if pos_arr.shape[0] < 3:
        raise ValueError(f"Position must have at least 3 components, got shape {pos_arr.shape} instead.")
    key = tuple(np.floor(pos_arr / voxel_size).astype(int))
    return key


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
        range_scale = np.ptp(table[:, 3]) or 1.0

        norm_inputs = np.column_stack(
            [
                table[:, 0] / alt_scale,
                table[:, 1] / vel_scale,
                table[:, 2] / gam_scale,
            ]
        )
        kdtree = KDTree(norm_inputs)

        # TODO: Use the second KDTree for range-based lookup of gamma given alt, vel, and range.
        # Removed the 2nd KDTree use for the moment.
        norm_inputs_range = np.column_stack(
            [
                table[:, 0] / alt_scale,
                table[:, 1] / vel_scale,
                table[:, 3] / range_scale,
            ]
        )
        kdtree_range = KDTree(norm_inputs_range)

        return BallisticTable(
            table=table,
            alt_scale=alt_scale,
            vel_scale=vel_scale,
            gam_scale=gam_scale,
            range_scale=range_scale,
            kdtree=kdtree,
            kdtree_range=kdtree_range,
        )

    except Exception as e:
        logger["I/O"].error(f"Error loading ballistic table from {file_path}: {e}")
        return None



