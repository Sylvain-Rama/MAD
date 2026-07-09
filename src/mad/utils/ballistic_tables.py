from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from dataclasses import dataclass
from scipy.spatial import KDTree
from mad.utils.logger import SourceLogger


@dataclass
class BallisticTable:
    table: NDArray
    alt_scale: float
    vel_scale: float
    gam_scale: float
    range_scale: float
    kdtree: KDTree  # Built from normalized (alt, vel, gamma) inputs
    kdtree_range: KDTree  # Built from normalized (alt, vel, range_rad) for range-based gamma lookup


BALLISTIC_FIELD_NAMES = ["altitude_m", "velocity_m_s", "gamma_rad", "range_rad", "range_km"]

logger = SourceLogger()


def load_ballistic_csv(table_name):
    """Load a ballistic table from a CSV file and return a DataFrame"""
    ballistic_values = load_ballistic_table(table_name)
    df = pd.DataFrame({k: ballistic_values.table[:, i] for i, k in enumerate(BALLISTIC_FIELD_NAMES)})
    df["altitude_km"] = np.round(df["altitude_m"] / 1000, 3)
    df["gamma_deg"] = np.round(df["gamma_rad"] * 180 / np.pi, 3)
    df["altitude_m"] = np.round(df["altitude_m"], 3)

    return df


def load_ballistic_table(table_name: str) -> BallisticTable | None:
    """Load a ballistic table from a CSV file and create the BallisticTable object.
    The CSV file must have columns: altitude_m, velocity_m_s, gamma_rad, range_rad.
    The first row must be a header with exactly those column names.
    """

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
