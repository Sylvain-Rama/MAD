from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator
from mad.utils.logger import SourceLogger


@dataclass
class BallisticTable:
    table: NDArray
    altitudes: NDArray  # sorted unique altitude_m grid axis
    velocities: NDArray  # sorted unique velocity_m_s grid axis
    gammas: NDArray  # sorted unique gamma_rad grid axis
    range_interp: RegularGridInterpolator  # linear interpolation of range_rad over (altitude, velocity, gamma)
    name: str = ""  # optional name for the table, e.g. the warhead or missile type

BALLISTIC_FIELD_NAMES = ["altitude_m", "velocity_m_s", "gamma_rad", "range_rad", "range_km"]

logger = SourceLogger()


def load_ballistic_csv(table_name: str, dropna: bool = True) -> NDArray:
    """Load a ballistic table from a CSV file and return a DataFrame.

    Set ``dropna=False`` to keep rows with missing values as NaN instead of removing
    them — required by `load_ballistic_table` to preserve the regular grid shape.
    """
    # TODO: Use a proper path management solution and make this more robust to different environments.
    # For now, we assume the tables are in src/mad/tables and the script is run from the root of the repo.
    file_path = f"/app/mad/tables/{table_name}.csv"
    try:
        with open(file_path, newline="") as f:
            header = f.readline().strip().split(",")
        if header != BALLISTIC_FIELD_NAMES:
            raise ValueError(f"Ballistic table must have columns {BALLISTIC_FIELD_NAMES}. Got {header} instead.")
        table = np.genfromtxt(file_path, delimiter=",", skip_header=1, filling_values=np.nan)
    except Exception as e:
        raise ValueError(f"Failed to load ballistic table from {file_path}. Error: {e}")

    if dropna:
        n_before = len(table)
        table = table[~np.isnan(table).any(axis=1)]
        n_dropped = n_before - len(table)
        if n_dropped:
            logger["I/O"].warning(f"Dropped {n_dropped} row(s) with missing values from {file_path}.")

    return table


def load_ballistic_df(table_name: str) -> pd.DataFrame:
    """Load a ballistic table from a CSV file and return a DataFrame"""
    ballistic_values = load_ballistic_csv(table_name)
    df = pd.DataFrame({k: ballistic_values[:, i] for i, k in enumerate(BALLISTIC_FIELD_NAMES)})
    df["altitude_km"] = np.round(df["altitude_m"] / 1000, 3)
    df["gamma_deg"] = np.round(df["gamma_rad"] * 180 / np.pi, 3)
    df["altitude_m"] = np.round(df["altitude_m"], 3)

    return df


def load_ballistic_table(table_name: str) -> BallisticTable:
    """Load a ballistic table from a CSV file and create the BallisticTable object.
    The CSV file must have columns: altitude_m, velocity_m_s, gamma_rad, range_rad.
    The first row must be a header with exactly those column names.

    Rows must form a complete regular grid over (altitude_m, velocity_m_s, gamma_rad), as
    produced by mad/scripts/tabulate_ballistic_range.py — range_rad is linearly interpolated
    over that grid, which is far more precise than nearest-neighbor lookup for a given table
    size, letting tables stay compact even when covering both low- and high-altitude regimes.
    """

    table = load_ballistic_csv(table_name, dropna=False)

    altitudes = np.unique(table[:, 0])
    velocities = np.unique(table[:, 1])
    gammas = np.unique(table[:, 2])

    expected_rows = len(altitudes) * len(velocities) * len(gammas)
    if expected_rows != len(table):
        raise ValueError(
            f"Ballistic table '{table_name}' is not a complete regular grid: expected "
            f"{expected_rows} rows ({len(altitudes)} altitudes x {len(velocities)} velocities x "
            f"{len(gammas)} gammas) but got {len(table)}. The table must be generated as a full "
            "cartesian product of altitude/velocity/gamma values."
        )

    range_grid = table[:, 3].reshape(len(altitudes), len(velocities), len(gammas))

    n_missing = int(np.isnan(range_grid).sum())
    if n_missing:
        logger["I/O"].warning(
            f"Ballistic table '{table_name}' has {n_missing} missing entrie(s); "
            "queries interpolated near them will return NaN."
        )

    range_interp = RegularGridInterpolator(
        (altitudes, velocities, gammas),
        range_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,  # linearly extrapolate slightly outside the grid instead of failing
    )

    return BallisticTable(
        table=table,
        altitudes=altitudes,
        velocities=velocities,
        gammas=gammas,
        range_interp=range_interp,
        name=table_name,
    )
