import numpy as np
from numpy.typing import NDArray
from mad.logger import SourceLogger
from mad.scripts.tabulate_ballistic_range import FIELD_NAMES

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

def load_ballistic_table(table_name: str) -> NDArray | None:
    """Load a ballistic table from a CSV file. The CSV file must have columns: altitude_m, velocity_m_s, gamma_rad, range_rad.
    The first row must be a header with exactly those column names. Returns a 2D numpy array with those four columns."""
    
    file_path = f"/tables/{table_name}.csv"
    expected_columns = FIELD_NAMES
    try:
        with open(file_path, newline="") as f:
            header = f.readline().strip().split(",")
        if header != expected_columns:
            logger["I/O"].error(
                f"Ballistic table must have columns {expected_columns}. Got {header} instead."
            )
            return None
        table = np.loadtxt(file_path, delimiter=",", skiprows=1)
        return table
    
    except Exception as e:
        logger["I/O"].error(f"Error loading ballistic table from {file_path}: {e}")
        return None