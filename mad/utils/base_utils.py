from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mad.configs.physics_cfg import VOXEL_SIZE


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
