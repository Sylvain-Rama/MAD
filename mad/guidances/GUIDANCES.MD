# Guidances


Guidance and GuidanceManager classes are responsible for giving the movement direction and the flight state, via the guidance results:

```
@dataclass
class GuidanceResults:
    """Results of a guidance law computation.
    Contains the desired acceleration direction, magnitude, and the current guidance state.
    """

    direction: NDArray  # Unit vector indicating the desired direction of acceleration (m/s²)
    state: GuidanceStates
    gamma: float | None = None  # Optional angular velocity command for advanced guidance laws
    magnitude: float | None = None  # Optional desired acceleration magnitude (m/s²)
    release_velocity: NDArray | None = None  # Optimal payload release velocity vector (m/s)
    next_guidance: bool = False  # Whether to switch to the next guidance in the guidance list.
```

Both Guidnce & GuidanceManager can be used in the same way, as they expose the same ```get_direction()``` method.
However, GuidanceManager classes can use multiple Guidances nd their interrupt functions to create complex guidances.