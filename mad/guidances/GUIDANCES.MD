# Guidances

Guidance classes compute a desired acceleration direction (and optionally magnitude) at each simulation step, and report the current flight state via `GuidanceResults`.

Both `Guidance` and `GuidanceManager` expose the same `get_guidance(missile, t)` interface and are interchangeable from the caller's perspective.

---

## Key abstractions

### `GuidableObj` (Protocol)

Guidance classes are decoupled from concrete missile types via a structural protocol. Any object that exposes `position`, `velocity`, `normalize`, `burned_fraction`, `has_thrust`, and `thrust_acc` can be steered by any guidance law — no inheritance required.

### `GuidanceResults`

The single return type of every guidance call:

```python
@dataclass
class GuidanceResults:
    direction: NDArray           # Unit vector — desired acceleration direction
    state: GuidanceStates        # Current flight phase
    gamma: float | None          # Flight-path angle command (rad), if applicable
    magnitude: float | None      # Desired acceleration magnitude (m/s²), if applicable
    release_velocity: NDArray | None  # Optimal payload release velocity (m/s)
    next_guidance: bool          # True → GuidanceManager should advance to the next law
```

### `GuidanceStates`

Enum encoding the current flight phase: `IDLE`, `POWERED`, `COASTING`, `TERMINAL`, `RELEASE_PAYLOAD`, `DETONATE`. Guidance laws set this to signal the missile object what action to take (e.g. release a payload, detonate).

---

## Sequencing — `GuidanceManager`

Complex multi-phase missions are built by composing simple, single-phase `Guidance` instances inside a `GuidanceManager`, rather than encoding phase logic inside a single monolithic class:

```python
guidance = GuidanceManager([boost_guidance, coast_guidance, terminal_guidance])
```

The manager calls the current guidance's `get_guidance()` each step and advances to the next one as soon as `next_guidance` is `True`. When all laws are exhausted it returns `IDLE`.

---

## Interrupts

Each `Guidance` accepts an optional `interrupt_fn: (GuidanceInterrupts) -> bool`. On every step, `Guidance.update()` evaluates this function and sets `next_guidance = True` when it fires. `GuidanceInterrupts` carries a snapshot of the current simulation state:

```python
@dataclass
class GuidanceInterrupts:
    missile: MovableObj | None
    target: MovableObj | None
    planet: Planet | None
    t: float                    # simulation time (s)
    travelled_distance_m: float # accumulated distance flown (m)
```

Ready-made interrupt functions live in `interrupt_guidances.py` (e.g. `interrupt_at_t`, `interrupt_at_altitude`, `interrupt_at_surface_distance_to_target`). Any callable with the right signature can be used, including lambdas and `functools.partial` wrappers.

---

## Implementing a new guidance law

Subclass `Guidance` and implement the single abstract method:

```python
class MyGuidance(Guidance):
    def _compute_guidance(self, missile: GuidableObj, t: float = 0.0) -> GuidanceResults:
        ...
```

`get_guidance()` (the public entry point) calls `update()` first — which ticks time, accumulates `travelled_distance`, and evaluates the interrupt — then delegates to `_compute_guidance()`. Subclasses should not override `get_guidance()`.

Shared spherical-geometry helpers available on every `Guidance` instance: `central_angle`, `local_frame` (returns `r_hat`, `t_hat`), `optimal_gamma`, and `_resolve_t_hat_sign`.