# MAD — Agent Instructions

MAD is a 3-D ballistic/missile simulation library written in Python 3.12+.
The package lives under `mad/` and is installed in editable mode via `uv`.

## Build & environment

```bash
# Install all dependencies (including dev extras)
uv sync --all-groups

# The project is also runnable inside the Docker dev container
docker compose up
```

## Linting & formatting

```bash
ruff check mad/          # fast lint (line-length 120)
black --check mad/       # formatting check (line-length 120)
pyrefly check mad/       # type-check (notebooks excluded)
```

Apply fixes:

```bash
ruff check --fix mad/
black mad/
```

## Tests

```bash
pytest                   # run all tests in tests/
```

The `tests/` directory is currently sparse; interactive validation lives in `notebooks/`.

## Simulation engine — key abstractions (`mad/objs/base.py`)

### Canonical runtime model

The codebase now uses a single dynamic-body abstraction: `Body`.

```
MovableObj                     # geometric carrier: position, velocity, id, active flag
└── BallisticObj(MovableObj)   # compatibility physics base with mass / area / Cd
    └── Body(BallisticObj)    # canonical runtime body used by the simulation loop

Body owns optional runtime components via composition:
- guidance: guidance object or strategy
- engine: propulsion / thrust model
- guidance_results: last guidance output
```

`BallisticObj` remains as a compatibility layer for older code paths, but new and refactored simulation code should target `Body` directly. The project intentionally avoids the deep inheritance tree previously used for `GuidedObj` and `SimulationInterface`.

### Simulation loop (`mad/simulation.py`)

`Simulation(max_time, dt)` orchestrates the loop:

1. Each active object is updated via `obj.update(dt, command)`.
2. `update()` may return new `Body` instances (for example, separated stages or released payloads), which are appended to the active list.
3. Each active object then advances with `obj.integrate(dt, planet)`.
4. A convenience function `run_simple_simulation(objs, planet, dt, max_time)` is available for quick runs.

### Physics conventions

- All distances in **metres**, velocities in **m/s**, time in **seconds**.
- Positions are 3-D ECEF-like vectors (`numpy` arrays via `mad.utils.to_vec3`).
- Gravity and drag are computed by `Planet` objects (`mad/objs/planets.py`).
- Integrator: **Velocity Verlet** is the default physics update pattern used by `BallisticObj.integrate`.

### Adding a new simulated object

1. Create a `Body` instance (or a subclass of `BallisticObj` if compatibility is needed).
2. Attach optional `guidance` and/or `engine` runtime components.
3. Implement `accelerations(planet)`, `integrate(dt, planet)`, and `update(dt, command)` as needed.
4. Create a `*Config` dataclass with a `create()` factory method when the object should be built from configuration.

For the current design, the object is the runtime state carrier and the behaviour is composed in via guidance and propulsion objects instead of inheriting through a rigid ABC stack.

## Key directories

| Path | Purpose |
|------|---------|
| `mad/objs/` | Simulation object classes |
| `mad/configs/` | Physical constants & object presets |
| `mad/simulation.py` | Main `Simulation` orchestrator and `run_simple_simulation` helper |
| `mad/utils/` | Helper utilities (`to_vec3`, `extract_history`, …) |
| `notebooks/` | Interactive validation / exploration notebooks |
| `tests/` | Pytest suite (currently minimal) |
