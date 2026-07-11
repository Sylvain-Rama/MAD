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

### Class hierarchy

```
MovableObj                     # geometric point: position, velocity, distance(), normalize
└── BallisticObj(MovableObj, SimulationInterface)
    │   Adds: mass, area, Cd (drag coefficient)
    ├── Projectile             # pure ballistic (gravity + drag), Velocity Verlet integrator
    ├── Payload(BallisticObj, GuidedObj)   # warhead / separating stage with optional guidance
    └── (Missile subclasses)

SimulationInterface (ABC)      # must implement: update(dt), accelerations(planet), integrate(dt, planet)
GuidedObj (ABC)                # mixin for guided objects; must implement: burned_fraction, thrust_acc
```

### Simulation loop (`mad/simulation.py`)

`Simulation(max_time, dt)` orchestrates the loop:

1. Each active object's `update(dt)` is called (internal state / staging), then `integrate(dt, planet)` advances position/velocity.
2. `update()` may return new `BallisticObj` instances (e.g. separated stages) that are appended to the object list.
3. `apply_collisions(objs, collisions)` marks colliding pairs inactive.
4. A convenience function `run_simple_simulation(objs, planet, dt, max_time)` is available for quick runs without collision detection.

Collision detection has been extracted to `mad/detection.py` (`CollisionDetector` class):

- `build_voxel_grid(objs)` — partitions active objects into a spatial hash grid (voxel size in **km**, default 50 km).
- `detect_collisions(objs, grid, collision_radius_m)` — broadphase via 26-neighbour voxel check, narrowphase via exact distance test.

### Physics conventions

- All distances in **metres**, velocities in **m/s**, time in **seconds**.
- Positions are 3-D ECEF-like vectors (`numpy` arrays via `mad.utils.to_vec3`).
- Gravity and drag are provided by `Planet` objects (`mad/objs/planets.py`).
- Integrator: **Velocity Verlet** (used in `Projectile.integrate`).

### Adding a new simulated object

1. Subclass `BallisticObj` (and optionally `GuidedObj`).
2. Implement `accelerations(planet)`, `integrate(dt, planet)`, and `update(dt)`.
3. Initialise a `History` instance and call `history.update(...)` inside `integrate`.
4. Create a `*Config` dataclass to hold construction parameters (see `ProjectileConfig` / `PayloadConfig`).

## Key directories

| Path | Purpose |
|------|---------|
| `mad/objs/` | Simulation object classes |
| `mad/configs/` | Physical constants & object presets |
| `mad/simulation.py` | Main `Simulation` orchestrator and `run_simple_simulation` helper |
| `mad/detection.py` | `CollisionDetector`: voxel grid construction and collision detection |
| `mad/utils.py` | Helper utilities (`to_vec3`, `extract_history`, …) |
| `notebooks/` | Interactive validation / exploration notebooks |
| `tests/` | Pytest suite (currently minimal) |
