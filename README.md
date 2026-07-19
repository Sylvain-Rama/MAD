# MAD — Missile and Ballistic Simulation Library

MAD is a Python 3.12+ library for simulating 3-D ballistic and guided-missile trajectories in a planetary environment. It is designed around composable abstractions: physics objects, guidance laws, and a simulation orchestrator are kept strictly separate, making it easy to mix and match components.

It is a simplified simulation designed to easily reproduce rocket launches, cruise missiles, satellites orbiting earth, etc...

Work In Progress!

---

## Table of contents

1. [Concepts](#concepts)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Project structure](#project-structure)
5. [Running tests and linting](#running-tests-and-linting)

---

## Concepts

### Simulation objects

Every entity that moves in the simulation is a subclass of `BallisticObj`, which itself combines `MovableObj` (position, velocity) with `SimulationInterface` (the contract the simulation loop depends on). Guided objects additionally mix in `GuidedObj` to expose propulsion state.

The simulation loop calls `update(dt)` — advancing internal state and optionally spawning child objects — then `integrate(dt, planet)` to step position and velocity forward using **Velocity Verlet** integration.

→ See [mad/objs/OBJECTS.MD](mad/objs/OBJECTS.MD) for the full class hierarchy, lifecycle, and a guide to implementing new objects.

### Guidance

Guidance classes compute a desired **acceleration direction** (and optionally magnitude) at each simulation step. They are decoupled from concrete missile types through the `GuidableObj` structural protocol.

Multi-phase missions are built by composing single-phase `Guidance` instances inside a `GuidanceManager`. Each guidance law can carry an optional `interrupt_fn` that triggers advancement to the next phase.

→ See [mad/guidances/GUIDANCES.MD](mad/guidances/GUIDANCES.MD) for the full API, interrupt system, and implementation guide.

### Configuration presets

Physical parameters for all object categories (planets, rockets, warheads, cruise missiles, radars …) live in `mad/configs/` as plain Python dicts. These are intentionally kept separate from the typed `*Config` dataclasses in `mad/objs/`, providing a human-readable layer that is easy to serialise and extend.

→ See [mad/configs/CONFIGS.MD](mad/configs/CONFIGS.MD) for the layout of every config file and the unit-conversion helpers.

### Physics conventions

| Quantity | Unit |
|---|---|
| Distance | metres (m) |
| Velocity | m·s⁻¹ |
| Time | seconds (s) |
| Mass | kg |

Positions are 3-D ECEF-like NumPy vectors. Use `mad.utils.to_vec3` to normalise inputs. Gravity and atmospheric drag are provided by `Planet` objects (`mad/objs/planets.py`).

---

## Installation

The project uses [uv](https://github.com/astral-sh/uv) and is installed in editable mode.

```bash
# Install all dependencies (including dev extras)
uv sync --all-groups
```

Alternatively, the repository ships a `Dockerfile` / `docker-compose.yml` for a fully reproducible environment:

```bash
docker compose up
```

The Docker image also starts JupyterLab, giving instant access to the exploration notebooks in `notebooks/`.

---

## Quick start

```python
import numpy as np
from mad.objs.planets import Planet, PlanetConfig
from mad.objs.projectiles import Projectile, ProjectileConfig
from mad.configs.planets_cfg import EARTH_SETTINGS
from mad.simulation import run_simple_simulation

# Build the planet
earth = Planet(PlanetConfig(**{**EARTH_SETTINGS, "position": [0.0, 0.0, 0.0]}))

# Place a 1 kg rock at 500 km altitude, give it a horizontal kick
r0 = earth.radius + 500_000.0
cfg = ProjectileConfig(mass=1.0, ref_radius=0.05, Cd=0.47)
rock = cfg.create(
    position=[r0, 0.0, 0.0],
    velocity=[0.0, 7_600.0, 0.0],  # roughly circular-orbit speed
)

# Run for 1 hour, 1 s time step
objects = run_simple_simulation([rock], earth, dt=1.0, max_time=3600.0)
print(objects[0].position)
```

For guided-missile and multi-stage rocket examples, see the notebooks in `notebooks/`.

---

## Project structure

```
mad/
├── simulation.py          # Simulation orchestrator and run_simple_simulation helper
├── objs/                  # Simulation object classes
│   ├── OBJECTS.MD         # ← architecture docs
│   ├── base.py            # MovableObj, BallisticObj, SimulationInterface, GuidedObj
│   ├── projectiles.py
│   ├── rockets.py
│   ├── satellites.py
│   ├── cruise_missiles.py
│   ├── planets.py
│   ├── radars.py
│   ├── launchers.py
│   └── battle_computers.py
├── guidances/             # Guidance laws
│   ├── GUIDANCES.MD       # ← architecture docs
│   ├── base_guidances.py
│   ├── ICBM_guidances.py
│   ├── cruise_missiles_guidances.py
│   ├── satellite_guidances.py
│   └── interrupt_guidances.py
├── configs/               # Physical parameter presets
│   ├── CONFIGS.MD         # ← architecture docs
│   ├── physics_cfg.py     # constants & unit conversions
│   ├── planets_cfg.py
│   ├── projectiles_cfg.py
│   ├── ballistic_objects_cfg.py
│   ├── cruise_missiles_cfg.py
│   ├── satellites_cfg.py
│   ├── warheads_cfg.py
│   └── radars_cfg.py
├── utils/                 # Helper utilities
│   ├── base_utils.py      # to_vec3, extract_history, …
│   ├── ballistic_tables.py
│   ├── plotters.py
│   └── logger.py
└── scripts/               # CLI tools
    └── tabulate_ballistic_range.py

notebooks/                 # Interactive validation / exploration
tests/                     # Pytest suite
```

---

## Running tests and linting

```bash
# Run the test suite
pytest

# Lint (line-length 120)
ruff check mad/

# Formatting check
black --check mad/

# Type-check (notebooks excluded)
pyrefly check mad/
```

Apply auto-fixes:

```bash
ruff check --fix mad/
black mad/
```
