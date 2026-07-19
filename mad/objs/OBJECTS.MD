# Objects

Simulation objects are responsible for their own motion: they compute forces, integrate their state, and can spawn new objects (e.g. separated stages or released payloads). The *direction* of thrust is delegated to Guidance classes; objects only know their current thrust magnitude.

---

## Class hierarchy

```
MovableObj                          # geometric point: position, velocity, id, active flag
└── BallisticObj(MovableObj, SimulationInterface)
    │   Adds: mass, area, Cd (drag coefficient)
    ├── Projectile                  # unguided ballistic (gravity + drag)
    ├── Payload(BallisticObj, GuidedObj)   # separating warhead / upper stage
    └── … (Missile subclasses, Rockets, Satellites, …)

SimulationInterface (ABC)           # contract for the simulation loop
GuidedObj (ABC)                     # mixin for propulsion / guidance
ReleasableConfig (Protocol)         # factory contract for deployable payloads
```

---

## Base classes

### `MovableObj` (`base.py`)

The root of every simulated entity. Tracks a 3-D `position` and `velocity` (NumPy arrays, metres / m·s⁻¹), an `active` flag, a human-readable `name`, and a unique `_id` string used as a dictionary key by `HistoryCollector`.

Key helpers:
- `distance(other)` — Euclidean separation in metres.
- `normalize` — unit vector along `position` (radial direction).
- `degrade()` — marks the object inactive; overridden by subclasses for extra teardown.

`MovableObj` has no mass and is not connected to the simulation loop. It is only a geometric carrier.

### `SimulationInterface` (ABC, `base.py`)

The contract the `Simulation` loop depends on. Any object that participates in the loop must satisfy it.

| Method | Responsibility |
|---|---|
| `update(dt, command)` | Advance internal state (staging timers, fuel consumption, guidance computation). **May return a list of new `BallisticObj`s** that are immediately injected into the simulation. |
| `accelerations(planet)` | Return the net acceleration vector (m·s⁻²) at the current state: gravity + drag + thrust + … |
| `integrate(dt, planet)` | Advance `position` and `velocity` by one time step. The default implementation uses **Velocity Verlet** and calls `accelerations` twice per step. |

`update` is always called before `integrate` within the same time step. This ordering means internal state (e.g. staging, fuel burn) is always consistent before the integrator asks for forces.

### `BallisticObj(MovableObj, SimulationInterface)` (`base.py`)

The standard base for any physical object subject to gravity and aerodynamic drag. Adds `mass` (kg), `area` (m²), and `Cd` (dimensionless drag coefficient), all read-only properties. It does not add guidance or propulsion; those are contributed by `GuidedObj`.

### `GuidedObj` (ABC, `base.py`)

A mixin for objects that carry propellant. Requires two abstract properties:

- `burned_fraction` — fraction of propellant consumed, in [0, 1]. Used by guidance laws to determine available manoeuvre authority.
- `thrust_acc` — current maximum propulsive acceleration (m·s⁻²).

Concrete subclasses combine `BallisticObj + GuidedObj` in the MRO and implement these properties based on their staging / fuel model.

---

## Config / factory pattern

Every concrete object is constructed via a paired `*Config` dataclass (e.g. `ProjectileConfig`, `PayloadConfig`). Configs hold all physical parameters and expose a `create(position, velocity, t)` factory method that instantiates the object. This keeps construction logic out of the simulation loop and makes configs serialisable and reusable.

Configs that represent deployable payloads satisfy the `ReleasableConfig` protocol (a `name` field plus `create()`). The simulation never needs to know the concrete type; it calls `config.create(...)` and receives a `BallisticObj`.

---

## Object lifecycle

1. **Created** — position and velocity set at construction; `active = True`.
2. **Active** — participates in every simulation step: `update` → `integrate`.
3. **Spawning** — `update` returns a list of child objects (e.g. a separated stage). The loop appends them to the active list immediately.
4. **Inactive** — `active = False`. The object is skipped by the loop but retained in the history. Inactivation is triggered by:
   - Ground impact (detected inside `accelerations` or `update`).
   - Collision (set externally by `apply_collisions`).
   - Propellant exhaustion or mission completion.
   - Explicit call to `degrade()`.

---

## Simulation loop interaction

```
for each time step:
    for each active object:
        new_objs = obj.update(dt, command)   # internal state, may spawn children
        active_list.extend(new_objs)
    for each active object:
        obj.integrate(dt, planet)            # physics step
    collision_detection_and_marking()
    history_collector.record(active_list)
```

`Simulation` is the full orchestrator. `run_simple_simulation` is a lightweight alternative with no history collection.

---

## Physics conventions

- Coordinates: 3-D ECEF-like vectors (`numpy` arrays). Use `mad.utils.to_vec3` to normalise inputs.
- Units: metres, m·s⁻¹, seconds throughout.
- Gravity and atmospheric drag are provided by `Planet` objects and queried via `planet.gravity(obj)` / `planet.drag(obj)`.
- Integrator: **Velocity Verlet** — second-order accurate, time-reversible, good energy conservation for orbital trajectories.

---

## Adding a new simulated object

1. Subclass `BallisticObj` (and `GuidedObj` if propelled).
2. Implement `accelerations(planet)` — sum all relevant force terms divided by mass.
3. Implement `update(dt, command)` — advance timers/fuel; return spawned objects or `None`.
4. Override `integrate(dt, planet)` only if the default Velocity Verlet is insufficient.
5. Create a `*Config` dataclass with a `create()` factory method.
6. If the object can be deployed as a payload from another object, ensure the config satisfies `ReleasableConfig`.