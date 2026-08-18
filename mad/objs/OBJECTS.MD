# Objects

Simulation objects are responsible for their own motion: they compute forces, integrate state, and can spawn new bodies such as separated stages or released payloads. The current design keeps a single runtime body model, with propulsion and guidance attached as optional runtime components rather than encoded in a deep inheritance tree.

---

## Canonical runtime model

The project now centers on `Body`, which is the simulation's canonical runtime object.

```
MovableObj                          # geometric point: position, velocity, id, active flag
└── BallisticObj(MovableObj)         # compatibility physics base with mass, area, Cd
    └── Body(BallisticObj)          # canonical dynamic body for the simulation loop

Body owns optional components via composition:
- guidance: guidance strategy or manager
- engine: propulsion model / thrust source
- guidance_results: last guidance outputs
```

`Body` is the baseline for the simulation loop, while `BallisticObj` remains available as a compatibility layer for older factories and external callers.

---

## Base classes

### `MovableObj` (`base.py`)

The root of every simulated entity. Tracks a 3-D `position` and `velocity` (NumPy arrays, metres / m·s⁻¹), an `active` flag, a human-readable `name`, and a unique `_id` string used as a dictionary key by `HistoryCollector`.

Key helpers:
- `distance(other)` — Euclidean separation in metres.
- `pos_norm` / `vel_norm` — normalized direction vectors.
- `degrade()` — marks the object inactive.

`MovableObj` has no mass and is not connected to the simulation loop. It is only a geometric carrier.

### `BallisticObj` (`base.py`)

A compatibility base for physical objects subject to gravity and aerodynamic drag. Adds `mass` (kg), `area` (m²), and `Cd` (dimensionless drag coefficient), as well as the standard physics API used by the integrator.

It includes the default Velocity Verlet `integrate(dt, planet)` and expects concrete subclasses to implement `accelerations(planet)`.

### `Body` (`base.py`)

The canonical dynamic body used by the active simulation. `Body` extends `BallisticObj` and adds optional runtime behaviour through composition:

- `guidance` — object with a `get_guidance()`-style interface
- `engine` — propulsion model with `thrust_acc`, `burned_fraction`, or update hooks
- `guidance_results` — last computed guidance result
- `t` — current simulation time for the object

The key idea is that an object is an instance of `Body` with the behaviour it needs attached, rather than a new class deep in an inheritance tree.

---

## Config / factory pattern

Every concrete object is constructed via a paired `*Config` dataclass (for example, `ProjectileConfig`, `RocketConfig`, `CruiseMissileConfig`). Configs hold all physical parameters and expose a `create(position, velocity, t)` factory method that instantiates the object.

A config that represents a deployable payload satisfies the `ReleasableConfig` protocol: it has a `name` field and a `create()` method. The simulation never needs to know the concrete type; it receives a `Body`-compatible object and runs it through the common loop.

---

## Object lifecycle

1. **Created** — position and velocity set at construction; `active = True`.
2. **Active** — participates in every simulation step: `update` → `integrate`.
3. **Spawning** — `update` returns a list of new child bodies (for example, a separated stage). The loop appends them to the active list immediately.
4. **Inactive** — `active = False`. The object is skipped by the loop but retained in history. Inactivation is triggered by:
   - Ground impact or planetary surface contact.
   - Explicit `degrade()` calls.
   - Propellant exhaustion or mission completion.
   - External game-state rules.

---

## Simulation loop interaction

```
for each time step:
    for each active object:
        new_objs = obj.update(dt, command)
        active_list.extend(new_objs)
    for each active object:
        obj.integrate(dt, planet)
    record_history(active_list)
```

`Simulation` is the orchestrator. `run_simple_simulation` is a lightweight alternative with no logging or history capture.

---

## Physics conventions

- Coordinates: 3-D ECEF-like vectors (`numpy` arrays). Use `mad.utils.to_vec3` to normalize inputs.
- Units: metres, m·s⁻¹, and seconds throughout.
- Gravity and atmospheric drag are provided by `Planet` objects and are queried via `planet.gravity(obj)` and `planet.drag(obj)`.
- Integrator: **Velocity Verlet** — second-order accurate and well suited to orbital and ballistic motion.

---

## Adding a new simulated object

1. Instantiate or subclass a `Body`/`BallisticObj` as appropriate.
2. Attach optional `guidance` and `engine` components via composition.
3. Implement `accelerations(planet)` and `update(dt, command)`.
4. Override `integrate(dt, planet)` only if the default Velocity Verlet logic is insufficient.
5. Create a `*Config` dataclass with a `create()` factory method.
6. If the object can be deployed from another object, ensure the config satisfies `ReleasableConfig`.

The intended model is not “one class per mission phase” but “one dynamic body with the required runtime behaviour attached.”