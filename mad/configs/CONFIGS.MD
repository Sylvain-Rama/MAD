# Configuration files

This folder holds preset physical parameters for every category of object in MAD. Configs are plain Python **dicts**, which keeps them human-readable, easy to serialise, and straightforward to pass to the `*Config` dataclasses that live in `mad/objs/`.

---

## Role in the object creation pipeline

```
mad/configs/          mad/objs/
  <category>_cfg.py  →  <Object>Config (dataclass)  →  <Object> instance
  (dict of params)       (typed, has .create())          (participates in sim)
```

A typical usage pattern:

```python
from mad.configs.projectiles_cfg import rock
from mad.objs.projectiles import ProjectileConfig, Projectile

cfg = ProjectileConfig(**rock)
obj = cfg.create(position=[...], velocity=[...], t=0.0)
```

---

## Unit conventions and helpers (`physics_cfg.py`)

All simulation quantities use **SI base units** (metres, kg, seconds).  `physics_cfg.py` provides the physical constants and conversion factors used across config files:

| Symbol | Value | Meaning |
|---|---|---|
| `G` | 6.674 × 10⁻¹¹ m³·kg⁻¹·s⁻² | Gravitational constant |
| `G0` | 9.80665 m·s⁻² | Standard gravity (Earth surface) |
| `POUNDS_FORCE_TO_NEWTONS` | 4.44822 N/lbf | Imperial force conversion |
| `MACHS_TO_METERS_S` | 340.29 m·s⁻¹ | Speed of sound at sea level |
| `MILES_TO_METERS` | 1609.34 m/mi | Distance conversion |

Import these constants in other config files rather than hard-coding magic numbers.

---

## Config files by category

### `planets_cfg.py` — Planetary bodies
Defines the gravitational and atmospheric properties consumed by `Planet` / `PlanetConfig`:
- `EARTH_SETTINGS` — standard Earth (radius, mass, sea-level air density, scale height).
- `MOON_SETTINGS` — the Moon (no atmosphere: `rho0 = 0`).

### `projectiles_cfg.py` — Unguided ballistic objects
Simple point masses subject to gravity and aerodynamic drag.  Keys: `mass`, `ref_radius`, `Cd`, `name`.
- `rock` — generic 1 kg sphere.
- `rock_no_drag` — drag-free test object (`ref_radius = 0`, `Cd = 0`).

### `satellites_cfg.py` — Orbital objects
Same schema as projectiles (gravity + optional drag).
- `sputnik` — Sputnik-1 replica (86 kg spherical satellite).

### `warheads_cfg.py` — Re-entry vehicles / kinetic impactors
- `B53_warhead` — representative thermonuclear re-entry vehicle.
- `rod_of_god` — kinetic energy impactor (dense, high-drag tungsten rod).

### `ballistic_objects_cfg.py` — Multi-stage rockets
Each entry is a **list of stage dicts**.  Every stage dict contains: `dry_mass`, `full_mass` (or `propellant_mass`), `thrust` (N), `Isp` (s), `ref_radius` (m), `Cd`, `name`.
- `titan1_stages` — Titan I (two liquid-fuel stages).
- `titan2_stages` — Titan II (two liquid-fuel stages).
- `minuteman_stages` — Minuteman III (three solid-fuel stages).

### `cruise_missiles_cfg.py` — Powered, guided missiles
Pairs of dicts: one for the missile body, one for its guidance parameters.  Missile keys: `mass`, `ref_radius`, `Cd`, `thrust_acc`, `max_range_m`, `yield_kt`, `name`.  Guidance keys vary by guidance class but typically include `max_speed_m_s`, `cruise_altitude_m`, `altitude_settling_time_s`.
- `tomahawk` / `tomahawk_guidance` — subsonic land-attack cruise missile.
- `aster_30` / `aster_30_guidance` — supersonic surface-to-air interceptor.
- `V1` / `V1_guidance` — WWII-era pulse-jet flying bomb.

### `radars_cfg.py` — Detection systems
Keys: `name`, `max_range_m`, `voxel_size` (broad-phase detection grid cell size in metres).
- `BMEWS` — Ballistic Missile Early Warning System (long-range, large voxels).
- `AN_SPY_1` — Aegis shipboard phased-array radar (medium-range, fine voxels).

---

## Adding a new preset

1. Identify the appropriate `*_cfg.py` file (or create one if needed).
2. Add a plain dict with the required keys for the target `*Config` dataclass.
3. Import any unit conversion factors from `physics_cfg.py` rather than embedding raw numbers.
4. Export the symbol from `mad/configs/__init__.py` if it should be part of the public API.