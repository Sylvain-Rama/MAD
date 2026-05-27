#!/usr/bin/env python3
"""
Pre-tabulate ballistic range (central angle, degrees) as a function of
  (altitude_km, velocity_km_s, gamma_deg)
for Earth, including atmospheric drag from the Planet model.

Drag only acts below atmosphere_height (8 500 m); above that the trajectory
is Keplerian.  The table can be used by guidance code as a lookup to decide
when to cut the engine — more accurate than the closed-form vacuum formula.

Usage:
    python scripts/tabulate_ballistic_range.py

Output:
    tables/ballistic_table.csv
"""

import os
import csv
import numpy as np
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from mad.objs.planets import Planet, PlanetConfig
from mad.objs.projectiles import Projectile, ProjectileConfig
from mad.configs.planets import EARTH_SETTINGS
from mad.configs.warheads import B53_warhead, rod_of_god
from mad.simulation import run_simple_simulation
from mad.utils.logger import SourceLogger, configure_logger
from mad.utils.ballistic_tables import BALLISTIC_FIELD_NAMES

configure_logger(active_sources=["I/O"])
logger = SourceLogger()

AVAILABLE_OBJECTS = {
    "B53_warhead": B53_warhead,
    "rod_of_god": rod_of_god,
}


DT = 10.0  # time step (s) — coarse is intentional
MAX_TIME = 3600.0  # 2 h; enough for any sub-orbital ballistic arc

ALTITUDES_KM = np.arange(500, 1201, 10)
VELOCITIES_KMS = np.arange(8, 10, 0.2)
GAMMAS_DEG = np.arange(-50, 20, 2)


def parse_args():
    parser = ArgumentParser(description="Tabulate ballistic range for Earth.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="rod_of_god",
        help="Ballistic Object config to use (default: rod_of_god). Available: " + ", ".join(AVAILABLE_OBJECTS.keys()),
    )
    return parser.parse_args()


# ── multiprocessing helpers ──────────────────────────────────────────────────
_worker_planet: Planet | None = None
_worker_config: ProjectileConfig | None = None


def _pool_initializer(planet: Planet, config: ProjectileConfig) -> None:
    """Copy planet and config into each worker process once."""
    global _worker_planet, _worker_config
    _worker_planet = planet
    _worker_config = config


def _simulate_row(args: tuple) -> dict:
    """Worker entry-point: unpack grid point, call simulate, return row dict."""
    assert _worker_planet is not None
    assert _worker_config is not None
    alt_km, v_kms, gamma_deg = args
    r0 = _worker_planet.radius + alt_km * 1e3
    v0 = v_kms * 1e3
    gamma_rad = np.radians(gamma_deg)
    result = simulate(_worker_planet, _worker_config, r0, v0, gamma_rad)
    return dict(
        altitude_m=alt_km * 1e3,
        velocity_m_s=v0,
        gamma_rad=gamma_rad,
        range_rad=result,
    )


def simulate(planet: Planet, config: ProjectileConfig, r0: float, v0: float, gamma_rad: float) -> float | None:
    """
    Simulate a ballistic arc in the (x, y) plane starting from radius r0,
    speed v0, elevation angle gamma_rad above local horizontal.

    Returns the central angle covered (rad) until ground impact,
    or np.nan if the object does not return within MAX_TIME.
    """
    # Place the missile at (r0, 0, 0); build velocity in the x-y plane.
    pos = np.array([r0, 0.0, 0.0], dtype=float)
    r_hat = pos / r0
    t_hat = np.array([0.0, 1.0, 0.0])  # tangential (prograde) direction

    vel = v0 * (np.sin(gamma_rad) * r_hat + np.cos(gamma_rad) * t_hat)

    config.position = pos.tolist()
    config.velocity = vel.tolist()
    obj = Projectile(config)
    start_pos = obj.position.copy()

    simulated_object = run_simple_simulation([obj], planet, dt=DT, max_time=MAX_TIME)

    final_pos = simulated_object[0].position

    cos_a = np.clip(
        np.dot(start_pos, final_pos) / (np.linalg.norm(start_pos) * np.linalg.norm(final_pos)),
        -1.0,
        1.0,
    )
    return float(np.arccos(cos_a))


def main() -> None:
    args = parse_args()
    planet = Planet(PlanetConfig(**EARTH_SETTINGS))
    config = AVAILABLE_OBJECTS[args.config]

    ballistic_config = ProjectileConfig(
        position=[0.0, 0.0, 0.0],
        mass=float(config["mass"] if "mass" in config else config["dry_mass"]),
        ref_radius=float(config["ref_radius"]),
        Cd=float(config["Cd"]),
        name=str(config["name"]),
    )

    grid = [
        (alt_km, v_kms, gamma_deg) for alt_km in ALTITUDES_KM for v_kms in VELOCITIES_KMS for gamma_deg in GAMMAS_DEG
    ]
    total = len(grid)
    logger["I/O"].info(f"Using config '{args.config}' for ballistic object properties.")
    logger["I/O"].info(f"Computing {total} trajectories  (dt={DT} s, max_time={MAX_TIME} s) …")

    n_workers = cpu_count() - 1  # leave one core free for the main process
    logger["I/O"].info(f"Using {n_workers} worker processes.")

    with Pool(
        processes=n_workers,
        initializer=_pool_initializer,
        initargs=(planet, ballistic_config),
    ) as pool:
        rows = list(
            tqdm(
                pool.imap(_simulate_row, grid),
                total=total,
                desc="Simulating trajectories",
            )
        )

    out_path = os.path.join("src/mad/tables", args.config + ".csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BALLISTIC_FIELD_NAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger["I/O"].success(f"Saved {total} rows to {out_path}")


if __name__ == "__main__":
    main()
