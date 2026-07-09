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
from mad.configs import EARTH_SETTINGS
from mad.simulation import run_simple_simulation
from mad.utils.logger import SourceLogger, configure_logger
from mad.utils.ballistic_tables import BALLISTIC_FIELD_NAMES
from ranges_cfgs import SIM_PARAMETERS, AVAILABLE_OBJECTS, SimParameters

configure_logger(active_sources=["I/O"])
logger = SourceLogger()


def parse_args():
    parser = ArgumentParser(description="Tabulate ballistic range for Earth.")
    parser.add_argument(
        "--object",
        "-o",
        type=str,
        default="V1",
        help="Ballistic Object config to use (default: V1). Available: " + ", ".join(AVAILABLE_OBJECTS.keys()),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="V1",
        help="Simulation config to use (default: V1). Available: " + ", ".join(SIM_PARAMETERS.keys()),
    )
    return parser.parse_args()


# ── multiprocessing helpers ──────────────────────────────────────────────────
_worker_planet: Planet | None = None
_worker_config: ProjectileConfig | None = None
_worker_simconfig: SimParameters | None = None


def _pool_initializer(planet: Planet, config: ProjectileConfig, simconfig: SimParameters) -> None:
    """Copy planet, config, and simconfig into each worker process once."""
    global _worker_planet, _worker_config, _worker_simconfig
    _worker_planet = planet
    _worker_config = config
    _worker_simconfig = simconfig


def _simulate_row(args: tuple) -> dict:
    """Worker entry-point: unpack grid point, call simulate, return row dict."""
    assert _worker_planet is not None
    assert _worker_config is not None
    assert _worker_simconfig is not None
    alt_km, v_kms, gamma_deg = args
    r0 = _worker_planet.radius + alt_km * 1e3
    v0 = v_kms * 1e3
    gamma_rad = np.radians(gamma_deg)
    central_angle, range_km = simulate(_worker_planet, _worker_config, _worker_simconfig, r0, v0, gamma_rad)

    return dict(
        altitude_m=alt_km * 1e3,
        velocity_m_s=v0,
        gamma_rad=gamma_rad,
        range_rad=central_angle,
        range_km=range_km,
    )


def simulate(
    planet: Planet, projconfig: ProjectileConfig, simconfig: SimParameters, r0: float, v0: float, gamma_rad: float
) -> tuple[float, float]:
    """
    Simulate a ballistic arc in the (x, y) plane starting from radius r0,
    speed v0, elevation angle gamma_rad above local horizontal.

    Returns the central angle covered (rad) until ground impact,
    and the range in kilometers, or (np.nan, np.nan) if the object does not return within MAX_TIME.
    """
    # Place the missile at (r0, 0, 0); build velocity in the x-y plane.
    pos = np.array([r0, 0.0, 0.0], dtype=float)
    r_hat = pos / r0
    t_hat = np.array([0.0, 1.0, 0.0])  # tangential (prograde) direction

    vel = v0 * (np.sin(gamma_rad) * r_hat + np.cos(gamma_rad) * t_hat)

    projconfig.position = pos.tolist()
    projconfig.velocity = vel.tolist()
    obj = Projectile(projconfig)
    start_pos = obj.position.copy()

    simulated_object = run_simple_simulation([obj], planet, dt=simconfig.dt, max_time=simconfig.max_time)

    if not simulated_object:
        logger["I/O"].warning(f"Simulation failed for r0={r0}, v0={v0}, gamma_rad={gamma_rad}.")
        return np.nan, np.nan

    if simulated_object[0].active:
        logger["I/O"].warning(
            f"Simulation did not return to ground within max_time for r0={r0}, v0={v0}, gamma_rad={gamma_rad}."
        )
        return np.nan, np.nan

    final_pos = simulated_object[0].position

    cos_a = np.clip(
        np.dot(start_pos, final_pos) / (np.linalg.norm(start_pos) * np.linalg.norm(final_pos)),
        -1.0,
        1.0,
    )
    central_angle = float(np.arccos(cos_a))
    range_km = central_angle * planet.radius / 1000

    return central_angle, range_km


def main() -> None:
    args = parse_args()
    planet = Planet(PlanetConfig(**EARTH_SETTINGS))
    object = AVAILABLE_OBJECTS[args.object]
    config = SIM_PARAMETERS[args.config]

    simconfig = SimParameters(**config)

    ballistic_config = ProjectileConfig(
        position=[0.0, 0.0, 0.0],
        mass=float(object["mass"] if "mass" in object else object["dry_mass"]),
        ref_radius=float(object["ref_radius"]),
        Cd=float(object["Cd"]),
        name=str(object["name"]),
    )

    grid = [
        (alt_km, v_kms, gamma_deg)
        for alt_km in simconfig.altitudes_km
        for v_kms in simconfig.velocities_kms
        for gamma_deg in simconfig.gammas_deg
    ]
    total = len(grid)
    logger["I/O"].info(f"Using config '{args.config}' for ballistic object properties.")
    logger["I/O"].info(f"Computing {total} trajectories  (dt={simconfig.dt} s, max_time={simconfig.max_time} s) …")

    n_workers = cpu_count() - 1  # leave one core free for the main process
    logger["I/O"].info(f"Using {n_workers} worker processes.")

    with Pool(
        processes=n_workers,
        initializer=_pool_initializer,
        initargs=(planet, ballistic_config, simconfig),
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
