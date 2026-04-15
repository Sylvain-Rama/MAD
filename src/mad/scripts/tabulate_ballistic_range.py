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
from tqdm import tqdm
from mad.objs.planets import Planet, PlanetConfig
from mad.objs.projectiles import Projectile, ProjectileConfig
from mad.configs.planets import EARTH_SETTINGS
from mad.configs.ballistic_objects import titan_stage_1, titan_stage_2, rock, rock_no_drag
from mad.simulation import run_simulation
from mad.logger import SourceLogger, configure_logger
from mad.utils import BALLISTIC_FIELD_NAMES

configure_logger(active_sources=["I/O"])
logger = SourceLogger()

AVAILABLE_OBJECTS = {
    "titan_stage_1": titan_stage_1,
    "titan_stage_2": titan_stage_2,
    "rock": rock,
    "rock_no_drag": rock_no_drag,
}


DT = 10.0  # time step (s) — coarse is intentional
MAX_TIME = 7200.0  # 4 h; enough for any sub-orbital ballistic arc

ALTITUDES_KM = np.arange(0, 1000, 10)
VELOCITIES_KMS = np.arange(0.5, 10, 0.5)
GAMMAS_DEG = np.arange(-10, 90, 2)


def parse_args():
    parser = ArgumentParser(description="Tabulate ballistic range for Earth.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="titan_stage_2",
        help="Ballistic Object config to use (default: titan_stage_2). Available: "
        + ", ".join(AVAILABLE_OBJECTS.keys()),
    )
    return parser.parse_args()


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

    simulated_object = run_simulation([obj], planet, dt=DT, max_time=MAX_TIME)

    final_pos = simulated_object[0].history.position[-1]

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
        mass=config["mass"] if "mass" in config else config["dry_mass"],
        area=config["area"],
        Cd=config["Cd"],
        name=config["name"],
    )

    grid = [
        (alt_km, v_kms, gamma_deg) for alt_km in ALTITUDES_KM for v_kms in VELOCITIES_KMS for gamma_deg in GAMMAS_DEG
    ]
    total = len(grid)
    logger["I/O"].info(f"Computing {total} trajectories  (dt={DT} s, max_time={MAX_TIME} s) …")

    rows = []
    for i, (alt_km, v_kms, gamma_deg) in tqdm(enumerate(grid), total=total, desc="Simulating trajectories"):
        r0 = planet.radius + alt_km * 1e3
        v0 = v_kms * 1e3
        gamma_rad = np.radians(gamma_deg)

        result = simulate(planet, ballistic_config, r0, v0, gamma_rad)

        # Fields are altitude (m), velocity (m/s), gamma (rad), range (rad)
        rows.append(
            dict(
                altitude_m=alt_km * 1e3,
                velocity_m_s=v0,
                gamma_rad=gamma_rad,
                range_rad=result,
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
