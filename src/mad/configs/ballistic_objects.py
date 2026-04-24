"""Ballistic objects."""

import numpy as np

titan_stage_1 = {
    "dry_mass": 4000.0,
    "propellant_mass": 76203.0 - 4000.0,
    "thrust": 1900 * 1000,  # N
    "Isp": 290.0,
    "area": np.pi * 3.1**2,
    "Cd": 1.08,  # smooth, long cylinder.
    "name": "Stage1",
}

titan_stage_2 = {
    "dry_mass": 1725.0,
    "propellant_mass": 28939 - 1725.0,
    "thrust": 356 * 1000,  # N
    "Isp": 316.0,
    "area": np.pi * 2.3**2,
    "Cd": 1.08,  # smooth, long cylinder.
    "name": "Stage2",
}

rock = {
    "mass": 1.0,  # kg
    "area": np.pi * 0.05**2,  # m^2
    "Cd": 0.47,  # Sphere
    "name": "Rock",
}

rock_no_drag = {
    "mass": 1.0,  # kg
    "area": 0.0,  # m^2
    "Cd": 0.0,  # No drag
    "name": "RockNoDrag",
}

B53_warhead = {
    "mass": 4000.0,  # kg
    "area": np.pi * 0.5**2,  # m^2
    "Cd": 0.47,  # Sphere
    "name": "B53Warhead",
    "yield_kt": 9.0 * 1000,  # kt
}

rod_of_god = {
    "mass": 8000.0,  # kg
    "area": np.pi * 0.1**2,  # m^2
    "Cd": 1.08,  # Long cylinder
    "name": "RodOfGod",
}
