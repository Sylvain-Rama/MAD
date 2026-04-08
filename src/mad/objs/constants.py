import numpy as np

G: float = 6.67408e-11  # m^3/(kg*s^2)
G0: float = 9.80665  # m/s^2, value for earth

EARTH_SETTINGS = {
    "position": [0.0, 0.0, 0.0],  # center of the sim
    "radius": 6371000.0,  # m
    "mass": 5.972e24,  # kg
    "spin_rate": 7.0882359e-5,  # rad/s
    "name": "Earth",
    "rho0": 1.225,  # kg / m^3
    "atmosphere_height": 8500.0,  # m
}

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
    "Isp": 308.0,
    "area": np.pi * 2.3**2,
    "Cd": 1.08,  # smooth, long cylinder.
    "name": "Stage2",
}
