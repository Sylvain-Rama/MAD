"""Planetary constants."""

EARTH_SETTINGS = {
    "position": [0.0, 0.0, 0.0],  # center of the sim
    "radius": 6371000.0,  # m
    "mass": 5.972e24,  # kg
    "name": "Earth",
    "rho0": 1.225,  # kg / m^3
    "atmosphere_height": 8500.0,  # m
}


MOON_SETTINGS = {
    "position": [384400000.0, 0.0, 0.0],  # m, average distance from Earth
    "radius": 1737400.0,  # m
    "mass": 7.346e22,  # kg
    "name": "Moon",
    "rho0": 0.0,  # kg / m^3, no atmosphere
    "atmosphere_height": 0.0,  # m
}
