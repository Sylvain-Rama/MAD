"""Satellite configurations.
For the moment, they are siple Projectile equivalents, but they can be extended to have propulsion and guidance systems.
See objs/satellites.py for the implementation of satellites.
"""

# See https://en.wikipedia.org/wiki/Sputnik_1
sputnik = {
    "mass": 86.0,  # kg
    "ref_radius": 0.29,  # m
    "Cd": 0.47,  # Spherical
    "name": "Sputnik",
}
