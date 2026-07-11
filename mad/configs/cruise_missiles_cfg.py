"""Configuration dicts for cruise missiles and their guidance systems.
Cruise missiles are much simpler than rockets: they fly until they reach their max_range_m.

See objs/cruise_missile.py for the implementation of cruise missiles and their guidance systems.
"""

from mad.configs.physics_cfg import MACHS_TO_METERS_S

# See https://en.wikipedia.org/wiki/Tomahawk_missile
tomahawk = {
    "name": "Tomahawk",
    "mass": 1300.0,  # kg
    "ref_radius": 0.25,  # m
    "Cd": 0.5,  # Approximate value for a streamlined missile
    "thrust_acc": 50.0,  # m/s² — moderate acceleration
    "max_range_m": 1_600 * 1000,  # 1600 km
    "yield_kt": 0.0,  # Conventional warhead
}

tomahawk_guidance = {
    "max_speed_m_s": 0.75 * MACHS_TO_METERS_S,  # 600 - 900 km/h
    "altitude_settling_time_s": 60.0,
    "cruise_altitude_m": 50.0,
}


# See https://en.wikipedia.org/wiki/Aster_(missile_family)
aster_30 = {
    "name": "ASTER-30",
    "mass": 450.0,  # kg
    "ref_radius": 0.09,  # m
    "Cd": 0.5,  # Approximate value for a streamlined missile
    "thrust_acc": 550.0,  # m/s² — higher acceleration
    "max_range_m": 120 * 1000,  # 120 km
    "yield_kt": 0.0,  # Conventional warhead
}

aster_30_guidance = {
    "max_speed_m_s": 4.5 * MACHS_TO_METERS_S,  # 1200 km/h
    "altitude_settling_time_s": 30.0,
    "cruise_altitude_m": 20_000.0,
    "kill_radius_m": 30.0,
}

# See https://en.wikipedia.org/wiki/V-1_flying_bomb
V1 = {
    "name": "V-1",
    "mass": 2150.0,  # kg
    "ref_radius": 0.3,  # m
    "Cd": 0.5,  # Approximate value for a streamlined missile
    "thrust_acc": 30.0,  # m/s² — moderate acceleration
    "max_range_m": 250 * 1000,  # 250 km
    "yield_kt": 0.0,  # Conventional warhead
}

V1_guidance = {
    "max_speed_m_s": 640 * 1000 / 3600,  # 640 km/h
    "altitude_settling_time_s": 60.0,
    "cruise_altitude_m": 900.0,
}
