from mad.configs.physics_cfg import MACHS_TO_METERS_S

# See https://en.wikipedia.org/wiki/Tomahawk_missile
tomahawk = {
    "name": "Tomahawk",
    "mass": 1300.0,  # kg
    "ref_radius": 0.25,  # m
    "Cd": 0.5,  # Approximate value for a streamlined missile
    "thrust_acc": 100.0,  # m/s² — moderate acceleration
    "max_range_m": 1_600 * 1000,  # 1600 km
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
}

aster_30_guidance = {
    "max_speed_m_s": 4.5 * MACHS_TO_METERS_S,  # 1200 km/h
    "altitude_settling_time_s": 30.0,
    "cruise_altitude_m": 20_000.0,
}
