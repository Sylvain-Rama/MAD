tomahawk = {
    "name": "Tomahawk",
    "mass": 1300.0,  # kg
    "ref_radius": 0.25,  # m
    "Cd": 0.5,  # Approximate value for a streamlined missile
    "thrust_acc": 30.0,  # m/s² — moderate acceleration
}

tomahawk_guidance = {
    "max_speed_m_s": 900 * 1000 / 3600,  # 600 - 900 km/h
    "max_range_m": 1_600 * 1000,  # 1600 km
    "cruise_altitude_m": 50.0,
}
