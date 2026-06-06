"""
Projectile configurations for MAD."""

rock = {
    "mass": 1.0,  # kg
    "ref_radius": 0.05,  # m
    "Cd": 0.47,  # Sphere
    "name": "Rock",
}

rock_no_drag = {
    "mass": 1.0,  # kg
    "ref_radius": 0.0,  # m
    "Cd": 0.0,  # No drag
    "name": "RockNoDrag",
}
