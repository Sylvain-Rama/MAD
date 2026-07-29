# See https://en.wikipedia.org/wiki/Mercury-Redstone_Launch_Vehicle
# And https://en.wikipedia.org/wiki/Mercury-Redstone_1A
# And https://en.wikipedia.org/wiki/PGM-11_Redstone
# And https://en.wikipedia.org/wiki/Project_Mercury#Spacecraft
mercury_redstone = [
    {
        "dry_mass": 3125.0,
        # "propellant_mass": 5051.0 + 11470.0 + 360.0,  # kg Alcohol, LOX, and Hydrogen Peroxide
        "full_mass": 28_440.0,
        "thrust": 350 * 1000,  # N
        "Isp": 215.0,
        "ref_radius": 0.9,  # m
        "Cd": 0.5,
        "name": "Mercury Redstone",
    },
]

mercury_capsule = {
    "mass": 1_200.0,  # kg
    "ref_radius": 0.9,  # m
    "Cd": 0.5,  # Pointy end
    "name": "Mercury Capsule",
}

# See https://en.wikipedia.org/wiki/Atlas_LV-3B
# https://en.wikipedia.org/wiki/Mercury-Atlas_6
# This needed adaptation as the rocket has boosters, dropped after 2 minutes, and the main stage has a different thrust and Isp than the boosters.
# So the rocket loses 3050 kg of boosters but the propellant tanks are in the main stage.
# This could not be simulated with the current code, so I use a beefy main stage for the whole flight.
# TODO: Implement the possibility to lose weight or other parameters during flight, to simulate the booster drop.
mercury_atlas = [
    {
        "dry_mass": 7300.0,
        "full_mass": 115_000.0,
        "thrust": 1517.4 * 1000,  # N This is cumulated thrust of boosters & main stage.
        "Isp": 292.0,  # s Vacuum. 252 for sea level.
        "ref_radius": 1.5,  # m
        "Cd": 0.5,
        "name": "Mercury Atlas",
    }
]
