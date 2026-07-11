"""Configs for Rocket objects
Rockets are defined by a list of stages, each with its own mass, thrust, and specific impulse (Isp).
"""

# See https://en.wikipedia.org/wiki/HGM-25A_Titan_I#Specifications
titan1_stages = [
    {
        "dry_mass": 4000.0,
        "full_mass": 76203.0,
        "thrust": 1900 * 1000,  # N
        "Isp": 290.0,
        "ref_radius": 1.5,  # m
        "Cd": 1.08,  # smooth, long cylinder.
        "name": "Stage1",
    },
    {
        "dry_mass": 1725.0,
        "full_mass": 28939.0,
        "thrust": 356 * 1000,  # N
        "Isp": 308.0,
        "ref_radius": 1.1,  # m
        "Cd": 1.08,  # smooth, long cylinder.
        "name": "Stage2",
    },
]

# See https://en.wikipedia.org/wiki/LGM-25C_Titan_II#Missile_characteristics
titan2_stages = [
    {
        "dry_mass": 4319.0,
        "full_mass": 121200.0,
        "thrust": 1900 * 1000,  # N
        "Isp": 258.0,
        "ref_radius": 1.5,  # m
        "Cd": 1.08,  # smooth, long cylinder.
        "name": "Stage1",
    },
    {
        "dry_mass": 2301.0,
        "full_mass": 28400.0,
        "thrust": 445 * 1000,  # N
        "Isp": 316.0,
        "ref_radius": 1.5,  # m
        "Cd": 1.08,  # smooth, long cylinder.
        "name": "Stage2",
    },
]


# Multiple research sources for minuteman stages
# See https://minutemanmissile.com/solidrocketboosters.html
minuteman_stages = [
    {
        "full_mass": 23247.0,
        "propellant_mass": 20715.0,
        "thrust": 930 * 1000,  # N
        "Isp": 270.0,
        "ref_radius": 0.75,  # m
        "Cd": 0.5,  # smooth, long cylinder.
        "name": "Stage1",
    },
    {
        "dry_mass": 4708.0,
        "full_mass": 5476.0,
        "thrust": 267 * 1000,  # N
        "Isp": 290.0,
        "ref_radius": 0.46,  # m
        "Cd": 0.5,  # smooth, long cylinder.
        "name": "Stage2",
    },
    {
        "dry_mass": 1664.0,
        "full_mass": 2034.0,
        "thrust": 152 * 1000,  # N
        "Isp": 300.0,
        "ref_radius": 0.46,  # m
        "Cd": 0.5,  # smooth, long cylinder.
        "name": "Stage3",
    },
]

# For Sputnik rocket: https://en.wikipedia.org/wiki/Sputnik_(rocket)
# Bloc_BVGD (4 strap-on boosters) and Bloc_A (core) fire simultaneously from T=0;
# Bloc_BVGD separates first (~71 s), Bloc_A continues to orbital speed.
sputnik_stages = [
    {
        "dry_mass": 3400.0,
        "full_mass": 43_000.0,
        "thrust": 4 * 520 * 1000,  # N
        "Isp": 310.0,
        "ref_radius": 3.6,  # m
        "Cd": 0.5,  # Pointy end
        "name": "Bloc_BVGD",
        "parallel": False,  # First stage — ignites at T=0 by default.
    },
    {
        "dry_mass": 7500.0,
        "full_mass": 94_000.0,
        "thrust": 970 * 1000,  # N
        "Isp": 380.0,
        "ref_radius": 1.5,  # m
        "Cd": 0.5,  # Pointy end
        "name": "Bloc_A",
        "parallel": True,  # Core stage fires in parallel with Bloc_BVGD from T=0.
        "separation_retrograde_dv": 75.0,  # m/s retrograde kick at separation to ensure reentry.
    },
]
