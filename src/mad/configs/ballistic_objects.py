"""Configs for missile stages"""

# See https://en.wikipedia.org/wiki/HGM-25A_Titan_I#Specifications
titan1_stages = [
    {
        "dry_mass": 4000.0,
        "propellant_mass": 76203.0 - 4000.0,
        "thrust": 1900 * 1000,  # N
        "Isp": 290.0,
        "ref_radius": 1.5,  # m
        "Cd": 1.08,  # smooth, long cylinder.
        "name": "Stage1",
    },
    {
        "dry_mass": 1725.0,
        "propellant_mass": 28939 - 1725.0,
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
        "propellant_mass": 121200.0 - 4319.0,
        "thrust": 1900 * 1000,  # N
        "Isp": 258.0,
        "ref_radius": 1.5,  # m
        "Cd": 1.08,  # smooth, long cylinder.
        "name": "Stage1",
    },
    {
        "dry_mass": 2301.0,
        "propellant_mass": 28400 - 2301.0,
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
        "dry_mass": 23247.0 - 20715.0,
        "propellant_mass": 20715.0,
        "thrust": 930 * 1000,  # N
        "Isp": 270.0,
        "ref_radius": 0.75,  # m
        "Cd": 1.08,  # smooth, long cylinder.
        "name": "Stage1",
    },
    {
        "dry_mass": 4708.0,
        "propellant_mass": 5476.0 - 4708.0,
        "thrust": 267 * 1000,  # N
        "Isp": 290.0,
        "ref_radius": 0.46,  # m
        "Cd": 1.08,  # smooth, long cylinder.
        "name": "Stage2",
    },
    {
        "dry_mass": 1664.0,
        "propellant_mass": 2034 - 1664.0,
        "thrust": 152 * 1000,  # N
        "Isp": 300.0,
        "ref_radius": 0.46,  # m
        "Cd": 1.08,  # smooth, long cylinder.
        "name": "Stage3",
    },
]
