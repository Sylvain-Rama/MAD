from simulation import Simulation
from objs.missiles import Missile, MissileConfig
from objs.planets import Planet
from objs.common_schemas import Position, Velocity

sim = Simulation()
planet = Planet(radius=100, mu=5e4, atmosphere_height=20)
sim.add_planet(planet)

sim.add_target()

mis_cfg_1 = MissileConfig(target=sim.targets[0], correction_factor=2, cruise_altitude=5)
mis_cfg_2 = MissileConfig(target=sim.targets[0], cruise_altitude=20, thrust=60, burn_time=1.5, cruise_duration=2)

# missile_1 = Missile(
#     config=mis_cfg_1,
#     position=Position.from_angle_radius(angle=120, radius=planet.radius),
#     velocity=Velocity(0, 0),
# )

missile_2 = Missile(
    config=mis_cfg_2,
    position=Position.from_angle_radius(angle=120, radius=planet.radius),
    velocity=Velocity(0, 0),
)


# sim.add_missile(missile_1)
sim.add_missile(missile_2)


sim.run()
sim.plot()
