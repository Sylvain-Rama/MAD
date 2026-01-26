from simulation import Simulation
from objs.missiles import Missile, MissileConfig
from objs.planets import Planet
from objs.common_schemas import Position, Velocity

sim = Simulation()
planet = Planet(radius=100, mu=2.0e5, atmosphere_height=150)
sim.add_planet(planet)

sim.add_target()

mis_cfg = MissileConfig(target=sim.targets[0], correction_factor=2, cruise_altitude=110)

missile = Missile(
    config=mis_cfg,
    position=Position.from_angle_radius(angle=120, radius=planet.radius),
    velocity=Velocity(0, 0),
)

sim.add_missile(missile)


sim.run()
sim.plot()
