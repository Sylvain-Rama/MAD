from simulation import Simulation
from objs.missiles import Missile, MissileConfig
from objs.planets import Planet

sim = Simulation()

sim.add_planet(Planet(20, 0.02, 5))

sim.add_target()

mis_cfg = MissileConfig(target=sim.targets[0])
missile = Missile(config=mis_cfg)

sim.add_missile(missile)


sim.run()
sim.plot()
