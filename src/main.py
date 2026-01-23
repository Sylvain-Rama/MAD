from simulation import Simulation

sim = Simulation()
sim.set_target(surface_angle_deg=120)


sim.add_missile(
    surface_angle_deg=0,
    thrust=140,
    burn_time=2.5,
    drag_coeff=0.08,
    target_altitude=200,
    cruise_thrust_ratio=0.6,
    cruise_duration=2.5,
    name="Balistique",
)

sim.add_missile(
    surface_angle_deg=0,
    thrust=100,
    burn_time=1.0,
    drag_coeff=0.3,
    target_altitude=150,
    cruise_thrust_ratio=0.5,
    cruise_duration=1.0,
    name="Standard",
)

sim.run()
sim.plot()
