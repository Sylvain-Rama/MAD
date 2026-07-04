"""Tests for mad.objs.missiles — MissileStageConfig, MissileStage, RVConfig,
ReentryVehicle, BallisticMissileConfig, and BallisticMissile."""

import numpy as np
import pytest
from mad.objs.rockets import (
    RocketStageConfig,
    RocketStage,
    RVConfig,
    ReentryVehicle,
    RocketConfig,
    Rocket,
)
from mad.objs.planets import Planet, PlanetConfig
from mad.configs.planets_cfg import EARTH_SETTINGS
from mad.configs.ballistic_objects_cfg import titan1_stages
from mad.configs.warheads_cfg import B53_warhead
from mad.configs.physics_cfg import G0
from mad.guidances import NoGuidance, IdleGuidance

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def earth():
    cfg = dict(EARTH_SETTINGS)
    cfg["position"] = [0.0, 0.0]
    return Planet(PlanetConfig(**cfg))


@pytest.fixture
def stage_cfg():
    return RocketStageConfig(
        thrust=1_900_000.0,
        ref_radius=1.5,
        dry_mass=4_000.0,
        propellant_mass=72_203.0,
        Isp=290.0,
        name="Stage1",
    )


@pytest.fixture
def stage(stage_cfg):
    return RocketStage(stage_cfg)


@pytest.fixture
def RV_cfg(earth):
    return RVConfig(**B53_warhead, guidance=IdleGuidance(earth, earth))


@pytest.fixture
def two_stage_missile(earth):
    stages = [RocketStage(RocketStageConfig(**s)) for s in titan1_stages]
    cfg = RocketConfig(stages=stages, guidance=NoGuidance(None, None))
    r = earth.radius + 10.0
    return Rocket(position=[r, 0.0], cfg=cfg, name="Titan")


# ---------------------------------------------------------------------------
# RocketStageConfig
# ---------------------------------------------------------------------------


class TestRocketStageConfig:
    def test_full_mass_computed_from_dry_and_propellant(self):
        cfg = RocketStageConfig(thrust=1_000.0, ref_radius=0.5, dry_mass=100.0, propellant_mass=900.0, Isp=300.0)
        assert cfg.full_mass == pytest.approx(1_000.0)

    def test_dry_mass_computed_from_full_and_propellant(self):
        cfg = RocketStageConfig(thrust=1_000.0, ref_radius=0.5, full_mass=1_000.0, propellant_mass=900.0, Isp=300.0)
        assert cfg.dry_mass == pytest.approx(100.0)

    def test_propellant_mass_computed_from_full_and_dry(self):
        cfg = RocketStageConfig(thrust=1_000.0, ref_radius=0.5, full_mass=1_000.0, dry_mass=100.0, Isp=300.0)
        assert cfg.propellant_mass == pytest.approx(900.0)

    def test_inconsistent_masses_raises(self):
        with pytest.raises(ValueError, match="Inconsistent"):
            RocketStageConfig(
                thrust=1_000.0,
                ref_radius=0.5,
                dry_mass=100.0,
                propellant_mass=900.0,
                full_mass=500.0,  # wrong: should be 1000
                Isp=300.0,
            )

    def test_isp_computed_from_burn_time(self):
        thrust = 1_000.0
        burn_time = 100.0
        propellant = 50.0
        cfg = RocketStageConfig(
            thrust=thrust,
            ref_radius=0.5,
            dry_mass=10.0,
            propellant_mass=propellant,
            burn_time=burn_time,
        )
        expected_isp = (thrust * burn_time) / (propellant * G0)
        assert cfg.Isp == pytest.approx(expected_isp, rel=1e-6)

    def test_area_computed(self):
        cfg = RocketStageConfig(thrust=1_000.0, ref_radius=1.0, dry_mass=100.0, propellant_mass=900.0, Isp=300.0)
        assert cfg.area == pytest.approx(np.pi * 1.0**2, rel=1e-9)

    def test_missing_isp_and_burn_time_raises(self):
        with pytest.raises((ValueError, TypeError)):
            RocketStage(
                RocketStageConfig(
                    thrust=1_000.0,
                    ref_radius=0.5,
                    dry_mass=100.0,
                    propellant_mass=900.0,
                    # no Isp, no burn_time
                )
            )


# ---------------------------------------------------------------------------
# RocketStage
# ---------------------------------------------------------------------------


class TestRocketStage:
    def test_initial_mass(self, stage, stage_cfg):
        expected = stage_cfg.dry_mass + stage_cfg.propellant_mass
        assert stage.mass == pytest.approx(expected)

    def test_thrust_force_with_propellant(self, stage):
        assert stage.thrust_force == pytest.approx(stage.thrust)

    def test_thrust_force_without_propellant(self, stage):
        stage.propellant_mass = 0.0
        assert stage.thrust_force == pytest.approx(0.0)

    def test_update_reduces_propellant(self, stage):
        initial_prop = stage.propellant_mass
        stage.update(1.0)
        assert stage.propellant_mass < initial_prop

    def test_update_does_not_go_negative(self, stage):
        # Burn for much longer than available propellant
        stage.update(1_000_000.0)
        assert stage.propellant_mass >= 0.0

    def test_becomes_inactive_when_propellant_depleted(self, stage):
        stage.propellant_mass = 0.001  # near zero
        stage.update(10.0)  # burns the last propellant; propellant reaches 0
        stage.update(0.0)  # next call detects propellant==0 and sets active=False
        assert stage.active is False

    def test_update_when_inactive_is_noop(self, stage):
        stage.active = False
        old_prop = stage.propellant_mass
        stage.update(1.0)
        assert stage.propellant_mass == old_prop

    def test_exhaust_velocity(self, stage, stage_cfg):
        assert stage.exhaust_velocity == pytest.approx(stage_cfg.Isp * G0, rel=1e-9)


# ---------------------------------------------------------------------------
# RVConfig
# ---------------------------------------------------------------------------


class TestRVConfig:
    def test_area_computed(self, RV_cfg):
        assert RV_cfg.area == pytest.approx(np.pi * B53_warhead["ref_radius"] ** 2, rel=1e-9)

    def test_defaults(self, RV_cfg):
        assert isinstance(RV_cfg.guidance, IdleGuidance)
        assert RV_cfg.RCS_thrust == pytest.approx(500.0)

    def test_yield(self, RV_cfg):
        assert RV_cfg.yield_kt == pytest.approx(9_000.0)

    def test_alias(self, RV_cfg):
        assert isinstance(RV_cfg, RVConfig)


# ---------------------------------------------------------------------------
# ReentryVehicle
# ---------------------------------------------------------------------------


class TestReentryVehicle:
    def test_thrust_acc(self, RV_cfg, earth):
        r = earth.radius + 500_000.0
        p = ReentryVehicle(RV_cfg, position=[r, 0.0])
        expected = RV_cfg.RCS_thrust / RV_cfg.mass
        assert p.thrust_acc == pytest.approx(expected)

    def test_burned_fraction_is_half(self, RV_cfg, earth):
        r = earth.radius + 500_000.0
        p = ReentryVehicle(RV_cfg, position=[r, 0.0])
        assert p.burned_fraction == pytest.approx(0.5)

    def test_update_advances_time(self, RV_cfg, earth):
        r = earth.radius + 500_000.0
        p = ReentryVehicle(RV_cfg, position=[r, 0.0], t=0.0)
        p.update(5.0)
        assert p.t == pytest.approx(5.0)

    def test_update_returns_none(self, RV_cfg, earth):
        r = earth.radius + 500_000.0
        p = ReentryVehicle(RV_cfg, position=[r, 0.0])
        assert p.update(1.0) is None

    def test_accelerations_without_guidance(self, RV_cfg, earth):
        """Without guidance the acceleration should be gravity + drag only."""
        r = earth.radius + 500_000.0
        p = ReentryVehicle(RV_cfg, position=[r, 0.0], velocity=[0.0, 0.0])
        acc = p.accelerations(earth)
        gravity = earth.gravity(p)
        drag = earth.drag(p)
        np.testing.assert_allclose(acc, gravity + drag, rtol=1e-9)

    def test_goes_inactive_below_surface(self, RV_cfg, earth):
        below = earth.radius - 1.0
        p = ReentryVehicle(RV_cfg, position=[below, 0.0])
        acc = p.accelerations(earth)
        assert p.active is False
        np.testing.assert_array_equal(acc, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Rocket — properties
# ---------------------------------------------------------------------------


class TestBallisticMissileProperties:
    def test_mass_is_sum_of_stages(self, two_stage_missile):
        expected = sum(s.mass for s in two_stage_missile.stages)
        assert two_stage_missile.mass == pytest.approx(expected)

    def test_area_is_last_stage_area(self, two_stage_missile):
        assert two_stage_missile.area == pytest.approx(two_stage_missile.stages[-1].area)

    def test_burned_fraction_zero_at_start(self, two_stage_missile):
        # At launch nothing has burned yet
        assert two_stage_missile.burned_fraction == pytest.approx(0.0, abs=1e-6)

    def test_deltav_positive(self, two_stage_missile):
        assert two_stage_missile.deltav > 0

    def test_thrust_acc_positive_with_propellant(self, two_stage_missile):
        assert two_stage_missile.thrust_acc > 0

    def test_thrust_acc_zero_when_stage_inactive(self, two_stage_missile):
        two_stage_missile.stages[0].active = False
        assert two_stage_missile.thrust_acc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BallisticMissile — update (stage separation)
# ---------------------------------------------------------------------------


class TestBallisticMissileUpdate:
    def test_update_advances_time(self, two_stage_missile):
        two_stage_missile.update(1.0)
        assert two_stage_missile.t == pytest.approx(1.0)

    def test_stage_separation_returns_projectile(self, earth):
        """When the first stage runs out, update should return the spent stage as a Projectile."""
        stages = [RocketStage(RocketStageConfig(**s)) for s in titan1_stages]
        # Deplete the first stage immediately
        stages[0].propellant_mass = 0.001
        cfg = RocketConfig(stages=stages, guidance=NoGuidance(None, None))
        r = earth.radius + 10.0
        missile = Rocket(position=[r, 0.0], cfg=cfg, velocity=[0.0, 100.0, 0.0], name="T")

        missile.update(1.0)  # burns last propellant; stage still active
        result = missile.update(0.0)  # stage now detects empty and separates
        assert result is not None
        assert len(result) >= 1

    def test_missile_inactive_after_all_stages_spent(self, earth):
        """A single-stage missile becomes inactive when its stage is exhausted."""
        single_stage_cfg = RocketStageConfig(
            thrust=500_000.0,
            ref_radius=1.0,
            dry_mass=500.0,
            propellant_mass=0.001,  # virtually empty
            Isp=300.0,
            name="OnlyStage",
        )
        stages = [RocketStage(single_stage_cfg)]
        cfg = RocketConfig(stages=stages, guidance=NoGuidance(None, None))
        r = earth.radius + 10.0
        missile = Rocket(position=[r, 0.0], cfg=cfg, velocity=[0.0, 100.0, 0.0], name="Single")

        missile.update(1.0)  # burns propellant; stage still active
        missile.update(0.0)  # stage detects empty → separates → missile inactive
        assert missile.active is False


# ---------------------------------------------------------------------------
# Rocket — accelerations
# ---------------------------------------------------------------------------


class TestRocketAccelerations:
    def test_accelerations_returns_array(self, two_stage_missile, earth):
        two_stage_missile.velocity = np.array([0.0, 100.0, 0.0])
        acc = two_stage_missile.accelerations(earth)
        assert acc.shape == two_stage_missile.position.shape
        assert acc.shape == two_stage_missile.position.shape

    def test_thrust_increases_acceleration(self, earth):
        """Missile with thrust should have larger acceleration magnitude than one without."""
        stages_thrust = [RocketStage(RocketStageConfig(**s)) for s in titan1_stages]
        stages_no_thrust = [RocketStage(RocketStageConfig(**s)) for s in titan1_stages]
        for s in stages_no_thrust:
            s.propellant_mass = 0.0  # no fuel → no thrust

        r = earth.radius + 500_000.0
        vel = np.array([0.0, 1000.0, 0.0])

        m_thrust = Rocket(
            position=[r, 0.0],
            cfg=RocketConfig(stages=stages_thrust, guidance=NoGuidance(None, None)),
            name="A",
            velocity=vel.copy(),
        )
        m_coast = Rocket(
            position=[r, 0.0],
            cfg=RocketConfig(stages=stages_no_thrust, guidance=NoGuidance(None, None)),
            name="B",
            velocity=vel.copy(),
        )

        acc_thrust = np.linalg.norm(m_thrust.accelerations(earth))
        acc_coast = np.linalg.norm(m_coast.accelerations(earth))
        assert acc_thrust > acc_coast


# ---------------------------------------------------------------------------
# Rocket — ballistic_range
# ---------------------------------------------------------------------------


class TestRocketRange:
    def test_ballistic_range_positive(self, two_stage_missile, earth):
        r = two_stage_missile.ballistic_range(earth)
        assert r > 0

    def test_ballistic_range_decreases_with_steeper_angle(self, two_stage_missile, earth):
        r_45 = two_stage_missile.ballistic_range(earth, gamma_rad=np.radians(45))
        r_80 = two_stage_missile.ballistic_range(earth, gamma_rad=np.radians(80))
        # 45° should give longer range than a very steep trajectory
        assert r_45 > r_80
