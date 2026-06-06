"""Tests for mad.objs.base — MovableObj, BallisticObj, SimulationInterface."""

import numpy as np
import pytest
from mad.objs.base import MovableObj
from mad.objs.planets import Planet, PlanetConfig
from mad.objs.projectiles import Projectile, ProjectileConfig
from mad.configs.planets_cfg import EARTH_SETTINGS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_earth():
    cfg = dict(EARTH_SETTINGS)
    cfg["position"] = [0.0, 0.0]
    return Planet(PlanetConfig(**cfg))


# ---------------------------------------------------------------------------
# MovableObj
# ---------------------------------------------------------------------------


class TestMovableObj:
    def test_init_list(self):
        obj = MovableObj(position=[1.0, 2.0, 3.0])
        np.testing.assert_array_equal(obj.position, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(obj.velocity, [0.0, 0.0, 0.0])

    def test_init_2d_padded_to_3d(self):
        """to_vec3 always produces a 3-element vector."""
        obj = MovableObj(position=[1.0, 0.0])
        assert obj.position.shape == (3,)
        np.testing.assert_array_equal(obj.position, [1.0, 0.0, 0.0])

    def test_init_velocity(self):
        obj = MovableObj(position=[0.0, 0.0, 0.0], velocity=[1.0, 2.0, 0.0])
        np.testing.assert_array_equal(obj.velocity, [1.0, 2.0, 0.0])

    def test_active_default_true(self):
        obj = MovableObj(position=[0.0, 0.0])
        assert obj.active is True

    def test_name(self):
        obj = MovableObj(position=[0.0, 0.0], name="TestObj")
        assert obj.name == "TestObj"

    def test_unique_ids(self):
        a = MovableObj(position=[0.0, 0.0])
        b = MovableObj(position=[1.0, 0.0])
        assert a._id != b._id

    def test_normalize_unit_vector(self):
        obj = MovableObj(position=[3.0, 4.0])
        norm = obj.normalize
        assert pytest.approx(np.linalg.norm(norm), abs=1e-9) == 1.0

    def test_normalize_zero_returns_zeros(self):
        obj = MovableObj(position=[0.0, 0.0, 0.0])
        np.testing.assert_array_equal(obj.normalize, [0.0, 0.0, 0.0])

    def test_distance_symmetric(self):
        a = MovableObj(position=[0.0, 0.0])
        b = MovableObj(position=[3.0, 4.0])
        assert pytest.approx(a.distance(b)) == b.distance(a)

    def test_distance_known_value(self):
        a = MovableObj(position=[0.0, 0.0])
        b = MovableObj(position=[3.0, 4.0])
        assert pytest.approx(a.distance(b)) == 5.0

    def test_equality_same_object(self):
        obj = MovableObj(position=[0.0, 0.0])
        assert obj == obj

    def test_equality_different_objects(self):
        a = MovableObj(position=[0.0, 0.0])
        b = MovableObj(position=[0.0, 0.0])
        assert a != b

    def test_equality_non_movable(self):
        obj = MovableObj(position=[0.0, 0.0])
        assert obj != "not_a_movable"


# ---------------------------------------------------------------------------
# BallisticObj
# ---------------------------------------------------------------------------


class TestBallisticObj:
    """BallisticObj is abstract; test its interface via Projectile."""

    def _make(self, **kwargs):
        cfg = ProjectileConfig(position=[6_371_000.0 + 1_000_000.0, 0.0, 0.0], mass=kwargs.get("mass", 1.0))
        if "Cd" in kwargs:
            cfg.Cd = kwargs["Cd"]
        return Projectile(cfg)

    def test_defaults(self):
        obj = self._make()
        assert obj.mass == pytest.approx(1.0)
        assert obj.Cd == pytest.approx(0.47)

    def test_custom_mass(self):
        obj = self._make(mass=50.0)
        assert obj.mass == pytest.approx(50.0)

    def test_inherits_movableobj(self):
        cfg = ProjectileConfig(position=[1.0, 2.0, 3.0], mass=1.0)
        obj = Projectile(cfg)
        np.testing.assert_array_equal(obj.position, [1.0, 2.0, 3.0])

    def test_mass_is_property(self):
        obj = self._make(mass=10.0)
        with pytest.raises(AttributeError):
            obj.mass = 99.0


# ---------------------------------------------------------------------------
# SimulationInterface.integrate (Velocity Verlet)
# ---------------------------------------------------------------------------


class TestVelocityVerletIntegration:
    """The shared integrator lives on SimulationInterface and is inherited
    by BallisticObj subclasses.  We test it via Projectile which provides
    a concrete accelerations() implementation."""

    def test_free_fall_acceleration(self):
        """A stationary object under constant downward acceleration should
        satisfy x = 0.5 * a * t^2 and v = a * t after one step."""
        earth = _make_earth()

        # Place projectile high above so it doesn't land
        r = earth.radius + 1_000_000.0
        proj_cfg = ProjectileConfig(
            position=[r, 0.0, 0.0],
            mass=1.0,
            ref_radius=0.0,  # no drag
            Cd=0.0,
            velocity=[0.0, 0.0, 0.0],
        )
        proj = Projectile(proj_cfg)

        a0 = proj.accelerations(earth)
        dt = 1.0
        proj.integrate(dt, earth)

        # After one step, position should have moved along a0 direction
        # x ≈ 0.5 * a * dt^2 (since initial v=0)
        expected_pos = np.array([r, 0.0, 0.0]) + 0.5 * a0 * dt**2
        np.testing.assert_allclose(proj.position, expected_pos, rtol=1e-5)

    def test_orbit_energy_conservation(self):
        """A drag-free projectile in circular orbit should conserve
        mechanical energy over many steps."""
        earth = _make_earth()

        r = earth.radius + 500_000.0  # 500 km orbit
        v_circ = earth.orbital_velocity * np.sqrt(earth.radius / r)

        proj_cfg = ProjectileConfig(
            position=[r, 0.0],
            mass=1.0,
            ref_radius=0.0,
            Cd=0.0,
            velocity=[0.0, v_circ],
        )
        proj = Projectile(proj_cfg)

        def energy(p):
            r_ = np.linalg.norm(p.position)
            v_ = np.linalg.norm(p.velocity)
            return 0.5 * v_**2 - earth.mu / r_

        e0 = energy(proj)
        dt = 1.0
        for _ in range(500):
            proj.integrate(dt, earth)

        ef = energy(proj)
        assert abs(ef - e0) / abs(e0) < 1e-4  # <0.01% drift over 500 steps
