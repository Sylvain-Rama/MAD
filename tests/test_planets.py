"""Tests for mad.objs.planets — Planet and PlanetConfig."""

import numpy as np
import pytest
from mad.objs.planets import Planet, PlanetConfig
from mad.objs.base import MovableObj
from mad.objs.projectiles import Projectile, ProjectileConfig
from mad.configs.planets import EARTH_SETTINGS
from mad.configs.physics import G

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def earth():
    cfg = dict(EARTH_SETTINGS)
    cfg["position"] = [0.0, 0.0, 0.0]
    return Planet(PlanetConfig(**cfg))


# ---------------------------------------------------------------------------
# PlanetConfig
# ---------------------------------------------------------------------------


class TestPlanetConfig:
    def test_required_fields(self):
        cfg = PlanetConfig(
            position=[0.0, 0.0, 0.0],
            radius=6_371_000.0,
            mass=5.972e24,
            spin_rate=7.0882359e-5,
        )
        assert cfg.name == "Planet"
        assert cfg.rho0 == pytest.approx(1.225)

    def test_to_dict(self):
        cfg = PlanetConfig(
            position=[0.0, 0.0, 0.0],
            radius=6_371_000.0,
            mass=5.972e24,
            spin_rate=7.0882359e-5,
        )
        d = cfg.to_dict
        assert isinstance(d, dict)
        assert "radius" in d


# ---------------------------------------------------------------------------
# Planet — construction
# ---------------------------------------------------------------------------


class TestPlanetInit:
    def test_mu(self, earth):
        expected_mu = EARTH_SETTINGS["mass"] * G
        assert earth.mu == pytest.approx(expected_mu, rel=1e-6)

    def test_radius(self, earth):
        assert earth.radius == pytest.approx(EARTH_SETTINGS["radius"])

    def test_name(self, earth):
        assert earth.name == "Earth"


# ---------------------------------------------------------------------------
# Derived orbital quantities
# ---------------------------------------------------------------------------


class TestOrbitalProperties:
    def test_escape_velocity(self, earth):
        expected = np.sqrt(2 * earth.mu / earth.radius)
        assert earth.escape_velocity == pytest.approx(expected, rel=1e-9)

    def test_orbital_velocity(self, earth):
        expected = np.sqrt(earth.mu / earth.radius)
        assert earth.orbital_velocity == pytest.approx(expected, rel=1e-9)

    def test_escape_gt_orbital(self, earth):
        assert earth.escape_velocity > earth.orbital_velocity

    def test_escape_is_sqrt2_times_orbital(self, earth):
        ratio = earth.escape_velocity / earth.orbital_velocity
        assert ratio == pytest.approx(np.sqrt(2), rel=1e-9)

    def test_gravity_at_surface_approx_9_81(self, earth):
        # Earth surface gravity ≈ 9.81 m/s²
        assert abs(earth.gravity_at_surface) == pytest.approx(9.81, abs=0.1)


# ---------------------------------------------------------------------------
# gravity()
# ---------------------------------------------------------------------------


class TestGravity:
    def test_gravity_points_inward(self, earth):
        """Gravity on an object above the surface must point toward the planet centre."""
        pos = [earth.radius + 100_000.0, 0.0, 0.0]
        obj = MovableObj(position=pos)
        g = earth.gravity(obj)
        # Should be negative in x (pointing toward origin) and zero in y, z
        assert g[0] < 0
        np.testing.assert_allclose(g[1:], 0.0, atol=1e-10)

    def test_gravity_inverse_square(self, earth):
        """Gravity magnitude scales as 1/r²."""
        r1 = earth.radius + 500_000.0
        r2 = earth.radius + 1_000_000.0
        obj1 = MovableObj(position=[r1, 0.0, 0.0])
        obj2 = MovableObj(position=[r2, 0.0, 0.0])
        g1 = np.linalg.norm(earth.gravity(obj1))
        g2 = np.linalg.norm(earth.gravity(obj2))
        assert g1 / g2 == pytest.approx((r2 / r1) ** 2, rel=1e-6)

    def test_gravity_at_zero_distance_returns_zeros(self, earth):
        obj = MovableObj(position=[0.0, 0.0, 0.0])
        g = earth.gravity(obj)
        np.testing.assert_array_equal(g, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# drag()
# ---------------------------------------------------------------------------


class TestDrag:
    def test_no_drag_above_atmosphere(self, earth):
        """An object far above the atmosphere should experience negligible drag
        (exponential atmosphere: rho ≈ 0 at very high altitude)."""
        alt = earth.atmosphere_height * 100  # 100× scale height → rho ≈ rho0*e^{-100}
        cfg = ProjectileConfig(
            position=[earth.radius + alt, 0.0],
            mass=1.0,
            ref_radius=0.5,
            Cd=1.0,
            velocity=[1000.0, 0.0],
        )
        obj = Projectile(cfg)
        drag = earth.drag(obj)
        assert np.linalg.norm(drag) < 1e-30

    def test_drag_opposes_velocity(self, earth):
        """Drag must be anti-parallel to velocity."""
        cfg = ProjectileConfig(
            position=[earth.radius + 1000.0, 0.0],
            mass=10.0,
            ref_radius=0.5,
            Cd=0.5,
            velocity=[500.0, 0.0],
        )
        obj = Projectile(cfg)
        drag = earth.drag(obj)
        # drag[0] should be negative (opposing positive vx)
        assert drag[0] < 0
        assert drag[1] == pytest.approx(0.0, abs=1e-10)

    def test_drag_zero_for_stationary_object(self, earth):
        cfg = ProjectileConfig(
            position=[earth.radius + 1000.0, 0.0],
            mass=1.0,
            ref_radius=0.5,
            Cd=1.0,
            velocity=[0.0, 0.0],
        )
        obj = Projectile(cfg)
        drag = earth.drag(obj)
        np.testing.assert_allclose(drag, 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# surface_distance()
# ---------------------------------------------------------------------------


class TestSurfaceDistance:
    def test_distance_to_self_is_zero(self, earth):
        p = MovableObj(position=[earth.radius, 0.0, 0.0])
        assert earth.surface_distance(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_distance_known_angle(self, earth):
        """Two points separated by 90° should be π/2 × R apart."""
        p1 = MovableObj(position=[earth.radius, 0.0])
        p2 = MovableObj(position=[0.0, earth.radius])
        expected = (np.pi / 2) * earth.radius
        assert earth.surface_distance(p1, p2) == pytest.approx(expected, rel=1e-6)

    def test_distance_symmetric(self, earth):
        p1 = earth.random_point_at_surface()
        p2 = earth.point_at_distance(p1, 1000.0)
        assert earth.surface_distance(p1, p2) == pytest.approx(earth.surface_distance(p2, p1), rel=1e-9)


# ---------------------------------------------------------------------------
# point_at_distance()
# ---------------------------------------------------------------------------


class TestPointCreation:
    def test_2d_point_at_surface(self, earth):
        p = earth.random_point_at_surface(altitude=0)
        r = np.linalg.norm(p.position)
        assert r == pytest.approx(earth.radius, rel=1e-9)

    def test_2d_point_at_altitude(self, earth):
        alt = 500.0
        p = earth.random_point_at_surface(altitude=alt)
        r = np.linalg.norm(p.position)
        assert r == pytest.approx(earth.radius + alt, rel=1e-9)

    def test_2d_point_at_distance_km(self, earth):
        p1 = earth.random_point_at_surface(altitude=0)
        distance_km = 1000.0
        p2 = earth.point_at_distance(p1, distance_km=distance_km)
        actual_km = earth.surface_distance(p1, p2) / 1000.0
        assert actual_km == pytest.approx(distance_km, rel=1e-6)
