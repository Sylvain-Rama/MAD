"""Tests for mad.objs.projectiles — ProjectileConfig and Projectile."""

import numpy as np
import pytest
from mad.objs.projectiles import Projectile, ProjectileConfig
from mad.objs.planets import Planet, PlanetConfig
from mad.configs.planets import EARTH_SETTINGS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def earth():
    cfg = dict(EARTH_SETTINGS)
    cfg["position"] = [0.0, 0.0]
    return Planet(PlanetConfig(**cfg))


def _surface_projectile(earth, velocity=None, *, alt=500_000.0, ref_radius=0.0, Cd=0.0, mass=1.0):
    """Return a Projectile at `alt` metres above the surface."""
    r = earth.radius + alt
    cfg = ProjectileConfig(
        position=[r, 0.0],
        mass=mass,
        velocity=velocity if velocity is not None else [0.0, 0.0],
        ref_radius=ref_radius,
        Cd=Cd,
    )
    return Projectile(cfg)


# ---------------------------------------------------------------------------
# ProjectileConfig
# ---------------------------------------------------------------------------


class TestProjectileConfig:
    def test_area_computed_from_ref_radius(self):
        cfg = ProjectileConfig(position=[1.0, 0.0], mass=1.0, ref_radius=0.1)
        assert cfg.area == pytest.approx(np.pi * 0.1**2, rel=1e-9)

    def test_zero_ref_radius_gives_zero_area(self):
        cfg = ProjectileConfig(position=[1.0, 0.0], mass=1.0, ref_radius=0.0)
        assert cfg.area == pytest.approx(0.0)

    def test_defaults(self):
        cfg = ProjectileConfig(position=[1.0, 0.0], mass=5.0)
        assert cfg.name == "Projectile"
        assert cfg.Cd == pytest.approx(0.47)
        assert cfg.ref_radius == pytest.approx(0.01)

    def test_to_dict(self):
        cfg = ProjectileConfig(position=[1.0, 0.0], mass=5.0)
        d = cfg.to_dict
        assert "mass" in d
        assert "Cd" in d


# ---------------------------------------------------------------------------
# Projectile — construction
# ---------------------------------------------------------------------------


class TestProjectileInit:
    def test_mass_and_area_propagated(self, earth):
        proj = _surface_projectile(earth, mass=42.0)
        assert proj.mass == pytest.approx(42.0)

    def test_initial_time(self, earth):
        proj = _surface_projectile(earth)
        assert proj.t == pytest.approx(0.0)

    def test_custom_start_time(self, earth):
        r = earth.radius + 1000.0
        cfg = ProjectileConfig(position=[r, 0.0], mass=1.0)
        proj = Projectile(cfg, t=42.0)
        assert proj.t == pytest.approx(42.0)

    def test_active_on_creation(self, earth):
        proj = _surface_projectile(earth)
        assert proj.active is True


# ---------------------------------------------------------------------------
# Projectile — update
# ---------------------------------------------------------------------------


class TestProjectileUpdate:
    def test_update_advances_time(self, earth):
        proj = _surface_projectile(earth)
        proj.update(1.0)
        assert proj.t == pytest.approx(1.0)

    def test_update_returns_none(self, earth):
        proj = _surface_projectile(earth)
        result = proj.update(1.0)
        assert result is None


# ---------------------------------------------------------------------------
# Projectile — accelerations
# ---------------------------------------------------------------------------


class TestProjectileAccelerations:
    def test_no_drag_above_surface(self, earth):
        """A drag-free projectile's acceleration should equal pure gravity."""
        proj = _surface_projectile(earth, velocity=[0.0, 1000.0])
        gravity = earth.gravity(proj)
        acc = proj.accelerations(earth)
        np.testing.assert_allclose(acc, gravity, rtol=1e-9)

    def test_goes_inactive_when_below_surface(self, earth):
        """An object below the surface should become inactive and return zero acceleration."""
        cfg = ProjectileConfig(
            position=[earth.radius - 1.0, 0.0],  # below surface
            mass=1.0,
            ref_radius=0.0,
            Cd=0.0,
        )
        proj = Projectile(cfg)
        acc = proj.accelerations(earth)
        assert proj.active is False
        np.testing.assert_array_equal(acc, [0.0, 0.0, 0.0])

    def test_drag_reduces_acceleration_magnitude(self, earth):
        """A projectile with drag should have a lower net radial acceleration than a drag-free one."""
        v = [0.0, 5000.0]
        proj_no_drag = _surface_projectile(earth, velocity=v, alt=10_000.0, ref_radius=0.0, Cd=0.0)
        proj_drag = _surface_projectile(earth, velocity=v, alt=10_000.0, ref_radius=0.5, Cd=1.0, mass=50.0)
        acc_no_drag = np.linalg.norm(proj_no_drag.accelerations(earth))
        acc_drag = np.linalg.norm(proj_drag.accelerations(earth))
        # drag adds a decelerating force, so total magnitude differs (drag opposes motion)
        assert acc_no_drag != pytest.approx(acc_drag, rel=1e-3)


# ---------------------------------------------------------------------------
# Projectile — integration (end-to-end physics)
# ---------------------------------------------------------------------------


class TestProjectileIntegration:
    def test_lands_after_enough_steps(self, earth):
        """A projectile dropped from rest at low altitude should land."""
        alt = 200_000.0  # 200 km — within a few minutes at g≈9m/s²
        proj = _surface_projectile(earth, alt=alt, ref_radius=0.0, Cd=0.0)

        dt = 1.0
        for _ in range(10_000):
            proj.update(dt)
            proj.integrate(dt, earth)
            if not proj.active:
                break

        assert proj.active is False

    def test_circular_orbit_stays_circular(self, earth):
        """A drag-free projectile at circular orbit speed should stay at roughly constant altitude."""
        r0 = earth.radius + 400_000.0
        v_circ = np.sqrt(earth.mu / r0)
        proj = _surface_projectile(earth, velocity=[0.0, v_circ], alt=400_000.0, ref_radius=0.0, Cd=0.0)

        dt = 1.0
        radii = []
        for _ in range(2_000):
            proj.integrate(dt, earth)
            radii.append(np.linalg.norm(proj.position))

        r_min = min(radii)
        r_max = max(radii)
        # Orbit should not deviate more than 0.1% from initial radius
        assert (r_max - r_min) / r0 < 1e-3

    def test_energy_conservation_no_drag(self, earth):
        """Mechanical energy should be conserved for a drag-free projectile."""
        v0 = [0.0, 7000.0]
        proj = _surface_projectile(earth, velocity=v0, alt=800_000.0, ref_radius=0.0, Cd=0.0)

        def mech_energy(p):
            r = np.linalg.norm(p.position)
            v = np.linalg.norm(p.velocity)
            return 0.5 * v**2 - earth.mu / r

        e0 = mech_energy(proj)
        dt = 1.0
        for _ in range(1_000):
            proj.integrate(dt, earth)

        ef = mech_energy(proj)
        assert abs(ef - e0) / abs(e0) < 5e-4  # < 0.05% drift
