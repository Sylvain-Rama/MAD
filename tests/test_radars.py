"""Tests for mad.objs.radars — RadarConfig and Radar."""

import numpy as np
import pytest
from mad.objs.radars import Radar, RadarConfig
from mad.objs.base import MovableObj
from mad.objs.planets import Planet, PlanetConfig
from mad.configs.planets_cfg import EARTH_SETTINGS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_earth():
    """A tiny planet (radius=100 m) to keep voxel grids manageable in tests."""
    cfg = dict(EARTH_SETTINGS)
    cfg["radius"] = 100.0
    return Planet(PlanetConfig(**cfg))


@pytest.fixture
def radar(small_earth):
    surface_pos = small_earth.random_point_at_surface(altitude=5.0)
    cfg = RadarConfig(
        position=surface_pos.position,
        range=45.0,
        name="TestRadar",
        voxel_size=9,
    )
    return Radar(cfg, small_earth)


# ---------------------------------------------------------------------------
# RadarConfig
# ---------------------------------------------------------------------------


class TestRadarConfig:
    def test_defaults(self):
        cfg = RadarConfig(position=[1.0, 0.0, 0.0])
        assert cfg.name == "Radar"
        assert cfg.range == pytest.approx(450_000.0)

    def test_custom_values(self):
        cfg = RadarConfig(position=[1.0, 2.0, 3.0], name="MyRadar", range=100_000.0, voxel_size=50_000.0)
        assert cfg.name == "MyRadar"
        assert cfg.range == pytest.approx(100_000.0)
        assert cfg.voxel_size == pytest.approx(50_000.0)

    def test_to_dict(self):
        cfg = RadarConfig(position=[1.0, 0.0, 0.0], name="R", range=1000.0)
        d = cfg.to_dict
        assert "position" in d
        assert "name" in d
        assert "range" in d


# ---------------------------------------------------------------------------
# Radar initialisation
# ---------------------------------------------------------------------------


class TestRadarInit:
    def test_radar_inherits_position(self, small_earth):
        pos = [105.0, 0.0, 0.0]
        cfg = RadarConfig(position=pos, range=45.0, voxel_size=9)
        r = Radar(cfg, small_earth)
        np.testing.assert_array_almost_equal(r.position[:2], pos[:2])

    def test_detection_voxels_not_empty(self, radar):
        assert len(radar.detection_voxels) > 0

    def test_detection_voxels_all_within_range(self, radar):
        for key, strength in radar.detection_voxels.items():
            voxel_center = (np.array(key) + 0.5) * radar.voxel_size
            dist = np.linalg.norm(voxel_center - radar.position)
            assert dist <= radar.range + 1e-6  # small tolerance for float arithmetic

    def test_detection_strength_values_in_0_1(self, radar):
        strengths = list(radar.detection_voxels.values())
        assert all(0.0 <= s <= 1.0 for s in strengths)


# ---------------------------------------------------------------------------
# Radar.detect
# ---------------------------------------------------------------------------


class TestRadarDetect:
    def test_detects_nearby_target(self, small_earth, radar):
        nearby = small_earth.point_at_distance(MovableObj(radar.position), distance_km=0.02, name="Near")
        assert radar.detect(nearby) is True

    def test_does_not_detect_far_target(self, small_earth, radar):
        far = small_earth.point_at_distance(MovableObj(radar.position), distance_km=0.2, name="Far")
        assert radar.detect(far) is False

    def test_detects_self(self, radar):
        """The radar's own voxel should be within its detection grid."""
        assert radar.detect(radar) is True


# ---------------------------------------------------------------------------
# Radar.get_detection_strength
# ---------------------------------------------------------------------------


class TestRadarGetDetectionStrength:
    def test_strength_nearby_greater_than_far(self, small_earth, radar):
        nearby = small_earth.point_at_distance(MovableObj(radar.position), distance_km=0.02, name="Near")
        far = small_earth.point_at_distance(MovableObj(radar.position), distance_km=0.2, name="Far")
        s_near = radar.get_detection_strength(nearby)
        s_far = radar.get_detection_strength(far)
        assert s_near > s_far

    def test_out_of_range_returns_zero(self, small_earth, radar):
        far = small_earth.point_at_distance(MovableObj(radar.position), distance_km=0.2, name="Far")
        assert radar.get_detection_strength(far) == pytest.approx(0.0)

    def test_self_strength_positive(self, radar):
        """The radar's own voxel should have a positive detection strength."""
        s = radar.get_detection_strength(radar)
        assert s > 0.0
