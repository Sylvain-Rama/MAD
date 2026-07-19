"""Tests for mad.objs.launchers and mad.objs.battle_computers."""

import numpy as np
import pytest

from mad.objs.launchers import Launcher, LauncherConfig, LauncherStates
from mad.objs.battle_computers import BattleComputer, ComputerCommand, ComputerEvent, ComputerOrder
from mad.objs.cruise_missiles import CruiseMissile, CruiseMissileConfig
from mad.objs.base import MovableObj
from mad.objs.planets import Planet, PlanetConfig
from mad.guidances.base_guidances import NoGuidanceNoThrust
from mad.configs.planets_cfg import EARTH_SETTINGS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def earth():
    cfg = dict(EARTH_SETTINGS)
    cfg["position"] = [0.0, 0.0]
    return Planet(PlanetConfig(**cfg))


@pytest.fixture
def target(earth):
    return MovableObj(position=[earth.radius + 1000.0, 0.0, 0.0], name="Target")


@pytest.fixture
def missile_config(earth, target):
    guidance = NoGuidanceNoThrust(planet=earth, target=target)
    return CruiseMissileConfig(
        mass=100.0,
        ref_radius=0.1,
        Cd=0.5,
        guidance=guidance,
        name="TestMissile",
    )


@pytest.fixture
def launcher_pos(earth):
    return np.array([earth.radius + 1000.0, 0.0, 0.0])


@pytest.fixture
def launcher(missile_config, launcher_pos):
    cfg = LauncherConfig(projectiles=missile_config, name="TestLauncher", n_projectiles=3)
    return Launcher(config=cfg, position=launcher_pos)


# ---------------------------------------------------------------------------
# LauncherConfig
# ---------------------------------------------------------------------------


class TestLauncherConfig:
    def test_create_returns_launcher(self, missile_config, launcher_pos):
        cfg = LauncherConfig(projectiles=missile_config, name="L", n_projectiles=2)
        result = cfg.create(launcher_pos)
        assert isinstance(result, Launcher)

    def test_to_dict_contains_expected_keys(self, missile_config):
        cfg = LauncherConfig(projectiles=missile_config, name="L", n_projectiles=2)
        d = cfg.to_dict
        assert "n_projectiles" in d
        assert "reload_time" in d
        assert "launch_delay" in d


# ---------------------------------------------------------------------------
# Launcher — construction
# ---------------------------------------------------------------------------


class TestLauncherInit:
    def test_initial_state_is_idle(self, launcher):
        assert launcher.state == LauncherStates.IDLE

    def test_n_projectiles_matches_config(self, launcher):
        assert launcher.n_projectiles == 3

    def test_initial_time_is_zero(self, launcher):
        assert launcher.t == pytest.approx(0.0)

    def test_position_set_correctly(self, launcher, launcher_pos):
        np.testing.assert_array_equal(launcher.position, launcher_pos)


# ---------------------------------------------------------------------------
# Launcher — launch
# ---------------------------------------------------------------------------


class TestLauncherLaunch:
    def test_launch_returns_cruise_missile(self, launcher):
        missile = launcher.launch()
        assert isinstance(missile, CruiseMissile)

    def test_launch_decrements_projectile_count(self, launcher):
        launcher.launch()
        assert launcher.n_projectiles == 2

    def test_launch_sets_last_release_time(self, launcher):
        launcher.t = 10.0
        launcher.launch()
        assert launcher.last_release_time == pytest.approx(10.0)

    def test_launch_sets_launching_state(self, launcher):
        launcher.launch()
        assert launcher.state == LauncherStates.LAUNCHING

    def test_launch_with_no_projectiles_returns_none(self, launcher):
        launcher.n_projectiles = 0
        assert launcher.launch() is None

    def test_launch_with_no_projectiles_does_not_change_state(self, launcher):
        launcher.n_projectiles = 0
        launcher.launch()
        assert launcher.state == LauncherStates.IDLE

    def test_launch_respects_launch_delay(self, missile_config, launcher_pos):
        cfg = LauncherConfig(projectiles=missile_config, n_projectiles=5, launch_delay=30.0)
        lnch = Launcher(config=cfg, position=launcher_pos)
        lnch.launch()               # first launch at t=0
        lnch.t = 10.0               # only 10 s elapsed, < 30 s delay
        assert lnch.launch() is None

    def test_launch_succeeds_after_delay_elapsed(self, missile_config, launcher_pos):
        cfg = LauncherConfig(projectiles=missile_config, n_projectiles=5, launch_delay=30.0)
        lnch = Launcher(config=cfg, position=launcher_pos)
        lnch.launch()
        lnch.t = 30.0               # exactly at the delay boundary
        second = lnch.launch()
        assert isinstance(second, CruiseMissile)

    def test_launched_missile_position_matches_launcher(self, launcher, launcher_pos):
        missile = launcher.launch()
        np.testing.assert_array_equal(missile.position, launcher_pos)

    def test_launch_overrides_target(self, launcher, earth):
        new_target = MovableObj(position=[earth.radius + 500.0, 100.0, 0.0], name="NewTarget")
        missile = launcher.launch(target=new_target)
        assert missile.guidance.target is new_target

    def test_each_launch_produces_independent_guidance(self, launcher):
        """Regression: missiles from the same launcher must not share guidance state."""
        m1 = launcher.launch()
        m2 = launcher.launch()
        assert m1.guidance is not m2.guidance

    def test_guidance_state_of_one_missile_does_not_affect_another(self, launcher):
        """Advancing one missile's guidance must not mutate the other missile's guidance."""
        m1 = launcher.launch()
        m2 = launcher.launch()
        # Manually advance m1's guidance manager index (simulating completion)
        if hasattr(m1.guidance, "current_index"):
            m1.guidance.current_index = 99
            assert m2.guidance.current_index == 0


# ---------------------------------------------------------------------------
# Launcher — reload
# ---------------------------------------------------------------------------


class TestLauncherReload:
    def test_reload_increments_count(self, missile_config, launcher_pos):
        cfg = LauncherConfig(projectiles=missile_config, n_projectiles=3, reload_time=5.0)
        lnch = Launcher(config=cfg, position=launcher_pos)
        lnch.n_projectiles = 2          # one was used
        lnch.last_reload_time = 0.0
        lnch.t = 10.0                   # reload_time has elapsed
        lnch.reload()
        assert lnch.n_projectiles == 3

    def test_reload_updates_last_reload_time(self, missile_config, launcher_pos):
        cfg = LauncherConfig(projectiles=missile_config, n_projectiles=3, reload_time=5.0)
        lnch = Launcher(config=cfg, position=launcher_pos)
        lnch.n_projectiles = 2
        lnch.last_reload_time = 0.0
        lnch.t = 10.0
        lnch.reload()
        assert lnch.last_reload_time == pytest.approx(10.0)

    def test_reload_does_not_exceed_max_capacity(self, missile_config, launcher_pos):
        cfg = LauncherConfig(projectiles=missile_config, n_projectiles=3, reload_time=0.0)
        lnch = Launcher(config=cfg, position=launcher_pos)
        lnch.reload()                   # already at max
        assert lnch.n_projectiles == 3

    def test_reload_blocked_before_reload_time(self, missile_config, launcher_pos):
        cfg = LauncherConfig(projectiles=missile_config, n_projectiles=3, reload_time=60.0)
        lnch = Launcher(config=cfg, position=launcher_pos)
        lnch.n_projectiles = 1
        lnch.last_reload_time = 0.0
        lnch.t = 10.0                   # only 10 s elapsed, < 60 s
        lnch.reload()
        assert lnch.n_projectiles == 1


# ---------------------------------------------------------------------------
# Launcher — update / receive_orders
# ---------------------------------------------------------------------------


class TestLauncherUpdate:
    def test_update_advances_time(self, launcher):
        launcher.update(5.0)
        assert launcher.t == pytest.approx(5.0)

    def test_update_without_command_returns_none(self, launcher):
        assert launcher.update(1.0) is None

    def test_receive_orders_launch_returns_missile_list(self, launcher):
        result = launcher.receive_orders(ComputerCommand(order=ComputerOrder.LAUNCH))
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], CruiseMissile)

    def test_receive_orders_idle_returns_none(self, launcher):
        result = launcher.receive_orders(ComputerCommand(order=ComputerOrder.IDLE))
        assert result is None

    def test_receive_orders_blocked_during_launch_delay(self, missile_config, launcher_pos):
        cfg = LauncherConfig(projectiles=missile_config, n_projectiles=5, launch_delay=30.0)
        lnch = Launcher(config=cfg, position=launcher_pos)
        lnch.receive_orders(ComputerCommand(order=ComputerOrder.LAUNCH))
        lnch.t = 10.0                   # still within launch delay
        result = lnch.receive_orders(ComputerCommand(order=ComputerOrder.LAUNCH))
        assert result is None


# ---------------------------------------------------------------------------
# BattleComputer — construction
# ---------------------------------------------------------------------------


class TestBattleComputerInit:
    def test_default_name(self):
        bc = BattleComputer()
        assert bc.name == "BattleComputer"

    def test_custom_name(self):
        bc = BattleComputer(name="HQ")
        assert bc.name == "HQ"

    def test_initial_time_is_zero(self):
        assert BattleComputer().t == pytest.approx(0.0)

    def test_initially_active(self):
        assert BattleComputer().active is True

    def test_no_launchers_or_events_by_default(self):
        bc = BattleComputer()
        assert bc.launchers == []
        assert bc.events == []

    def test_accelerations_are_zero(self):
        bc = BattleComputer()
        np.testing.assert_array_equal(bc.accelerations(None), np.zeros(3))

    def test_integrate_is_noop(self):
        bc = BattleComputer()
        original_pos = bc.position.copy()
        bc.integrate(1.0, None)
        np.testing.assert_array_equal(bc.position, original_pos)


# ---------------------------------------------------------------------------
# BattleComputer — update / event dispatch
# ---------------------------------------------------------------------------


class TestBattleComputerUpdate:
    def test_update_advances_time(self):
        bc = BattleComputer()
        bc.update(7.5)
        assert bc.t == pytest.approx(7.5)

    def test_update_without_events_returns_none(self):
        bc = BattleComputer()
        assert bc.update(1.0) is None

    def test_event_does_not_fire_before_its_time(self, launcher):
        bc = BattleComputer()
        bc.launchers = [launcher]
        bc.events = [ComputerEvent(time=10.0, command=ComputerCommand(order=ComputerOrder.LAUNCH))]
        bc.update(5.0)                  # t = 5 — event at t=10 should not fire
        assert launcher.n_projectiles == 3

    def test_event_fires_at_its_time(self, launcher):
        bc = BattleComputer()
        bc.launchers = [launcher]
        bc.events = [ComputerEvent(time=10.0, command=ComputerCommand(order=ComputerOrder.LAUNCH))]
        bc.update(10.0)                 # t = 10 — event fires
        assert launcher.n_projectiles == 2

    def test_event_consumed_after_firing(self, launcher):
        bc = BattleComputer()
        bc.launchers = [launcher]
        bc.events = [ComputerEvent(time=5.0, command=ComputerCommand(order=ComputerOrder.LAUNCH))]
        bc.update(10.0)
        assert bc.events == []

    def test_unfired_event_remains_in_list(self, launcher):
        bc = BattleComputer()
        bc.launchers = [launcher]
        event = ComputerEvent(time=20.0, command=ComputerCommand(order=ComputerOrder.LAUNCH))
        bc.events = [event]
        bc.update(5.0)
        assert event in bc.events

    def test_multiple_events_fire_in_separate_steps(self, launcher):
        bc = BattleComputer()
        bc.launchers = [launcher]
        bc.events = [
            ComputerEvent(time=5.0, command=ComputerCommand(order=ComputerOrder.LAUNCH)),
            ComputerEvent(time=15.0, command=ComputerCommand(order=ComputerOrder.LAUNCH)),
        ]
        bc.update(6.0)                  # fires first event → n=2
        assert launcher.n_projectiles == 2
        bc.update(10.0)                 # fires second event → n=1
        assert launcher.n_projectiles == 1


# ---------------------------------------------------------------------------
# BattleComputer — send_command
# ---------------------------------------------------------------------------


class TestBattleComputerSendCommand:
    def test_send_launch_triggers_launcher(self, launcher):
        bc = BattleComputer()
        bc.launchers = [launcher]
        result = bc.send_command(ComputerCommand(order=ComputerOrder.LAUNCH))
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], CruiseMissile)

    def test_send_command_broadcasts_to_all_launchers(self, missile_config, launcher_pos):
        l1 = Launcher(LauncherConfig(projectiles=missile_config, n_projectiles=2), launcher_pos)
        l2 = Launcher(LauncherConfig(projectiles=missile_config, n_projectiles=2), launcher_pos)
        bc = BattleComputer()
        bc.launchers = [l1, l2]
        result = bc.send_command(ComputerCommand(order=ComputerOrder.LAUNCH))
        assert result is not None
        assert len(result) == 2

    def test_send_command_with_no_launchers_returns_none(self):
        bc = BattleComputer()
        result = bc.send_command(ComputerCommand(order=ComputerOrder.LAUNCH))
        assert result is None

    def test_send_command_idle_returns_none(self, launcher):
        bc = BattleComputer()
        bc.launchers = [launcher]
        result = bc.send_command(ComputerCommand(order=ComputerOrder.IDLE))
        assert result is None
