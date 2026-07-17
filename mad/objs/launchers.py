from mad.objs.base import MovableObj, ReleasableConfig, BallisticObj, SimulationInterface
from mad.objs.planets import Planet
from mad.guidances import Guidance, GuidanceManager
from mad.objs.battle_computers import ComputerOrder, ComputerCommand
from dataclasses import dataclass, asdict
import numpy as np
from enum import Enum
from copy import deepcopy
from numpy.typing import NDArray
from mad.utils.logger import SourceLogger

logger = SourceLogger()

LauncherStates = Enum("LauncherStates", ["IDLE", "RELOADING", "LAUNCHING", "MOVING"])


@dataclass
class LauncherConfig:
    projectiles: ReleasableConfig
    name: str = "Launcher"
    n_projectiles: int = 1
    speed: float = 0.0  # m / s
    reload_time: float = 0.0  # s
    launch_delay: float = 0.0  # s

    @property
    def to_dict(self):
        return asdict(self)

    def create(self, position, velocity=None, t=0.0) -> "Launcher":
        launcher = Launcher(self, position, velocity, t)
        return launcher


class Launcher(MovableObj, SimulationInterface):
    def __init__(self, config: LauncherConfig, position: NDArray, velocity=None, t=0.0):
        MovableObj.__init__(self, position, velocity, config.name)
        self.config = config
        self.projectiles: ReleasableConfig = config.projectiles
        self.n_projectiles = config.n_projectiles
        self.t = t
        self.last_release_time = -np.inf  # Initialize to negative infinity to allow immediate launch
        self.last_reload_time = t
        self._last_state: LauncherStates = LauncherStates.IDLE
        self.state: LauncherStates = LauncherStates.IDLE
        self.launch_delay = config.launch_delay

    def launch(self, target: MovableObj | None = None) -> BallisticObj | None:
        if self.n_projectiles <= 0:
            logger["Launcher"].warning(f"{self.t:<.2f}s - {self.name} has no projectiles to launch!")
            return None

        if self.t - self.last_release_time < self.config.launch_delay:
            logger["Launcher"].warning(f"{self.t:<.2f}s - {self.name} cannot launch yet! Launch delay not met.")
            return None

        self.state = LauncherStates.LAUNCHING
        projectile = self.projectiles.create(self.position.copy(), self.velocity.copy(), deepcopy(self.t))
        # This allows to either load the launcher with projectiles having their target already set
        # Or override this target at launch.
        guidance = getattr(projectile, "guidance", None)
        if isinstance(guidance, (Guidance, GuidanceManager)) and target is not None:
            guidance.target = target
        self.n_projectiles -= 1
        self.last_release_time = self.t
        return projectile

    def reload(self):
        if self._can_reload():
            self.n_projectiles += 1
            self.last_reload_time = self.t
            logger["Launcher"].info(f"{self.t:<.2f}s - {self.name} reloaded! Projectiles left: {self.n_projectiles}")

    def _can_reload(self) -> bool:
        return (self.n_projectiles < self.config.n_projectiles) and (
            self.t - self.last_reload_time >= self.config.reload_time
        )

    def update(self, dt: float, command: ComputerCommand | None = None) -> list[BallisticObj] | None:
        self.t += dt
        if command is None:
            return None
        # Timing logic for state transitions:
        # We cannot do anything if we are in the middle of launching, so we need to check that first.
        if self.state == LauncherStates.LAUNCHING and (self.t - self.last_release_time < self.config.launch_delay):
            return
        else:
            self.state = LauncherStates.IDLE

        # Second priority: firing.
        result = None
        if command is not None and command.order == ComputerOrder.LAUNCH and self.state == LauncherStates.IDLE:
            projectile = self.launch(target=command.target)
            if projectile is not None:
                result = [projectile]

            return result

        # Third, if we are idle, we can move.
        if command is not None and command.order == ComputerOrder.MOVE and self.state == LauncherStates.IDLE:
            self.state = LauncherStates.MOVING
            self.move(dt)

        # Fourth priority: reloading.
        if self._can_reload() and self.state == LauncherStates.IDLE:
            self.reload()

        return None

    def move(self, dt: float):
        # For the moment, it's not moving anywhere...
        # Needs guidance and all.
        self.position += self.velocity * dt

    @property
    def thrust_acc(self) -> float:
        return self.config.speed

    def accelerations(self, planet: Planet) -> NDArray[np.floating]:
        return np.zeros_like(self.position)  # No acceleration for a launcher

    @property
    def burned_fraction(self) -> float:
        return 0.0  # No fuel consumption for a launcher
