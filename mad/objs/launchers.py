from mad.objs.base import MovableObj, ReleasableConfig, GuidedObj, BallisticObj
from mad.guidances import Guidance, GuidanceManager
from dataclasses import dataclass, asdict
import numpy as np
from enum import Enum
from copy import deepcopy
from numpy.typing import NDArray
from mad.utils.logger import SourceLogger

logger = SourceLogger()

LauncherStates = Enum("LauncherStates", ["IDLE", "RELOAD", "LAUNCH", "MOVE"])


@dataclass
class LauncherConfig:
    projectiles: ReleasableConfig
    guidance: Guidance | GuidanceManager
    name: str = "Launcher"
    n_projectiles = 1
    speed: float = 0.0  # m / s
    reload_time: float = 0.0  # s
    launch_delay: float = 0.0  # s

    @property
    def to_dict(self):
        return asdict(self)

    def create(self, position, velocity=None, t=0.0) -> "Launcher":
        launcher = Launcher(self, position, velocity, t)
        return launcher


class Launcher(MovableObj, GuidedObj):
    def __init__(self, config: LauncherConfig, position: NDArray, velocity=None, t=0.0):
        super().__init__(position, velocity, config.name)
        self.config = config
        self.guidance = config.guidance
        self.projectiles: ReleasableConfig = config.projectiles
        self.n_projectiles = config.n_projectiles
        self.t = t
        self.last_release_time = -np.inf  # Initialize to negative infinity to allow immediate launch
        self.last_reload_time = t

    def launch(self, target: MovableObj | None = None) -> BallisticObj | None:
        if self.n_projectiles <= 0:
            logger["Launcher"].warning(f"{self.t:<.2f}s - {self.name} has no projectiles to launch!")
            return None

        if self.t - self.last_release_time < self.config.launch_delay:
            logger["Launcher"].warning(f"{self.t:<.2f}s - {self.name} cannot launch yet! Launch delay not met.")
            return None

        projectile = self.projectiles.create(self.position.copy(), self.velocity.copy(), deepcopy(self.t))
        guidance = getattr(projectile, "guidance", None)
        if isinstance(guidance, (Guidance, GuidanceManager)) and target is not None:
            guidance.target = target
        self.n_projectiles -= 1
        self.last_release_time = self.t
        return projectile

    def reload(self):
        if self.t - self.last_reload_time < self.config.reload_time:
            logger["Launcher"].warning(f"{self.t:<.2f}s - {self.name} cannot reload yet! Reload time not met.")
            return
        if self.n_projectiles < self.config.n_projectiles:
            self.n_projectiles += 1
            self.last_reload_time = self.t
            logger["Launcher"].info(f"{self.t:<.2f}s - {self.name} reloaded! Projectiles left: {self.n_projectiles}")

    def update(self, dt: float, state: LauncherStates) -> list[BallisticObj] | None:
        self.t += dt
        if state == LauncherStates.RELOAD:
            self.reload()
        elif state == LauncherStates.LAUNCH:
            projectile = self.launch()
            if projectile is not None:
                return [projectile]
        
        return None

    @property
    def thrust_acc(self) -> float:
        return self.config.speed

    @property
    def burned_fraction(self) -> float:
        return 0.0  # No fuel consumption for a launcher
