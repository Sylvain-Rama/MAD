from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from mad.objs.base import MovableObj, SimulationInterface
if TYPE_CHECKING:
    from mad.objs.base import BallisticObj
    from mad.objs.launchers import Launcher
    from mad.objs.planets import Planet

ComputerOrder = Enum("ComputerOrder", ["IDLE", "RELOAD", "LAUNCH", "MOVE"])


@dataclass
class ComputerCommand:
    order: ComputerOrder = ComputerOrder.IDLE
    target: MovableObj | None = None

@dataclass
class ComputerEvent:
    time: float
    command: ComputerCommand



class BattleComputer(SimulationInterface):
    def __init__(self, name: str = "BattleComputer", t: float = 0.0) -> None:
        self.name = name
        self.active = True
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.launchers: list[Launcher] = []
        self.radars: list[MovableObj] = []
        self.events: list[ComputerEvent] = []
        self.t = t

    def issue_command(self, order: ComputerOrder, target: MovableObj | None = None) -> ComputerCommand:
        return ComputerCommand(order=order, target=target)

    def send_command(self, command: ComputerCommand) -> list[BallisticObj] | None:
        """Forward *command* to all managed launchers without advancing their clock."""
        spawned: list[BallisticObj] = []
        for launcher in self.launchers:
            result = launcher.receive_orders(command)
            if result:
                spawned.extend(result)
        return spawned if spawned else None

    def update(self, dt: float, command: ComputerCommand | None = None) -> list[BallisticObj] | None:
        self.t += dt
        fired = [e for e in self.events if e.time <= self.t]
        for event in fired:
            command = event.command
            self.events.remove(event)

        if command is None:
            return None
        return self.send_command(command)

    def accelerations(self, planet: Planet) -> NDArray:
        return np.zeros(3)

    def integrate(self, dt: float, planet: Planet) -> None:
        pass  # BattleComputer is not a physical object
            
