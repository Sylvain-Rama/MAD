"""Propulsion components attached to `Body` objects via composition.

This mirrors the `Guidance` base class pattern in `mad.guidances.base_guidances`:
`Engine` is a small ABC defining the shared interface (`has_thrust`, `thrust_acc`,
`burned_fraction`, `update`), and concrete engines only implement what differs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mad.objs.base import Body
    from mad.objs.battle_computers import ComputerCommand


class Engine(ABC):
    """Base class for propulsion components composed onto a `Body`."""

    @property
    @abstractmethod
    def has_thrust(self) -> bool: ...

    @property
    def burned_fraction(self) -> float:
        return 0.0

    @abstractmethod
    def thrust_acc(self, body: "Body") -> float:
        """Return the acceleration (m/s²) the engine can currently deliver to `body`."""
        ...

    def update(self, body: "Body", dt: float, command: "ComputerCommand | None" = None) -> None:
        """Advance internal engine state (propellant burn, etc). No-op by default."""
        return None


class ConstantAccelerationEngine(Engine):
    """Fixed acceleration independent of body mass, e.g. a cruise-missile sustainer motor."""

    def __init__(self, thrust_acc: float, burned_fraction: float = 1.0):
        self._thrust_acc = float(thrust_acc)
        self._burned_fraction = burned_fraction

    @property
    def has_thrust(self) -> bool:
        return self._thrust_acc > 0

    @property
    def burned_fraction(self) -> float:
        return self._burned_fraction

    def thrust_acc(self, body: "Body") -> float:
        return self._thrust_acc


class RCSEngine(Engine):
    """Constant-thrust reaction-control engine for terminal-guidance payloads with unlimited propellant."""

    def __init__(self, thrust_n: float):
        self.thrust_n = thrust_n

    @property
    def has_thrust(self) -> bool:
        return self.thrust_n > 0

    @property
    def burned_fraction(self) -> float:
        # Payloads don't burn propellant; fixed at 0.5 to smoothly bias ballistic-to-terminal steering laws.
        return 0.5

    def thrust_acc(self, body: "Body") -> float:
        return self.thrust_n / body.mass
