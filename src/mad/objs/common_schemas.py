import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from mad.objs.planets import Planet


@dataclass
class History:
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)


class MovableObject:
    def __init__(self, position: list[float], velocity: list[float] | None = None, name: str = "MovableObject"):

        self.position = np.asarray(position)  # m
        if velocity:
            self.velocity = np.asarray(velocity)  # m/s
        else:
            self.velocity = np.zeros_like(self.position)
        self.active: bool = True
        self.name = name

    @property
    def magnitude(self) -> np.floating:
        return np.linalg.norm(self.position)

    @property
    def norm(self) -> NDArray[np.floating]:
        if self.magnitude == 0.0:
            return np.zeros_like(self.position)
        return self.position / self.magnitude

    def distance(self, other: "MovableObject") -> np.floating:
        return np.linalg.norm(self.position - other.position)

    def __repr__(self):
        a = "active" if self.active else "inactive"
        return f"{self.name} at {self.position}, velocity {self.velocity}, {a}."


class SimulationInterface(ABC):

    @abstractmethod
    def update(self, dt: float) -> MovableObject | None:
        """This abstract method is dedicted to the update of the object itself.
        It can return other Movable objects to be able to spawn elements in the simulation."""
        pass

    @abstractmethod
    def step(self, dt: float, planet: Planet) -> None:
        """This abstract method is dedicated to the update of the object position / velocity according to the selected planet."""
        pass
