import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field


@dataclass
class History:
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)


class MovableObject:
    def __init__(
        self, position: list[float], velocity: list[float] | None = None, mass: float = 1.0, name: str = "MovableObject"
    ):

        self.position = np.asarray(position)  # m
        if velocity:
            self.velocity = np.asarray(velocity)  # m/s
        else:
            self.velocity = np.zeros_like(self.position)
        self.mass = mass  # kg
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

    @property
    def state_vector(self):
        return np.concatenate([self.position, self.velocity])

    @classmethod
    def from_state_vector(cls, vector: NDArray, mass: float, name: str) -> "MovableObject":
        i_split = vector.shape[0] // 2
        pos, vel = vector[:i_split].tolist(), vector[i_split:].tolist()

        return cls(pos, vel, mass, name)

    def distance(self, other: "MovableObject") -> np.floating:
        return np.linalg.norm(self.position - other.position)

    def __repr__(self):
        a = "active" if self.active else "inactive"
        return f"{self.name} at {self.position}, velocity {self.velocity}, {a}."
