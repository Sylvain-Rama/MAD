import numpy as np
from numpy.typing import NDArray
from mad.objs.constants import G


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
        i_split = len(vector) // 2
        pos, vel = vector[:i_split].tolist(), vector[i_split:].tolist()

        return cls(pos, vel, mass, name)

    def distance(self, other: "MovableObject") -> np.floating:
        return np.linalg.norm(self.position - other.position)

    def gravity_v(self, other: "MovableObject") -> np.floating:
        dist = self.distance(other)
        grav = G * self.mass * other.mass / dist**2
        return grav

    def gravity_acc(self, other: "MovableObject") -> NDArray:
        r_vec = self.position - other.position
        dist = np.linalg.norm(r_vec)
        return -G * other.mass * r_vec / dist**3

    def __repr__(self):
        a = "active" if self.active else "inactive"
        return f"{self.name} at {self.position}, velocity {self.velocity}, {a}."
