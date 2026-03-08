import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field


@dataclass
class History:
    time: list = field(default_factory=list)
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)

    def update(self, time: float, position: list, velocity: list):
        self.time.append(time)
        self.position.append(position)
        self.velocity.append(velocity)


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

    def central_angle(self, other: "MovableObject") -> NDArray:
        return np.arccos(np.clip(np.dot(self.norm, other.norm), -1, 1))

    def local_frame(self, target: "MovableObject") -> tuple[NDArray, NDArray]:
        r_hat = self.norm
        t_hat = target.position - np.dot(target.position, r_hat) * r_hat
        t_hat_norm = np.linalg.norm(t_hat)
        if t_hat_norm < 1e-8:
            return r_hat, np.zeros_like(self.position)  # target is directly above/below
        t_hat /= t_hat_norm
        return r_hat, t_hat

    def distance(self, other: "MovableObject") -> np.floating:
        return np.linalg.norm(self.position - other.position)

    def __repr__(self):
        a = "active" if self.active else "inactive"
        return f"{self.name} at {self.position}, velocity {self.velocity}, {a}."
