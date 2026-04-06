import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from mad.utils import to_vec3


@dataclass
class History:
    time: list = field(default_factory=list)
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)

    def update(self, time: float, position: list, velocity: list):
        self.time.append(time)
        self.position.append(position)
        self.velocity.append(velocity)


class MovableObj:
    _id_counter: int = 0

    def __init__(
        self,
        position: list[float] | NDArray,
        velocity: list[float] | NDArray | None = None,
        name: str = "MovableObject",
    ):

        self.position = to_vec3(position)
        if velocity is not None:
            self.velocity = to_vec3(velocity)  # m/s
        else:
            self.velocity = np.zeros_like(self.position)
        self.active: bool = True
        self.name = name
        self._id = self.__class__.__name__ + f"_{self.name}_" + str(MovableObj._id_counter)
        MovableObj._id_counter += 1

    @property
    def normalize(self) -> NDArray[np.floating]:
        norm = np.linalg.norm(self.position)
        if norm < 1e-8:
            return np.zeros_like(self.position)
        return self.position / norm

    def distance(self, other: "MovableObj") -> np.floating:
        return np.linalg.norm(self.position - other.position)

    def __repr__(self):
        a = "active" if self.active else "inactive"
        return f"{self.name} at {self.position}, velocity {self.velocity}, {a}."

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MovableObj):
            return False
        return self._id == other._id


class BallisticObj(MovableObj):

    def __init__(
        self,
        position: list[float] | NDArray,
        velocity: list[float] | NDArray | None = None,
        name: str = "BallisticObject",
    ):
        super().__init__(position, velocity, name)

    pass
