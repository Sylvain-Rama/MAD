import math
from utils import shortest_angle
from common_schemas import Position, Velocity
from targets import Target
from planets import Planet

from dataclasses import dataclass
from enum import StrEnum


class MissileState(StrEnum):
    ASCENT = "ascent"
    CRUISE = "cruise"
    BALLISTIC = "ballistic"


@dataclass
class MissileConfig:

    thrust: float
    burn_time: float
    drag_coeff: float
    cruise_altitude: float
    target: Target
    cruise_thrust_ratio: float = 0.6
    cruise_duration: float = 2.0
    terminal_guidance: bool = True
    correction_factor: float = 0.02


class Missile:
    def __init__(self, config: MissileConfig, position: Position, velocity: Velocity):
        self.config = config
        self.pos = position
        self.vel = velocity
        self.alive: bool = True
        self.state: MissileState = MissileState.ASCENT

        self.history = [self.pos.to_tuple()]

    def __repr__(self):
        if self.alive:
            return f"Missile - {self.pos}, {self.vel}, {self.state} mode."
        else:
            return f"Crashed missile - {self.pos}."

    def get_direction(self, target: Target) -> float:
        missile_angle = math.atan2(self.pos.y, self.pos.x)
        target_angle = math.atan2(target.y, target.x)
        angle_error = shortest_angle(missile_angle, target_angle)
        return 1.0 if angle_error > 0 else -1.0

    def step(self, dt: float, planet: Planet):
        if not self.alive:
            return

        r = self.pos.distance_to_core()
        nx, ny = self.pos.x / r, self.pos.y / r  # radial
        tx, ty = -ny, nx  # tangent
        altitude = r - planet.radius

        gx, gy = planet.gravity(self.pos)
        drag_x, drag_y = planet.atmosphere(self.pos, self.vel, self.config.drag_coeff)

        thrust_x = thrust_y = 0.0

        direction = self.get_direction(self.config.target)

        if self.state == MissileState.ASCENT:
            thrust_x = nx * self.config.thrust
            thrust_y = ny * self.config.thrust
            if altitude >= self.config.cruise_altitude or self.time >= self.config.burn_time:
                self.state = MissileState.CRUISE
                self.cruise_time = 0.0

        elif self.state == MissileState.CRUISE:

            thrust_x = tx * self.config.thrust * self.config.cruise_thrust_ratio * direction
            thrust_y = ty * self.config.thrust * self.config.cruise_thrust_ratio * direction

            self.cruise_time += dt
            if self.cruise_time >= self.config.cruise_duration or self.time >= self.config.burn_time:
                self.state = MissileState.BALLISTIC

        elif self.state == MissileState.BALLISTIC and self.config.terminal_guidance:

            dx = self.config.target.x - self.pos.x
            dy = self.config.target.y - self.pos.y
            dist = math.hypot(dx, dy)
            if dist > 1e-3:
                nx_t = dx / dist
                ny_t = dy / dist

                thrust_x += nx_t * self.config.thrust * self.config.correction_factor
                thrust_y += ny_t * self.config.thrust * self.config.correction_factor

        else:
            raise ValueError(f"Missile state not recognised: {self.state}")

        ax = gx + drag_x + thrust_x
        ay = gy + drag_y + thrust_y

        self.vel.vx += ax * dt
        self.vel.vy += ay * dt
        self.pos.x += self.vel.vx * dt
        self.pos.y += self.vel.vy * dt

        self.history.append(self.pos.to_tuple())
        self.time += dt
