import math
from utils import shortest_angle


class Missile:
    def __init__(
        self,
        position,
        thrust,
        burn_time,
        drag_coeff,
        target_altitude,
        target,
        cruise_thrust_ratio=0.6,
        cruise_duration=2.0,
        terminal_guidance=True,
        name="Missile",
    ):
        self.x, self.y = position
        self.vx, self.vy = 0.0, 0.0

        self.thrust = thrust
        self.burn_time = burn_time
        self.drag_coeff = drag_coeff
        self.target_altitude = target_altitude
        self.cruise_thrust_ratio = cruise_thrust_ratio
        self.cruise_duration = cruise_duration
        self.target = target
        self.terminal_guidance = terminal_guidance

        self.time = 0.0
        self.cruise_time = 0.0
        self.alive = True
        self.state = "ASCENT"
        self.name = name

        self.xs = [self.x]
        self.ys = [self.y]

    def step(self, dt, gravity_func, atmosphere_func, planet_radius):
        if not self.alive:
            return

        r = math.hypot(self.x, self.y)
        nx, ny = self.x / r, self.y / r  # radial
        tx, ty = -ny, nx  # tangent
        altitude = r - planet_radius

        gx, gy = gravity_func(self.x, self.y)
        drag_x, drag_y = atmosphere_func(self.x, self.y, self.vx, self.vy, self.drag_coeff)

        thrust_x = thrust_y = 0.0

        if self.state == "ASCENT":
            thrust_x = nx * self.thrust
            thrust_y = ny * self.thrust
            if altitude >= self.target_altitude or self.time >= self.burn_time:
                self.state = "CRUISE"
                self.cruise_time = 0.0

        elif self.state == "CRUISE":

            missile_angle = math.atan2(self.y, self.x)
            target_angle = math.atan2(self.target.y, self.target.x)
            angle_error = shortest_angle(missile_angle, target_angle)
            direction = 1.0 if angle_error > 0 else -1.0

            print(direction)

            # Poussée tangentielle vers la cible
            thrust_x = tx * self.thrust * self.cruise_thrust_ratio * direction
            thrust_y = ty * self.thrust * self.cruise_thrust_ratio * direction

            self.cruise_time += dt
            if self.cruise_time >= self.cruise_duration or self.time >= self.burn_time:
                self.state = "BALLISTIC"

        elif self.state == "BALLISTIC" and self.terminal_guidance:
            # petit guidage terminal vers la cible
            dx = self.target.x - self.x
            dy = self.target.y - self.y
            dist = math.hypot(dx, dy)
            if dist > 1e-3:
                nx_t = dx / dist
                ny_t = dy / dist
                correction_factor = 0.02  # réglable
                thrust_x += nx_t * self.thrust * correction_factor
                thrust_y += ny_t * self.thrust * correction_factor

        # =========================
        # Intégration
        # =========================
        ax = gx + drag_x + thrust_x
        ay = gy + drag_y + thrust_y

        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Sauvegarde
        self.xs.append(self.x)
        self.ys.append(self.y)
        self.time += dt
